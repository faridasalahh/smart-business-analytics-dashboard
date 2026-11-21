import os, json, uuid, shutil, glob
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import json
import plotly.express as px
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents_core import (
    Orchestrator, smart_preprocess, compute_unified_outliers,
    run_predictive_anomaly, run_similarity_recommender,
    fig_to_json, fig_to_html, generate_pdf, check_llm_health,
    ask_llama, RecommenderAgent, SUPPORTED_FILE_TYPES,
    UPLOAD_DIR, CHART_DIR, PDF_MAX_CHARTS,
    build_views, is_triple_view, get_view_keys,
    VIEW_A, VIEW_B, VIEW_INTERSECTION,
    style_fig, PLOTLY_LAYOUT, CHART_PALETTE,
    HISTOGRAM_BINS, PIE_MAX_CATEGORIES, BAR_TOP_N,
)
from pandas.api.types import is_numeric_dtype

app = FastAPI(title="Smart Business Analytics Dashboard", version="4.0")

# Global catch-all: ensure every error returns JSON, never HTML
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    tb = traceback.format_exc()
    print(f"\n=== GLOBAL ERROR ===\n{tb}\n===================\n", flush=True)
    return JSONResponse(status_code=500, content={"detail": f"Server error: {type(exc).__name__}: {exc}"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

sessions: Dict[str, Dict[str, Any]] = {}


def _load_file(file_path: str) -> Optional[pd.DataFrame]:
    lower = file_path.lower()
    try:
        if lower.endswith((".csv", ".txt")): return pd.read_csv(file_path)
        if lower.endswith((".xlsx", ".xls")): return pd.read_excel(file_path)
        if lower.endswith(".json"): return pd.read_json(file_path)
        if lower.endswith(".xml"): return pd.read_xml(file_path)
        if lower.endswith(".parquet"): return pd.read_parquet(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load {file_path}: {str(e)}")
    return None


@app.get("/api/health")
def health():
    return {"status": "ok", "llm": check_llm_health(), "version": "4.1-insight-rebuild"}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    uploaded = {}
    for file in files:
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in SUPPORTED_FILE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        uploaded[file.filename] = file_path

    sessions[session_id] = {
        "files": uploaded, "datasets": {}, "preprocessed": {},
        "prep_reports": {}, "views": {}, "view_titles": {},
        "shared_cols": [], "results_by_view": {}
    }
    return {"session_id": session_id, "files": list(uploaded.keys())}


@app.post("/api/preprocess")
def preprocess(session_id: str = Form(...)):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    raw = {}
    for name, path in sess["files"].items():
        df = _load_file(path)
        if df is not None: raw[name] = df
    if not raw: raise HTTPException(400, "No valid datasets")

    preprocessed, reports = {}, {}
    for name, df in raw.items():
        df_c, report = smart_preprocess(df)
        preprocessed[name] = df_c
        reports[name] = report
    sess["datasets"] = raw
    sess["preprocessed"] = preprocessed
    sess["prep_reports"] = reports

    # Build triple-views
    views, view_titles, shared_cols = build_views(preprocessed)
    sess["views"] = views
    sess["view_titles"] = view_titles
    sess["shared_cols"] = shared_cols

    summary = {}
    for name, df in preprocessed.items():
        summary[name] = {"rows": df.shape[0], "cols": df.shape[1], "columns": list(df.columns)}
    return {
        "summary": summary,
        "reports": reports,
        "views": {k: list(v.keys()) for k, v in views.items()},
        "view_titles": view_titles,
        "shared_cols": shared_cols,
        "triple_mode": len(preprocessed) == 2,
    }


@app.get("/api/views/{session_id}")
def get_views(session_id: str):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    views = sess.get("views", {})
    return {
        "views": {k: list(v.keys()) for k, v in views.items()},
        "view_titles": sess.get("view_titles", {}),
        "shared_cols": sess.get("shared_cols", []),
        "triple_mode": is_triple_view(views),
        "view_keys": get_view_keys(views),
    }


@app.get("/api/data/{session_id}/preview")
def data_preview(session_id: str, dataset: str = "", rows: int = 10):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    datasets = sess.get("preprocessed") or sess.get("datasets", {})
    if dataset:
        df = datasets.get(dataset)
        if df is None: raise HTTPException(404, f"Dataset {dataset} not found")
        return {"columns": list(df.columns), "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                "data": json.loads(df.head(rows).to_json(orient="records")),
                "shape": list(df.shape), "name": dataset}
    return {name: {"columns": list(df.columns), "shape": list(df.shape), "data": json.loads(df.head(rows).to_json(orient="records"))}
            for name, df in datasets.items()}


@app.post("/api/pipeline/run")
def run_pipeline(session_id: str = Form(...), run_insight: bool = Form(False)):
    try:
        sess = sessions.get(session_id)
        if not sess: raise HTTPException(404, "Session not found")
        views = sess.get("views", {})
        if not views: raise HTTPException(400, "No views found. Upload and preprocess first.")
        view_titles = sess.get("view_titles", {})

        results_by_view = {}
        view_keys = list(views.keys())
        orch = Orchestrator()

        print(f"\n--- Pipeline starting: {len(view_keys)} view(s), run_insight={run_insight} ---", flush=True)

        for vk in view_keys:
            view_datasets = views[vk]
            results = orch.run(view_datasets)
            if run_insight:
                # For intersection view pass ALL raw datasets
                # For individual views pass only the matching raw dataset
                if vk == "intersection":
                    view_raw = sess.get("datasets", {})
                else:
                    view_raw = {
                        name: sess["datasets"][name] 
                        for name in sess.get("datasets", {}) 
                        if name in view_datasets or any(
                            name.lower() in k.lower() or k.lower() in name.lower() 
                            for k in view_datasets.keys()
                        )
                    }
                    # Fallback — if view_raw is empty use all datasets
                    if not view_raw:
                        view_raw = sess.get("datasets", {})

                insight_result = orch.run_insight(results, view_datasets, view_raw)
                results["insight"] = insight_result
                rec = RecommenderAgent().run(
                    insight_report=insight_result["output"],
                )
                results["recommender"] = rec
            results_by_view[vk] = results

        sess["results_by_view"] = results_by_view
        return {
            "status": "complete",
            "views": list(results_by_view.keys()),
            "view_titles": view_titles,
            "triple_mode": is_triple_view(views),
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n{'='*60}\nPIPELINE CRASHED!\n{tb}\n{'='*60}", flush=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {type(e).__name__}: {e}")


@app.get("/api/results/{session_id}")
def get_results(session_id: str):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    results_by_view = sess.get("results_by_view", {})
    if not results_by_view: return {"status": "no_results"}

    views = sess.get("views", {})
    view_titles = sess.get("view_titles", {})

    out = {}
    for vk, results in results_by_view.items():
        logs = {}
        for agent in ["retriever", "planner", "stylist", "visualizer", "critic"]:
            if agent in results:
                logs[agent] = results[agent].get("log", "")

        # Get datasets for this view
        view_ds = views.get(vk, {})
        kpi = {}
        for name, df in view_ds.items():
            num = df.select_dtypes(include=np.number)
            kpi[name] = {
                "rows": df.shape[0], "cols": df.shape[1],
                "numeric_cols": len(num.columns),
                "nulls": int(df.isnull().sum().sum()),
            }

        outliers = compute_unified_outliers(view_ds)
        out[vk] = {
            "kpi": kpi,
            "logs": logs,
            "outliers": outliers,
            "has_insight": "insight" in results,
            "has_recommender": "recommender" in results,
            "title": view_titles.get(vk, vk),
        }
    return {
        "status": "complete",
        "views": out,
        "view_keys": list(results_by_view.keys()),
        "view_titles": view_titles,
        "triple_mode": is_triple_view(views),
        "shared_cols": sess.get("shared_cols", []),
    }


@app.get("/api/charts/{session_id}")
def get_charts(session_id: str):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    results_by_view = sess.get("results_by_view", {})
    view_titles = sess.get("view_titles", {})
    out = {}

    for vk, results in results_by_view.items():
        viz = results.get("visualizer", {}).get("output", {})
        charts_data = {}
        for ds_name, col_figs in viz.items():
            charts_data[ds_name] = {}
            for col, fig_data in col_figs.items():
                charts_data[ds_name][col] = {
                    "json": fig_to_json(fig_data["fig"]),
                    "type": fig_data["type"],
                    "original_type": fig_data.get("original_type", fig_data["type"]),
                    "guard_warning": fig_data.get("guard_warning"),
                }
        out[vk] = {"charts": charts_data, "title": view_titles.get(vk, vk)}

    return {"views": out, "view_keys": list(results_by_view.keys())}


@app.get("/api/insight/{session_id}")
def get_insight(session_id: str):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    results_by_view = sess.get("results_by_view", {})
    out = {}
    for vk, results in results_by_view.items():
        insight = results.get("insight", {}).get("output", "")
        out[vk] = {"insight": insight}
    return {"views": out}


@app.get("/api/recommender/{session_id}")
def get_recommender(session_id: str):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    results_by_view = sess.get("results_by_view", {})
    out = {}
    for vk, results in results_by_view.items():
        recommender = results.get("recommender", {}).get("output", "")
        out[vk] = {"recommender": recommender}
    return {"views": out}


@app.post("/api/anomaly/run")
def run_anomaly(session_id: str = Form(...), view_key: str = Form(""), horizon: int = Form(30)):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    views = sess.get("views", {})
    datasets = views.get(view_key, {}) if view_key else (sess.get("preprocessed") or sess.get("datasets", {}))
    if not datasets: raise HTTPException(400, "No data for this view")
    result = run_predictive_anomaly(datasets, horizon=horizon)
    if "_anomaly_cache" not in sess:
        sess["_anomaly_cache"] = {}
    sess["_anomaly_cache"][view_key] = result
    return {
        "method": result["method"],
        "predicted_anomalies": result["predicted_anomalies"],
        "horizon": result["horizon"],
        "confidence": result["confidence"],
        "per_dataset": result["per_dataset"],
    }


@app.post("/api/anomaly/explain")
def explain_anomaly(session_id: str = Form(...), view_key: str = Form("")):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    cache = sess.get("_anomaly_cache", {})
    if view_key in cache:
        result = cache[view_key]
    else:
        views = sess.get("views", {})
        datasets = views.get(view_key, {}) if view_key else (sess.get("preprocessed") or sess.get("datasets", {}))
        if not datasets: return {"explanation": "No data found."}
        result = run_predictive_anomaly(datasets)
    combined = result.get("_combined")
    feat_cols = result.get("_feature_cols", [])
    method = result["method"]
    horizon = result["horizon"]
    predicted = result["predicted_anomalies"]
    per_dataset = result["per_dataset"]
    if combined is not None and not combined.empty and feat_cols:
        valid_feat = [c for c in feat_cols if c in combined.columns]
        normal_rows = combined[combined["Status"] == "Normal"]
        anomaly_rows = combined[combined["Status"] == "Anomaly"]
        normal_stats = normal_rows[valid_feat].describe().to_string() if valid_feat and not normal_rows.empty else "N/A"
        anomaly_sample = anomaly_rows[valid_feat].head(5).to_string() if valid_feat and not anomaly_rows.empty else "N/A"
    else:
        normal_stats = "N/A"
        anomaly_sample = "N/A"
    per_ds_summary = "\n".join(
        f"  - {d['dataset']}: {d['predicted_anomalies']} predicted anomalies "
        f"({'Prophet time-series' if d['method'] == 'prophet' else 'Bootstrap simulation'}, "
        f"time column: {d.get('time_column', 'none')})"
        for d in per_dataset
    )
    prompt = f"""You are a ruthless, highly decisive Chief Operating Officer reporting to the Board of Directors.
You DO NOT use weak language. You DO NOT guess. You state physical facts.

A predictive system analysed '{method}' with horizon '{horizon}'.
Total predicted anomalies: {predicted}. Confidence: {result['confidence']}.
Per-dataset details: {per_ds_summary}
Normal stats: {normal_stats}
Anomaly sample: {anomaly_sample}

Write a 200-word executive brief with these EXACT sections:
1) Prediction Meaning: What does an anomaly in these metrics physically mean for the facility's operations?
2) Riskiest Metrics: Which specific raw volume or consumption metrics are most volatile? (IGNORE any rate/conversion metrics).
3) Root Causes: Provide 3 VERIFIED physical reasons for these spikes (e.g., turbine degradation, boiler leaks, unauthorized overtime). Do NOT mention supply chain or stockpiling as a cause for consumption.
4) Action: Provide ONE aggressive, immediate operational order to the floor managers.

ABSOLUTE RESTRICTIONS (VIOLATION WILL RESULT IN TERMINATION):
- DO NOT use the words "may", "might", "could", "suggests", "potentially", or "possibly". 
- Use ONLY definitive verbs: "indicates", "proves", "is caused by", "requires".
- DO NOT blame algorithms or forecasting models.
"""
    explanation = ask_llama(prompt, "Predictive Anomaly Explainer")
    return {
        "explanation": explanation,
        "method": method,
        "predicted_anomalies": predicted,
        "horizon": horizon,
        "confidence": result["confidence"],
    }


@app.post("/api/recommender/similar")
def find_similar(session_id: str = Form(...), dataset: str = Form(...), row_index: int = Form(0), top_n: int = Form(5)):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    datasets = sess.get("preprocessed") or sess.get("datasets", {})
    df = datasets.get(dataset)
    if df is None: raise HTTPException(404, f"Dataset {dataset} not found")
    result = run_similarity_recommender(df, row_index, top_n)
    if result is None: return {"error": "Not enough numeric data"}
    return {"matches": json.loads(result.to_json(orient="records"))}


@app.post("/api/pdf/generate")
def generate_pdf_report(session_id: str = Form(...), view_key: str = Form("")):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    results_by_view = sess.get("results_by_view", {})
    if view_key and view_key in results_by_view:
        results = results_by_view[view_key]
    else:
        # Use first view with results
        results = next(iter(results_by_view.values())) if results_by_view else {}

    insight = results.get("insight", {}).get("output", "") if results else ""
    rec = results.get("recommender", {}).get("output", "") if results else ""
    viz = results.get("visualizer", {}).get("output", {}) if results else {}

    chart_save_dir = os.path.join(CHART_DIR, session_id)
    os.makedirs(chart_save_dir, exist_ok=True)
    count = 0
    for ds_name, col_figs in viz.items():
        for col, fig_data in col_figs.items():
            if count >= PDF_MAX_CHARTS: break
            try:
                fig_data["fig"].write_image(os.path.join(chart_save_dir, f"chart_{count}.png"),
                                            width=800, height=450, engine="kaleido")
                count += 1
            except: pass
        if count >= PDF_MAX_CHARTS: break

    pdf_path = generate_pdf(insight, rec, chart_save_dir)
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type="application/pdf", filename="Executive_Report.pdf")
    raise HTTPException(500, "PDF generation failed")


@app.get("/api/llm/status")
def llm_status():
    status = check_llm_health()
    return {"online": status == "online", "status": status}


@app.post("/api/ask")
def ask_llm(prompt: str = Form(...)):
    return {"response": ask_llama(prompt, "API")}

# ─── Interactive Chart Builder ───

def build_interactive_chart(df, x_column, y_column, chart_type, title=None, agg_func="count"):
    """Build a Plotly figure from user-chosen columns and chart type.

    Parameters
    ----------
    agg_func : str
        How to aggregate y_column when grouped by x_column.
        One of: count | sum | mean | median | min | max
        Ignored for scatter/box/histogram where raw values are used.
    """
    from agents_core import VisualizerAgent
    VALID_AGG = {"count", "sum", "mean", "median", "min", "max"}
    if agg_func not in VALID_AGG:
        agg_func = "mean"

    if chart_type not in ("histogram", "box", "bar", "line", "scatter", "pie"):
        raise HTTPException(400, f"Unsupported chart type: {chart_type}")

    style = {"chart_type": chart_type, "color": CHART_PALETTE[0]}
    fig = VisualizerAgent._make_fig(df, x_column, style,
                                    y_col=y_column or None,
                                    agg_func=agg_func)
    if title:
        fig = style_fig(fig, title=title)
    return fig


@app.get("/api/data/{session_id}/columns")
def get_columns(session_id: str):
    """Return all column names + types per dataset.
    Falls back to preprocessed/raw datasets if pipeline views not yet available.
    """
    sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    views = sess.get("views", {})
    view_titles = sess.get("view_titles", {})
    out = {}

    if views:
        # Pipeline has run — use views (preferred, has proper grouping)
        for vk, view_datasets in views.items():
            ds_info = {}
            for name, df in view_datasets.items():
                cols = []
                for c in df.columns:
                    if c == "_source":
                        continue
                    is_num = is_numeric_dtype(df[c])
                    cols.append({"name": c, "type": "numeric" if is_num else "categorical"})
                ds_info[name] = {"columns": cols, "shape": list(df.shape)}
            out[vk] = {"datasets": ds_info, "title": view_titles.get(vk, vk)}
    else:
        # Pipeline not yet run — use preprocessed or raw uploaded datasets
        fallback = sess.get("preprocessed") or sess.get("datasets") or {}
        if fallback:
            ds_info = {}
            for name, df in fallback.items():
                cols = []
                for c in df.columns:
                    if c == "_source":
                        continue
                    is_num = is_numeric_dtype(df[c])
                    cols.append({"name": c, "type": "numeric" if is_num else "categorical"})
                ds_info[name] = {"columns": cols, "shape": list(df.shape)}
            out["unified"] = {"datasets": ds_info, "title": "Uploaded Data"}

    return {"views": out, "view_keys": get_view_keys(views) if views else ["unified"]}


@app.post("/api/chart/build")
def build_chart(
    session_id: str = Form(...),
    dataset_name: str = Form(...),
    x_column: str = Form(...),
    y_column: str = Form(""),
    chart_type: str = Form(...),
    agg_func: str = Form("count"),
    title: str = Form(""),
):
    """Build a custom Plotly chart from user-chosen columns.

    agg_func controls how y_column is aggregated per x group:
      count | sum | mean | median | min | max
    """
    sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    # Search all view datasets for the requested dataset
    views = sess.get("views", {})
    df = None
    for vk, view_datasets in views.items():
        if dataset_name in view_datasets:
            df = view_datasets[dataset_name]
            break
    if df is None:
        datasets = sess.get("preprocessed") or sess.get("datasets", {})
        df = datasets.get(dataset_name)
    if df is None:
        raise HTTPException(404, f"Dataset '{dataset_name}' not found in any view")
    # Validate columns exist
    if x_column not in df.columns:
        raise HTTPException(400, f"X column '{x_column}' not found in dataset '{dataset_name}'")
    if y_column and y_column not in df.columns:
        raise HTTPException(400, f"Y column '{y_column}' not found in dataset '{dataset_name}'")
    if chart_type not in ("histogram", "box", "bar", "line", "scatter", "pie"):
        raise HTTPException(400, f"Unsupported chart type: {chart_type}")
    fig = build_interactive_chart(df, x_column, y_column or None, chart_type, title or None, agg_func=agg_func)
    return {"fig": fig_to_json(fig), "x": x_column, "y": y_column, "type": chart_type,
            "dataset": dataset_name, "agg_func": agg_func}


# Serve frontend as static files — mount AFTER all API routes so they take priority
_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

@app.get("/api/kpis")
async def get_kpis(dataset_name: str):
    """
    100% Accurate KPI Calculator using raw Pandas math (No LLM involved)
    """
    datasets = orchestrator.datasets  
    if dataset_name not in datasets:
        raise HTTPException(404, f"Dataset '{dataset_name}' not found")
        
    df = datasets[dataset_name].copy()
    
    # 1. فلترة العواميد (تجاهل معدل التحويل)
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if "rate" not in c.lower()]
    
    # التأكد من وجود عمود التاريخ
    time_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    
    kpis = {}
    
    # لو عندنا عمود استهلاك الطاقة (TOE Consumption)
    target_col = None
    for col in num_cols:
        if "toe" in col.lower() and "consumption" in col.lower():
            target_col = col
            break
            
    # لو ملقاش TOE، ياخد عمود القيمة (Value)
    if not target_col and num_cols:
        target_col = [c for c in num_cols if "value" in c.lower()][0] if any("value" in c.lower() for c in num_cols) else num_cols[0]

    if target_col and time_cols:
        time_col = time_cols[0]
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        
        # تجميع الداتا بالشهور عشان الحسابات تطلع صحيحة
        monthly_data = df.groupby(df[time_col].dt.to_period("M"))[target_col].sum().reset_index()
        monthly_data = monthly_data.sort_values(time_col)
        
        # --- حساب الأرقام الدقيقة ---
        
        # 1. إجمالي الاستهلاك (Total)
        total_value = float(monthly_data[target_col].sum())
        
        # 2. متوسط الاستهلاك الشهري (Monthly Average)
        avg_monthly = float(monthly_data[target_col].mean())
        
        # 3. أعلى شهر استهلاك (Max Peak)
        max_value = float(monthly_data[target_col].max())
        
        # 4. نسبة النمو (YoY Growth) - مقارنة أول سنة بآخر سنة
        yearly_data = df.groupby(df[time_col].dt.year)[target_col].sum().reset_index()
        if len(yearly_data) >= 2:
            first_year_val = yearly_data.iloc[0][target_col]
            last_year_val = yearly_data.iloc[-1][target_col]
            growth_rate = ((last_year_val - first_year_val) / first_year_val) * 100 if first_year_val > 0 else 0
        else:
            growth_rate = 0.0

        # تنسيق الأرقام عشان ترجع بشكل شيك للـ Frontend
        kpis = {
            "Total Volume": f"{total_value:,.2f}",
            "Monthly Average": f"{avg_monthly:,.2f}",
            "Peak Consumption": f"{max_value:,.2f}",
            "Overall Growth": f"{growth_rate:,.1f}%"
        }
    else:
        kpis = {"Status": "No numerical/time data available for KPIs"}

    return {"dataset": dataset_name, "kpis": kpis}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
