import json, os, operator, re, requests, textwrap, subprocess, sys, io, base64, math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, TypedDict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from langgraph.graph import StateGraph, END
from typing import TypedDict
from fpdf import FPDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#f8fafc", size=12),
    title=dict(font=dict(family="Space Grotesk, sans-serif", size=16, color="#f8fafc"), x=0.02, xanchor="left"),
    xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", linecolor="rgba(148, 163, 184, 0.2)", zerolinecolor="rgba(148, 163, 184, 0.15)",
               tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#94a3b8"),
               title=dict(font=dict(family="Inter", size=12, color="#cbd5e1"))),
    yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", linecolor="rgba(148, 163, 184, 0.2)", zerolinecolor="rgba(148, 163, 184, 0.15)",
               tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#94a3b8"),
               title=dict(font=dict(family="Inter", size=12, color="#cbd5e1"))),
    legend=dict(font=dict(family="Inter", size=11, color="#cbd5e1"), bgcolor="rgba(13, 16, 36, 0.6)",
                bordercolor="rgba(148, 163, 184, 0.15)", borderwidth=1),
    margin=dict(l=50, r=20, t=50, b=50),
    hoverlabel=dict(bgcolor="#161938", bordercolor="#00e5ff", font=dict(family="JetBrains Mono", size=12, color="#f8fafc")),
)


def style_fig(fig, title=None):
    layout_args = dict(PLOTLY_LAYOUT)
    if title is not None:
        layout_args = {**layout_args, "title": {**PLOTLY_LAYOUT["title"], "text": title}}
    fig.update_layout(**layout_args)
    return fig


def ask_llama(prompt: str, agent_name: str = "") -> str:
    try:
        res = requests.post(
            LLM_ENDPOINT,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=LLM_TIMEOUT_SEC,
        )
        data = res.json()
        if "error" in data:
            err = data["error"]
            if "memory" in err.lower() or "available" in err.lower():
                return f"[{agent_name}] Model '{LLM_MODEL}' needs more memory than available ({err}). Try a smaller model: run `ollama pull llama3.2:1b` then set LLM_MODEL=llama3.2:1b in .env"
            return f"[{agent_name}] API Error: {err}"
        return data.get("response", "No response received.")
    except requests.exceptions.Timeout:
        return f"[{agent_name}] AI Engine timed out after {LLM_TIMEOUT_SEC}s."
    except requests.exceptions.ConnectionError:
        return f"[{agent_name}] AI Engine Offline"
    except Exception as e:
        return f"[{agent_name}] Error: {type(e).__name__}: {e}"


def check_llm_health() -> str:
    """Returns 'online' if Ollama is reachable, else 'offline'."""
    try:
        r = requests.get(LLM_TAGS_ENDPOINT, timeout=LLM_HEALTHCHECK_SEC)
        return "online" if r.status_code == 200 else "offline"
    except requests.exceptions.RequestException:
        return "offline"


class RetrieverAgent:
    name = "Retriever Agent"
    icon = "🔍"

    @staticmethod
    def _classify_distribution(skew: float) -> str:
        if abs(skew) < SKEW_NORMAL_THRESHOLD:
            return "roughly normal"
        return "right-skewed" if skew > 0 else "left-skewed"

    def run(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        summaries, log_lines = {}, [f"{self.icon} {self.name} activated"]
        for name, df in datasets.items():
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            null_total = int(df.isnull().sum().sum())
            dist_notes = {col: self._classify_distribution(df[col].skew()) for col in num_cols}
            summaries[name] = {
                "shape": df.shape, "num_cols": num_cols, "cat_cols": cat_cols,
                "null_total": null_total, "dist_notes": dist_notes,
                "describe": df[num_cols].describe().to_string() if num_cols else "No numeric columns",
                "sample": df.head(SAMPLE_PREVIEW_ROWS).to_string(),
            }
            log_lines.append(f"  📂 {name}: {df.shape[0]} rows x {df.shape[1]} cols | {len(num_cols)} numeric | {len(cat_cols)} categorical | {null_total} nulls")
        return {"output": summaries, "log": "\n".join(log_lines)}


class PlannerAgent:
    name = "Planner Agent"
    icon = "🎯"

    @staticmethod
    def _build_steps(ctx):
        steps = []
        n_num, n_cat = len(ctx.get("num_cols", [])), len(ctx.get("cat_cols", []))
        if ctx.get("null_total", 0) > 0: steps.append("handle_missing_values")
        if n_num > 0: steps.extend(["descriptive_statistics", "outlier_detection"])
        if n_num >= 2: steps.extend(["correlation_analysis", "anomaly_detection"])
        if n_cat > 0: steps.append("categorical_analysis")
        steps.extend(["visualization", "ai_insight_generation"])
        return steps

    def run(self, retriever_output):
        log_lines = [f"{self.icon} {self.name} activated"]
        plans = {}
        for name, ctx in retriever_output.items():
            steps = self._build_steps(ctx)
            prompt = f"Dataset: shape={ctx['shape']}, numeric={ctx['num_cols']}, categorical={ctx['cat_cols']}, distributions={ctx['dist_notes']}. Briefly explain analytical priority (2-3 sentences)."
            rationale = ask_llama(prompt, self.name)
            plans[name] = {"steps": steps, "rationale": rationale}
            log_lines.append(f"  📂 {name}: {' → '.join(steps)}")
        return {"output": plans, "log": "\n".join(log_lines)}


class StylistAgent:
    name = "Stylist Agent"
    icon = "🎨"
    COLORS = CHART_PALETTE
    VALID_CHARTS = VALID_CHART_TYPES

    @staticmethod
    def _fallback_chart(col_type, dist):
        if col_type == "categorical": return "pie", "fallback: categorical → pie"
        if dist in ("right-skewed", "left-skewed"): return "histogram", f"fallback: {dist} → histogram"
        if dist == "roughly normal": return "box", "fallback: normal → box"
        return "bar", "fallback: default → bar"

    def _ask_llm_for_chart(self, col, dataset_name, col_type, dist, nuniq, n_rows, sample_vals, plan_steps):
        prompt = f"""Choose the BEST chart type for column '{col}' (dataset: '{dataset_name}').
Type: {col_type}, Distribution: {dist}, Unique: {nuniq}/{n_rows}, Samples: {sample_vals}
Available: histogram, box, bar, line, scatter, pie
Reply EXACTLY:
CHART: <type>
REASON: <one sentence>"""
        response = ask_llama(prompt, self.name)
        chart_type, reason = None, "LLM decision"
        for line in response.strip().splitlines():
            ls = line.strip()
            if ls.upper().startswith("CHART:"):
                raw = ls.split(":", 1)[1].strip().lower()
                for v in self.VALID_CHARTS:
                    if v in raw: chart_type = v; break
            elif ls.upper().startswith("REASON:"):
                reason = ls.split(":", 1)[1].strip()
        if chart_type not in self.VALID_CHARTS:
            chart_type, r2 = self._fallback_chart(col_type, dist)
            reason = f"[fallback] {r2}"
        return chart_type, reason

    @staticmethod
    def _real_stats(df, col):
        if df is None or col not in df.columns: return 0, 0, []
        try: return int(df[col].nunique()), len(df), df[col].dropna().head(SAMPLE_PREVIEW_ROWS).tolist()
        except Exception: return 0, 0, []

    def run(self, retriever_output, planner_output, datasets):
        log_lines = [f"{self.icon} {self.name} activated"]
        styles, color_idx = {}, 0
        for name, ctx in retriever_output.items():
            col_styles = {}
            plan_steps = planner_output.get(name, {}).get("steps", [])
            df = datasets.get(name)
            for col in ctx["num_cols"]:
                color = self.COLORS[color_idx % len(self.COLORS)]; color_idx += 1
                dist = ctx["dist_notes"].get(col, "unknown")
                nuniq, n_rows, sv = self._real_stats(df, col)
                ct, reason = self._ask_llm_for_chart(col, name, "numeric", dist, nuniq, n_rows, sv, plan_steps)
                col_styles[col] = {"chart_type": ct, "color": color, "reason": reason, "decided_by": "LLM"}
                log_lines.append(f"  📂 {name} → '{col}' [{dist}]: {ct} — {reason}")
            for col in ctx["cat_cols"]:
                color = self.COLORS[color_idx % len(self.COLORS)]; color_idx += 1
                nuniq, n_rows, sv = self._real_stats(df, col)
                ct, reason = self._ask_llm_for_chart(col, name, "categorical", "N/A", nuniq, n_rows, sv, plan_steps)
                col_styles[col] = {"chart_type": ct, "color": color, "reason": reason, "decided_by": "LLM"}
                log_lines.append(f"  📂 {name} → '{col}' [categorical]: {ct} — {reason}")
            styles[name] = col_styles
        return {"output": styles, "log": "\n".join(log_lines)}


class VisualizerAgent:
    name = "Visualizer Agent"
    icon = "📈"

    @staticmethod
    def _is_time_like(df):
        if isinstance(df.index, pd.DatetimeIndex): return True
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]): return True
        return False

    @classmethod
    def _apply_safety_guards(cls, df, col, ctype):
        try: nuniq = int(df[col].nunique())
        except Exception: return ctype, None
        is_num = is_numeric_dtype(df[col])
        time_ordered = cls._is_time_like(df)
        if ctype == "pie" and nuniq > PIE_MAX_CATEGORIES:
            return "bar", f"Pie with {nuniq} cats unreadable → bar"
        if ctype in ("pie", "bar") and is_num and nuniq > BAR_MAX_CATEGORIES:
            return "histogram", f"{ctype} on numeric with {nuniq} unique → histogram"
        if ctype == "histogram" and nuniq < HISTOGRAM_MIN_UNIQUE:
            return "bar", f"Histogram on {nuniq} unique values → bar"
        if ctype == "line" and not time_ordered and not is_num:
            return "bar", "Line on categorical without time index → bar"
        # scatter: always allow when user picks it — no time restriction
        if ctype in ("histogram", "box") and not is_num:
            return "bar", f"{ctype} needs numeric data → bar"
        return ctype, None

    @staticmethod
    def _make_fig(df, col, style, y_col=None, agg_func="count"):
        """
        Build a Plotly figure for a single column (auto-pipeline) or an X/Y pair
        (interactive chart builder).

        Parameters
        ----------
        df        : DataFrame
        col       : X-axis column (always required)
        style     : dict with keys chart_type / color
        y_col     : optional Y-axis column; when supplied the chart shows a real
                    numeric value instead of a count
        agg_func  : aggregation applied when grouping X → Y
                    one of "count" | "sum" | "mean" | "median" | "min" | "max"
                    Only used when y_col is set and chart type groups data (bar, pie, line).
        """
        ctype, color = style["chart_type"], style["color"]

        # ── helpers ──────────────────────────────────────────────────────────
        AGG_MAP = {
            "count":  "count",
            "sum":    "sum",
            "mean":   "mean",
            "median": "median",
            "min":    "min",
            "max":    "max",
        }
        safe_agg = AGG_MAP.get(agg_func, "mean")

        def _agg_df(group_col, value_col, limit=None):
            """Group group_col, aggregate value_col, return sorted DataFrame."""
            grp = df.groupby(group_col)[value_col]
            if safe_agg == "count":
                agg = grp.count()
            elif safe_agg == "sum":
                agg = grp.sum()
            elif safe_agg == "mean":
                agg = grp.mean()
            elif safe_agg == "median":
                agg = grp.median()
            elif safe_agg == "min":
                agg = grp.min()
            else:  # max
                agg = grp.max()
            agg = agg.sort_values(ascending=False)
            if limit:
                agg = agg.head(limit)
            return agg.reset_index()

        def _y_label():
            if not y_col:
                return "Count"
            return f"{safe_agg.capitalize()} of {y_col}"

        # ── histogram ────────────────────────────────────────────────────────
        if ctype == "histogram":
            if y_col and is_numeric_dtype(df[y_col]):
                # Overlay two numeric distributions
                fig = px.histogram(df, x=col, y=y_col, nbins=HISTOGRAM_BINS,
                                   histfunc=safe_agg if safe_agg != "median" else "avg",
                                   color_discrete_sequence=[color])
                fig.update_layout(yaxis_title=_y_label())
            else:
                fig = px.histogram(df, x=col, nbins=HISTOGRAM_BINS,
                                   color_discrete_sequence=[color])
            fig.update_traces(marker_line_width=0, opacity=0.85)

        # ── box ──────────────────────────────────────────────────────────────
        elif ctype == "box":
            if y_col and is_numeric_dtype(df[y_col]):
                # Group distribution: X = category, Y = numeric values
                fig = px.box(df, x=col, y=y_col, color_discrete_sequence=[color])
                fig.update_layout(yaxis_title=y_col)
            else:
                fig = px.box(df, y=col, color_discrete_sequence=[color])
            fig.update_traces(marker_size=4, line_width=2)

        # ── pie ──────────────────────────────────────────────────────────────
        elif ctype == "pie":
            if y_col and is_numeric_dtype(df[y_col]):
                agg_data = _agg_df(col, y_col, PIE_MAX_CATEGORIES)
                fig = px.pie(agg_data, names=col, values=y_col,
                             color_discrete_sequence=CHART_PALETTE, hole=0.45)
                fig.update_layout(title_text=f"{_y_label()} by {col}")
            else:
                vc = df[col].value_counts().head(PIE_MAX_CATEGORIES).reset_index()
                vc.columns = [col, "count"]
                fig = px.pie(vc, names=col, values="count",
                             color_discrete_sequence=CHART_PALETTE, hole=0.45)
            fig.update_traces(textfont=dict(family="Inter", size=12, color="#f8fafc"),
                              marker=dict(line=dict(color="#070912", width=2)))

        # ── line ─────────────────────────────────────────────────────────────
        elif ctype == "line":
            if y_col and is_numeric_dtype(df[y_col]):
                if is_numeric_dtype(df[col]):
                    # Both numeric: simple X vs Y line
                    sorted_df = df[[col, y_col]].dropna().sort_values(col)
                    fig = px.line(sorted_df, x=col, y=y_col,
                                  color_discrete_sequence=[color])
                else:
                    # Categorical X: aggregate then plot
                    agg_data = _agg_df(col, y_col)
                    fig = px.line(agg_data, x=col, y=y_col,
                                  color_discrete_sequence=[color])
                    fig.update_layout(yaxis_title=_y_label())
            else:
                fig = px.line(df, y=col, color_discrete_sequence=[color])
            fig.update_traces(line=dict(width=2.5), mode="lines+markers",
                              marker=dict(size=5))

        # ── scatter ──────────────────────────────────────────────────────────
        elif ctype == "scatter":
            if y_col and is_numeric_dtype(df[y_col]):
                fig = px.scatter(df, x=col, y=y_col,
                                 color_discrete_sequence=[color])
                fig.update_layout(yaxis_title=y_col)
            else:
                fig = px.scatter(df, x=df.index, y=col,
                                 color_discrete_sequence=[color])
            fig.update_traces(marker=dict(size=7, opacity=0.75,
                              line=dict(width=0.5, color="#070912")))

        # ── bar (default) ─────────────────────────────────────────────────────
        else:
            if y_col and is_numeric_dtype(df[y_col]):
                agg_data = _agg_df(col, y_col, BAR_TOP_N)
                total_cats = int(df[col].nunique())
                suffix = f" (top {BAR_TOP_N} of {total_cats})" if total_cats > BAR_TOP_N else ""
                fig = px.bar(agg_data, x=col, y=y_col,
                             color_discrete_sequence=[color])
                fig.update_layout(yaxis_title=_y_label())
                fig.update_traces(marker_line_width=0)
                return style_fig(fig, title=f"{col} · {ctype}{suffix}")
            else:
                vc = df[col].value_counts().head(BAR_TOP_N).reset_index()
                vc.columns = [col, "count"]
                total_cats = int(df[col].nunique())
                suffix = f" (top {BAR_TOP_N} of {total_cats})" if total_cats > BAR_TOP_N else ""
                fig = px.bar(vc, x=col, y="count", color_discrete_sequence=[color])
                fig.update_traces(marker_line_width=0)
                return style_fig(fig, title=f"{col} · {ctype}{suffix}")

        # Build title
        title_y = f" vs {y_col}" if y_col else ""
        return style_fig(fig, title=f"{col}{title_y} · {ctype}")

    def run(self, datasets, stylist_output):
        log_lines = [f"{self.icon} {self.name} activated"]
        figures, guard_corrections = {}, 0
        for name, df in datasets.items():
            figures[name] = {}

            # Build a list of numeric columns to use as Y candidates for categorical X
            num_cols = [c for c in df.columns if is_numeric_dtype(df[c]) and c != "_source"]
            cat_cols = [c for c in df.columns if not is_numeric_dtype(df[c]) and c != "_source"]

            for col, style in stylist_output.get(name, {}).items():
                if col not in df.columns:
                    continue
                try:
                    original_type = style["chart_type"]
                    safe_type, gw = self._apply_safety_guards(df, col, original_type)
                    if safe_type != original_type:
                        style = {**style, "chart_type": safe_type}
                        guard_corrections += 1

                    # --- Smart Y pairing ---
                    # For categorical X columns, pick the best numeric Y so charts show
                    # real values instead of defaulting to value_counts().
                    # For numeric X columns that work better with another numeric Y
                    # (scatter/line), also pair them if possible.
                    y_col = None
                    agg_func = "mean"
                    col_is_cat = not is_numeric_dtype(df[col])

                    if col_is_cat and num_cols and safe_type in ("bar", "line", "pie"):
                        # Pick the numeric column with the highest variance as Y
                        best = max(num_cols, key=lambda c: df[c].std() if df[c].std() > 0 else 0)
                        y_col = best
                        agg_func = "mean"
                    elif not col_is_cat and safe_type in ("scatter",) and num_cols:
                        # For scatter on a numeric col, pair with the next best numeric col
                        candidates = [c for c in num_cols if c != col]
                        if candidates:
                            y_col = max(candidates, key=lambda c: abs(df[col].corr(df[c])) if df[c].std() > 0 else 0)
                            agg_func = "mean"

                    fig = self._make_fig(df, col, style, y_col=y_col, agg_func=agg_func)
                    figures[name][col] = {
                        "fig": fig, "type": safe_type, "color": style["color"],
                        "original_type": original_type, "guard_warning": gw,
                        "y_col": y_col, "agg_func": agg_func,
                    }
                    y_info = f" vs {y_col} [{agg_func}]" if y_col else ""
                    log_lines.append(
                        f"  {'🛡️' if gw else '✅'} {name} → '{col}'{y_info}: "
                        f"{'∼' if gw else ''}{safe_type}"
                        f"{f' (was {original_type})' if gw else ''}"
                    )
                except Exception as e:
                    log_lines.append(f"  ⚠️ {name} → '{col}': {type(e).__name__}: {e}")
        return {"output": figures, "log": "\n".join(log_lines)}


class CriticAgent:
    name = "Critic Agent"
    icon = "🧪"
    VALID_CHARTS = VALID_CHART_TYPES

    def _ask_llm_to_critique(self, col, dataset_name, ctype, dist, nuniq, n_rows, sample_vals, stylist_reason):
        prompt = f"""Review chart choice for '{col}' (dataset: '{dataset_name}').
Chosen: {ctype}, Reason: {stylist_reason}, Distribution: {dist}, Unique: {nuniq}/{n_rows}, Samples: {sample_vals}
Reply EXACTLY:
VERDICT: <APPROVE or REJECT>
REPLACEMENT: <chart if REJECT, else same>
CRITIQUE: <one sentence>"""
        response = ask_llama(prompt, self.name)
        verdict, replacement, critique = "APPROVE", ctype, "Approved as-is."
        for line in response.strip().splitlines():
            ls = line.strip()
            if ls.upper().startswith("VERDICT:"): verdict = "REJECT" if "REJECT" in ls.upper() else "APPROVE"
            elif ls.upper().startswith("REPLACEMENT:"):
                raw = ls.split(":", 1)[1].strip().lower()
                for v in self.VALID_CHARTS:
                    if v in raw: replacement = v; break
            elif ls.upper().startswith("CRITIQUE:"): critique = ls.split(":", 1)[1].strip()
        if verdict == "REJECT" and replacement == ctype: replacement = "bar"
        return verdict, replacement, critique

    def run(self, retriever_output, stylist_output, visualizer_output, datasets):
        log_lines = [f"{self.icon} {self.name} activated"]
        critiques, overrides = {}, {}
        for name, ctx in retriever_output.items():
            critiques[name], overrides[name] = {}, {}
            col_styles = stylist_output.get(name, {})
            for col, style in col_styles.items():
                if col not in datasets[name].columns: continue
                df = datasets[name]
                ctype, sr = style["chart_type"], style.get("reason", "")
                dist = ctx["dist_notes"].get(col, "N/A") if is_numeric_dtype(df[col]) else "categorical"
                nuniq, n_rows, sv = int(df[col].nunique()), len(df), df[col].dropna().head(SAMPLE_PREVIEW_ROWS).tolist()
                verdict, replacement, critique = self._ask_llm_to_critique(col, name, ctype, dist, nuniq, n_rows, sv, sr)
                critiques[name][col] = [f"Stylist: {ctype} — {sr}", f"Verdict: {verdict}", f"Critique: {critique}"]
                if verdict == "REJECT":
                    overrides[name][col] = replacement
                    log_lines.append(f"  🔄 {name} → '{col}': REJECTED '{ctype}' → '{replacement}'")
                else:
                    log_lines.append(f"  ✅ {name} → '{col}': APPROVED")
        total = sum(len(v) for v in overrides.values())
        return {"output": critiques, "overrides": overrides, "log": "\n".join(log_lines)}


# ─── Helper: Domain Detection ───
DOMAIN_KEYWORDS = {
    "Finance": ["revenue", "profit", "cost", "expense", "income", "budget", "financial", "account", "audit", "tax", "invoice", "payment", "transaction", "balance", "cash", "asset", "liability", "equity", "depreciation", "amortization"],
    "Energy": ["toe", "consumption", "energy", "power", "fuel", "gas", "electricity", "emission", "carbon", "solar", "wind", "thermal", "kwh", "mw", "gwh", "steam", "coal", "oil", "renewable", "efficiency", "heat", "cooling", "generation", "load", "demand"],
    "Retail": ["sales", "customer", "product", "inventory", "order", "sku", "store", "price", "quantity", "category", "brand", "supplier", "warehouse", "shipment", "promotion", "discount", "margin", "revenue"],
    "HR": ["employee", "salary", "hiring", "turnover", "attrition", "department", "manager", "payroll", "bonus", "benefit", "promotion", "tenure", "headcount", "recruit", "training", "performance"],
    "Manufacturing": ["production", "defect", "quality", "machine", "downtime", "throughput", "yield", "batch", "assembly", "maintenance", "waste", "oee", "cycle", "output", "capacity"],
    "Healthcare": ["patient", "diagnosis", "treatment", "hospital", "doctor", "prescription", "surgery", "admission", "discharge", "bmi", "blood", "heart", "cancer", "diabetes", "symptom"],
    "Supply Chain": ["logistics", "supplier", "warehouse", "shipment", "carrier", "lead", "delivery", "route", "freight", "transport", "inventory", "stock", "dispatch", "container"],
}

# Directionality rules: tells the LLM whether a higher or lower value is desirable per domain
def _is_timestamp_col(series: pd.Series) -> bool:
    """Detect if a numeric column is actually a Unix timestamp (epoch nanoseconds/milliseconds)."""
    if not is_numeric_dtype(series):
        return False
    vals = series.dropna()
    if vals.empty:
        return False
    # Unix timestamps in seconds: ~1.5e9 (year 2018), ms: ~1.5e12, ns: ~1.5e18
    mean_val = vals.mean()
    if mean_val > 1e15:  # nanosecond timestamps
        return True
    if mean_val > 1e11:  # millisecond timestamps
        return True
    return False

def _is_conversion_rate_col(col_name: str) -> bool:
    """Detect if a column name suggests it's a fixed conversion rate/ratio."""
    name = col_name.lower().replace(" ", "_").replace("-", "_")
    keywords = ["conversion", "conversion_rate", "conversion_factor", "rate", "multiplier", "coefficient", "factor"]
    return any(kw == name or name.endswith("_" + kw) or name.startswith(kw + "_") or kw in name for kw in keywords)

def detect_domain(datasets: Dict[str, pd.DataFrame]) -> str:
    scores = {d: 0 for d in DOMAIN_KEYWORDS}
    for df in datasets.values():
        all_cols = " ".join(str(c).lower() for c in df.columns)
        all_vals = " ".join(
            " ".join(df[c].dropna().astype(str).tolist()).lower()[:2000]
            for c in df.columns if not is_numeric_dtype(df[c])
        )
        text = all_cols + " " + all_vals
        for domain, kws in DOMAIN_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in text:
                    scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Business"


# ═══════════════════════════════════════════════════════════════════════════
# 5. INSIGHT AGENT  — LLM-Powered Strategic Analysis
# ═══════════════════════════════════════════════════════════════════════════
# 5. INSIGHT AGENT  — LLM-Powered Strategic Analysis
class InsightAgent:
    """
    Insight Agent — Deterministic stats + LLM narrative.
    Works on any business domain, any number of datasets.
    Maps out isolated analyses first, then performs a cross-dataset reduction synthesis.
    """
    name = "Insight Agent"
    icon = "🧠"

    @staticmethod
    def _fmt(x):
        try:
            x = float(x)
            if x != x or x in (float("inf"), float("-inf")):
                return "n/a"
            r = round(x, 2)
            return f"{int(r):,}" if r == int(r) else f"{r:,.2f}"
        except (TypeError, ValueError):
            return str(x)

    @staticmethod
    def _compute_stats(df: pd.DataFrame) -> dict:
        """Extract exact statistics from raw data — no LLM involved here."""
        stats = {}
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        # Numeric stats
        numeric_stats = []
        for col in num_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outlier_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
            # Get actual outlier values, not just count
            outlier_vals = []
            if iqr > 0:
                mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
                outlier_vals = sorted(s[mask].tolist(), reverse=True)[:3]  # top 3 worst outliers
            numeric_stats.append({
                "column": col,
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "skew": round(float(s.skew()), 2),
                "outliers": outlier_count,
                "outlier_examples": [round(v, 2) for v in outlier_vals],
                "null_pct": round(df[col].isnull().mean() * 100, 1),
            })

        # Categorical stats
        categorical_stats = []
        for col in cat_cols:
            vc = df[col].value_counts()
            all_values = [
                {"value": str(v), "pct": round(float(cnt / len(df) * 100), 1)}
                for v, cnt in vc.items()
            ]

            categorical_stats.append({
                "column": col,
                "unique_values": int(df[col].nunique()),
                "all_values": all_values,   # full breakdown, not just top
                "null_pct": round(df[col].isnull().mean() * 100, 1),
            })

        # Correlations (top pairs only)
        top_correlations = []
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()
            pairs = (
                corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                .stack()
                .reset_index()
            )
            pairs.columns = ["col_a", "col_b", "correlation"]
            pairs["abs_corr"] = pairs["correlation"].abs()
            top = pairs.nlargest(5, "abs_corr")
            top_correlations = [
                {"col_a": r["col_a"], "col_b": r["col_b"], "correlation": round(r["correlation"], 3)}
                for _, r in top.iterrows()
            ]

        # Year-over-year trend for date columns
        yoy_trends = []
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower()]
        for dcol in date_cols:
            try:
                parsed = pd.to_datetime(df[dcol], errors='coerce')
                df_temp = df.copy()
                df_temp['_year'] = parsed.dt.year
                
                # 🚨 التعديل السحري 1: فلترة العواميد عشان نتجاهل أي حاجة فيها "Rate" وميجمعهاش
                num_cols_local = [c for c in df.select_dtypes(include=np.number).columns.tolist() if "rate" not in c.lower()]
                
                for ncol in num_cols_local[:2]:  # limit to first 2 numeric cols
                    yearly = df_temp.groupby('_year')[ncol].sum().reset_index()
                    trend = [{"year": int(r['_year']), "total": round(float(r[ncol]), 2)} for _, r in yearly.iterrows()]
                    
                    if len(trend) > 1:
                        # 🚨 التعديل السحري 2: بايثون هو اللي بيحسب الزيادة والنقصان مش الموديل
                        first_val = trend[0]["total"]
                        last_val = trend[-1]["total"]
                        diff = last_val - first_val
                        
                        direction = "increased" if diff > 0 else "decreased" if diff < 0 else "remained flat"
                        pct_change = abs((diff / first_val) * 100) if first_val != 0 else 0
                        
                        summary_text = f"From {trend[0]['year']} to {trend[-1]['year']}, '{ncol}' {direction} by {pct_change:.1f}% (from {first_val} to {last_val})."
                        
                        yoy_trends.append({"date_col": dcol, "metric": ncol, "by_year": trend, "summary": summary_text})
            except Exception:
                pass

        stats["shape"] = df.shape
        stats["null_total"] = int(df.isnull().sum().sum())
        stats["duplicate_rows"] = int(df.duplicated().sum())
        stats["numeric"] = numeric_stats
        stats["categorical"] = categorical_stats
        stats["top_correlations"] = top_correlations
        stats["yoy_trends"] = yoy_trends
        return stats

    def _compile_single_context(self, name: str, stats: dict) -> str:
        """Helper to convert the raw numerical stats dictionary into a clean string for the isolated prompt."""
        lines = [
            f"Shape: {stats['shape'][0]} rows x {stats['shape'][1]} cols",
            f"Total Nulls: {stats['null_total']} | Duplicate Rows: {stats['duplicate_rows']}"
        ]
        
        if stats["numeric"]:
            lines.append("Numeric Features:")
            for c in stats["numeric"]:
                outlier_str = f"outliers={c['outliers']}"
                if c.get("outlier_examples"):
                    outlier_str += f" (top values: {c['outlier_examples']})"
                lines.append(
                    f"  - {c['column']}: mean={self._fmt(c['mean'])}, median={self._fmt(c['median'])}, "
                    f"std={self._fmt(c['std'])}, min={self._fmt(c['min'])}, max={self._fmt(c['max'])}, "
                    f"{outlier_str}, nulls={c['null_pct']}%"
                )
                
        if stats["categorical"]:
            lines.append("Categorical Features:")
            for c in stats["categorical"]:
                breakdown = ", ".join([f"'{v['value']}' ({v['pct']}%)" for v in c["all_values"]])
                lines.append(
                    f"  - {c['column']}: {c['unique_values']} unique values — {breakdown}, nulls={c['null_pct']}%"
                )
                
        if stats["top_correlations"]:
            lines.append("Top Internal Correlations:")
            for corr in stats["top_correlations"]:
                lines.append(f"  - {corr['col_a']} ↔ {corr['col_b']}: {corr['correlation']}")

        if stats.get("yoy_trends"):
            # 🚨 إجبار الموديل يقرا الحقائق المحسوبة جاهزة
            lines.append("Year-over-Year Trends (PRE-CALCULATED FACTS):")
            for t in stats["yoy_trends"]:
                lines.append(f"  - {t.get('summary', '')}")
                
        return "\n".join(lines)

    def _compute_cross_dataset_facts(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """Compute cross-dataset facts using fuzzy column matching and full category breakdowns."""
        fact_lines = []
        names = list(datasets.keys())

        # 1. Fuzzy-match numeric columns across datasets by keyword
        keywords = [
            'consumption', 'toe', 'value', 'amount', 'sales', 'sum',
            'rate', 'revenue', 'cost', 'price', 'quantity', 'units',
            'profit', 'loss', 'budget', 'actual', 'target', 'count',
            'volume', 'weight', 'score', 'index', 'total', 'average'
        ]
        
        # Build a profile for each dataset: {keyword_group: {name: {mean, max, total}}}
        metric_profiles = {}
        for name, df in datasets.items():
            for col in df.columns:
                col_lower = col.lower().replace(" ", "").replace("_", "")
                matched_kw = next((kw for kw in keywords if kw in col_lower), None)
                if matched_kw:
                    if matched_kw not in metric_profiles:
                        metric_profiles[matched_kw] = {}
                    s = df[col].dropna()
                    metric_profiles[matched_kw][name] = {
                        "col": col,
                        "mean": round(float(s.mean()), 2),
                        "max": round(float(s.max()), 2),
                        "total": round(float(s.sum()), 2),
                        "median": round(float(s.median()), 2),
                    }

        # 2. Output cross-dataset metric comparisons
        for kw, profile in metric_profiles.items():
            if len(profile) >= 2:
                fact_lines.append(f"\n--- Metric Group: '{kw}' ---")
                sorted_profile = sorted(profile.items(), key=lambda x: x[1]["mean"], reverse=True)
                for rank, (name, vals) in enumerate(sorted_profile, 1):
                    fact_lines.append(
                        f"  {rank}. {name} ({vals['col']}): "
                        f"Mean={self._fmt(vals['mean'])}, "
                        f"Max={self._fmt(vals['max'])}, "
                        f"Total={self._fmt(vals['total'])}, "
                        f"Median={self._fmt(vals['median'])}"
                    )
                highest = sorted_profile[0]
                lowest = sorted_profile[-1]
                diff_pct = round((highest[1]["mean"] - lowest[1]["mean"]) / max(lowest[1]["mean"], 0.001) * 100, 1)
                fact_lines.append(
                    f"  👉 VERIFIED MATHEMATICAL FACT: {highest[0]} is greater than {lowest[0]}. "
                    f"Specifically, {highest[0]} has a {diff_pct}% HIGHER mean '{kw}' "
                    f"({self._fmt(highest[1]['mean'])}) compared to {lowest[0]} ({self._fmt(lowest[1]['mean'])})."
                )

        # 3. Full categorical breakdown per dataset for cross-comparison
        fact_lines.append("\n--- Categorical Breakdown Per Dataset ---")
        for name, df in datasets.items():
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            date_like = [c for c in cat_cols if 'date' in c.lower() or 'year' in c.lower()]
            cat_cols = [c for c in cat_cols if c not in date_like]
            for col in cat_cols:
                vc = df[col].value_counts()
                all_vals = ", ".join([
                    f"'{v}' ({round(cnt/len(df)*100, 1)}%)"
                    for v, cnt in vc.items()
                ])
                fact_lines.append(f"  {name} | {col}: {all_vals}")

        # 4. Year-over-year comparison across datasets
        fact_lines.append("\n--- Year-over-Year Trends Per Dataset ---")
        for name, df in datasets.items():
            date_cols = [c for c in df.columns if 'date' in c.lower()]
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            for dcol in date_cols:
                try:
                    df_temp = df.copy()
                    df_temp['_year'] = pd.to_datetime(df_temp[dcol], errors='coerce').dt.year
                    for ncol in num_cols[:2]:
                        yearly = df_temp.groupby('_year')[ncol].sum().reset_index()
                        trend_str = ", ".join([
                            f"{int(r['_year'])}: {self._fmt(r[ncol])}"
                            for _, r in yearly.iterrows()
                        ])
                        fact_lines.append(f"  {name} | {ncol} by year: {trend_str}")
                except Exception:
                    pass

        # 5. Shared column detection with fuzzy normalization
        fact_lines.append("\n--- Column Overlap Analysis ---")
        normalized = {
            name: {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
            for name, df in datasets.items()
        }
        if len(names) >= 2:
            norm_a = set(normalized[names[0]].keys())
            norm_b = set(normalized[names[1]].keys())
            shared_norm = norm_a & norm_b
            if shared_norm:
                fact_lines.append(f"  Fuzzy-matched shared columns: {list(shared_norm)}")
            else:
                fact_lines.append(
                    "  No exact shared columns — datasets use different schemas but cover the same business domain."
                )

        return "\n".join(fact_lines) if fact_lines else "Insufficient data for cross-dataset comparison."

    def run(self, retriever_output: dict, planner_output: dict,
            critic_output: dict, datasets: dict) -> dict:
        log_lines = [f"{self.icon} {self.name} activated"]
        
        isolated_reports = {}
        all_stats = {}
        
        # ==========================================
        # PHASE 1: ISOLATED DATASET ANALYSIS (THE MAP)
        # ==========================================
        for name, df in datasets.items():
            log_lines.append(f"  🧵 Processing isolated analysis for: '{name}'")
            
            # Compute exact, clean stats for this dataset alone
            stats = self._compute_stats(df)  
            all_stats[name] = stats
            
            # Build context for just this single dataset
            detected_domain = detect_domain({name: df})
            single_context = self._compile_single_context(name, stats)
            
            isolated_prompt = f"""You are a seasoned business analyst and consultant writing for senior management and stakeholders.
Your report must use formal, polished language, be data-supported, and emphasize strategic value.
You have been given exact statistics about the dataset '{name}'.
Infer the business domain yourself from the column names, category values, and data patterns provided below.

Dataset Statistics (PRE-CALCULATED FACTS):
{single_context}

Write an executive business report with these exact sections:

1. Business Domain & Key Categories
   - Infer the business domain from column names and category values
   - List ALL categories with their exact percentages
   - Identify which categories dominate the business and what dependency or risk that creates
   - State clearly what the dominant category means for operations and strategy

2. Operational Efficiency Assessment
   - Quote the Year-over-Year trends exactly as provided in the PRE-CALCULATED FACTS.
   - Translate these trends into operational impact: what does this mean for production capacity, resource allocation, procurement exposure, or operational continuity?
   - Flag any metric that changed more than 20% as a major strategic shift requiring management attention.

CRITICAL RULES:
- NEVER USE $ £ € OR ANY CURRENCY SYMBOL ANYWHERE — ABSOLUTE VIOLATION
- NEVER WRITE "Others" FOR CATEGORIES — LIST EVERY SINGLE VALUE
- DO NOT attempt to calculate differences, percentage changes, or trends yourself. You MUST directly quote the "PRE-CALCULATED FACTS" provided in the context exactly as written.
- Every finding must answer "so what?" — connect numbers to business consequences
- Never use vague language like "might suggest", "could indicate", or "needs further assessment" — be direct and decisive.
- Do not mention data cleaning, charts, or technical methods
- Use whatever units are present in the data — infer them from column names and values, never assume or invent units
- Only use numbers that appear in the dataset statistics above — never invent figures
- Do not include a Recommendations section under any circumstances
- Never mix values between columns — each statistic belongs to exactly one column
"""
            # Generate report for this dataset alone
            isolated_reports[name] = ask_llama(isolated_prompt, f"{self.name}_{name}")
            log_lines.append(f"  ✅ Isolated report completed for '{name}'")

        # ==========================================
        # PHASE 2: CROSS-DATASET INTERSECTION (THE REDUCE)
        # ==========================================
        
        # Compile the individual reports into a single text block
        compiled_isolated_text = ""
        for name, report in isolated_reports.items():
            compiled_isolated_text += f"\n--- ISOLATED REPORT FOR {name} ---\n{report}\n"
            
        # THE BYPASS: If only 1 dataset is sent to this view, skip the intersection!
        combined_report = "N/A - Single dataset analysis"
        if len(datasets) == 1:
            log_lines.append("  ⏭️ Only one dataset detected in this view. Skipping cross-dataset intersection.")
            full_report_narrative = f"=== INDIVIDUAL DATASET PROFILE ===\n{compiled_isolated_text}"
            
        # THE SYNTHESIS: If we are in the "Intersection" view (2 datasets), run the combo
        else:
            log_lines.append("  🔀 Synthesizing cross-dataset intersections...")
            python_facts = self._compute_cross_dataset_facts(datasets)

            combined_prompt = f"""You are a seasoned business analyst and consultant writing a board-level strategic report for senior management and stakeholders.
Your report must use formal, polished language, be data-supported, and emphasize strategic value.
You are comparing {len(datasets)} datasets: {', '.join(datasets.keys())}.

CRITICAL: If the CROSS-DATASET FACTUAL MATRIX below is empty or says 
"Insufficient data", respond with exactly this text and nothing else:
"Insufficient cross-dataset data available. Please ensure both datasets 
are loaded correctly before running intersection analysis."
DO NOT invent any numbers, company names, metrics, or business domains.

Individual Dataset Reports:
{compiled_isolated_text}

CROSS-DATASET FACTUAL MATRIX (Exact figures — do not alter):
{python_facts}

Write a Unified Strategic Board Report with these exact sections:

1. Cross-Company Comparison
   - Infer the shared business domain from the individual reports
   - State which entity performs better on every major metric and by exactly how much
   - Identify shared dependencies or dominant categories that create group-level risk
   - Flag any metric where one entity outperforms the other by more than 30%

2. Critical Divergences
   - Where do the entities contradict each other in efficiency or operational trajectory?
   - Which entity's data shows more concerning signals and why?
   - What is the performance gap and what is driving it?

3. Shared Strategic Opportunities
   - What can the lower-performing entity learn from the higher-performing one?
   - What single initiative would benefit ALL entities simultaneously?
   - What shared dependencies create collective risk that needs group-level action?

Rules:
- NEVER USE $ £ € OR ANY CURRENCY SYMBOL — USE RAW NUMBERS ONLY
- NEVER WRITE "Others" FOR CATEGORIES — LIST EVERY SINGLE VALUE
- A NUMBER GOING UP IS AN INCREASE. 1,235 → 1,857 IS +50% IMPROVEMENT. NEVER CALL AN INCREASE A DECREASE.
- NEVER INVENT NUMBERS — ONLY USE FIGURES FROM THE FACTUAL MATRIX BELOW
- NEVER INVENT COMPANY NAMES — USE EXACT DATASET FILENAMES PROVIDED
- IF THE FACTUAL MATRIX IS EMPTY WRITE EXACTLY: "Insufficient cross-dataset data available" AND STOP — DO NOT GENERATE A FICTIONAL REPORT
- Only use numbers that appear verbatim in the CROSS-DATASET FACTUAL MATRIX or individual reports above — never invent figures
- If the factual matrix is empty or insufficient, explicitly state: "Insufficient cross-dataset data to generate comparison" and stop
- Never invent company names — use the exact dataset filenames provided
- Use whatever units are present in the data — infer them from column names, never assume or invent units
- Every finding must answer "so what?" — connect numbers to business consequences
- Never use vague language like "might suggest" or "could indicate" — be direct and decisive
- Do not mention data cleaning, charts, or technical methods
- Only use numbers that appear in the statistics above — never invent figures
- Never present correlation coefficients as percentage changes — 
  describe the direction of the relationship only
- Do not include a Recommendations section under any circumstances
"""
            combined_report = ask_llama(combined_prompt, f"{self.name}_Combined")
            full_report_narrative = f"=== COMBINED STRATEGIC INTERSECTION ===\n\n{combined_report}\n\n=== INDIVIDUAL DATASET PROFILES ===\n{compiled_isolated_text}"

        return {
            "output": full_report_narrative,
            "isolated_reports": isolated_reports,
            "combined_report": combined_report,
            "stats": all_stats,
            "log": "\n".join(log_lines)
        }

class RecommenderAgent:
    name = "Recommender Agent"
    icon = "💡"

    def run(self, insight_report: str, stats_context: str = "") -> dict:
        log_lines = [f"{self.icon} {self.name} activated"]

        grounding = f"\n\nSupporting statistics (exact figures from raw data):\n{stats_context}" if stats_context else ""

        prompt = f"""You are a Chief Operating Officer (COO) and senior business strategy consultant.
Based on the insight report below, provide a Board-Level Operational Action Plan.

Provide exactly 3 business recommendations strictly focused on operations, strategy, cost, risk, and growth.
DO NOT mention data quality, data cleaning, charts, technical methods, or IT systems.

Format EVERY recommendation exactly like this:
- 📌 [Title]
- 🎯 Why: [Reason referencing an exact number from the insight report]
- 🚀 Action Step: [Specific, actionable step]
- 📈 Expected Outcome: [Measurable business result]
- 📅 Timeline: [Specific quarter or timeframe]

STRICT RULES — VIOLATION INVALIDATES THE REPORT:
1. Every recommendation must reference an exact number from the insight report
2. Never mention data, IT, charts, cleaning, or technical methods
3. Never use currency symbols
4. Every recommendation must have all 5 fields — missing any field is a VIOLATION
5. Be direct and decisive — never use "might", "could", or "further investigation required"
6. Rank recommendations by business impact — highest ROI or risk reduction first

Insight Report:
{insight_report}
"""
        log_lines.append("  🔄 Generating Hybrid (Technical + Operational) recommendations...")
        recommendations = ask_llama(prompt, self.name)
        log_lines.append("  ✅ Hybrid recommendations generated.")

        return {
            "output": recommendations,
            "log": "\n".join(log_lines),
        }


class AgentState(TypedDict):
    datasets: Dict[str, pd.DataFrame]
    results: dict
    overrides_applied: bool
    _has_re_rendered: bool
    isolated_reports: Dict[str, str]
    combined_report: str
    stats: Dict[str, Any]


class Orchestrator:
    def __init__(self):
        self.retriever  = RetrieverAgent()
        self.planner    = PlannerAgent()
        self.stylist    = StylistAgent()
        self.visualizer = VisualizerAgent()
        self.critic     = CriticAgent()
        self.insight    = InsightAgent()

        self.builder = StateGraph(AgentState)
        self._build_graph()

    def _build_graph(self):
        self.builder.add_node("retriever",  self._node_retriever)
        self.builder.add_node("planner",    self._node_planner)
        self.builder.add_node("stylist",    self._node_stylist)
        self.builder.add_node("visualizer", self._node_visualizer)
        self.builder.add_node("critic",     self._node_critic)

        self.builder.set_entry_point("retriever")
        self.builder.add_edge("retriever",  "planner")
        self.builder.add_edge("planner",    "stylist")
        self.builder.add_edge("stylist",    "visualizer")
        self.builder.add_edge("visualizer", "critic")

        # THE KEY PART: feedback loop — critic can send back to visualizer
        self.builder.add_conditional_edges(
            "critic",
            self._critic_decision,
            {
                "re_render": "visualizer",
                "finish":    END
            }
        )
        self.graph = self.builder.compile()

    def _node_retriever(self, state: AgentState):
        r = self.retriever.run(state["datasets"])
        results = state.get("results", {})
        results["retriever"] = r
        return {"results": results}

    def _node_planner(self, state: AgentState):
        r = self.planner.run(state["results"]["retriever"]["output"])
        results = state.get("results", {})
        results["planner"] = r
        return {"results": results}

    def _node_stylist(self, state: AgentState):
        r = self.stylist.run(
            state["results"]["retriever"]["output"],
            state["results"]["planner"]["output"],
            state["datasets"]
        )
        results = state.get("results", {})
        results["stylist"] = r
        return {"results": results}

    def _node_visualizer(self, state: AgentState):
        r = self.visualizer.run(
            state["datasets"],
            state["results"]["stylist"]["output"]
        )
        results = state.get("results", {})
        results["visualizer"] = r
        return {"results": results, "_has_re_rendered": state.get("overrides_applied", False) and not state.get("_has_re_rendered", False)}

    def _node_critic(self, state: AgentState):
        # Skip second critic pass — only run once
        if state.get("overrides_applied"):
            return {"overrides_applied": True}

        r = self.critic.run(
            state["results"]["retriever"]["output"],
            state["results"]["stylist"]["output"],
            state["results"]["visualizer"]["output"],
            state["datasets"]
        )
        results = state.get("results", {})
        results["critic"] = r

        # Apply overrides directly into stylist output
        has_overrides = False
        stylist_out = results["stylist"]["output"]
        for ds_name, overrides in r["overrides"].items():
            for col, new_type in overrides.items():
                if ds_name in stylist_out and col in stylist_out[ds_name]:
                    stylist_out[ds_name][col]["chart_type"] = new_type
                    has_overrides = True

        results["stylist"]["output"] = stylist_out
        return {"results": results, "overrides_applied": has_overrides}

    def _critic_decision(self, state: AgentState):
        # If critic made overrides and visualizer hasn't re-rendered yet → loop back
        if state.get("overrides_applied") and not state.get("_has_re_rendered"):
            return "re_render"
        return "finish"

    def run(self, datasets: dict) -> dict:
        final_state = self.graph.invoke({
            "datasets":          datasets,
            "results":           {},
            "overrides_applied": False,
            "_has_re_rendered":  False,
        })
        return final_state["results"]

    def run_insight(self, results: dict, view_datasets: dict, raw_datasets: dict) -> dict:
        return self.insight.run(
            results["retriever"]["output"],
            results["planner"]["output"],
            results.get("critic", {}),
            raw_datasets
        )


# ─── Triple-View Builder ───
VIEW_A = "view_a"
VIEW_B = "view_b"
VIEW_INTERSECTION = "view_intersection"

VIEW_LABELS = {
    VIEW_A: "Dataset A",
    VIEW_B: "Dataset B",
    VIEW_INTERSECTION: "Intersection (A ∩ B)",
}

def build_views(preprocessed: Dict[str, pd.DataFrame]):
    names = list(preprocessed.keys())
    views: Dict[str, Dict[str, pd.DataFrame]] = {}
    view_titles: Dict[str, str] = {}
    shared_cols: List[str] = []

    if len(names) == 2:
        name_a, name_b = names
        df_a, df_b = preprocessed[name_a], preprocessed[name_b]
        shared_cols = sorted(set(df_a.columns) & set(df_b.columns))
        short_a = name_a.rsplit(".", 1)[0] if "." in name_a else name_a
        short_b = name_b.rsplit(".", 1)[0] if "." in name_b else name_b

        views[VIEW_A] = {name_a: df_a}
        view_titles[VIEW_A] = f"Dataset A — {short_a}"
        views[VIEW_B] = {name_b: df_b}
        view_titles[VIEW_B] = f"Dataset B — {short_b}"

        if shared_cols:
            df_a_s = df_a[shared_cols].copy()
            df_b_s = df_b[shared_cols].copy()
            df_a_s["_source"] = short_a
            df_b_s["_source"] = short_b
            combined = pd.concat([df_a_s, df_b_s], ignore_index=True)
            views[VIEW_INTERSECTION] = {"Intersection (A ∩ B)": combined}
            view_titles[VIEW_INTERSECTION] = (
                f"Intersection — {short_a} ∩ {short_b} "
                f"({len(shared_cols)} shared col{'s' if len(shared_cols) != 1 else ''})"
            )
        return views, view_titles, shared_cols

    views["unified"] = preprocessed
    view_titles["unified"] = "Unified Analysis"
    return views, view_titles, shared_cols


def is_triple_view(views):
    return VIEW_A in views or VIEW_B in views


def get_view_keys(views):
    order = [VIEW_A, VIEW_B, VIEW_INTERSECTION]
    return [v for v in order if v in views]

DATE_NAME_HINTS = [
    "date", "time", "timestamp", "created", "updated",
    "year", "month", "day", "dt", "period"
]

def _is_likely_date_col(col_name: str) -> bool:
    col_lower = col_name.lower().replace("_", "").replace(" ", "")
    return any(hint in col_lower for hint in DATE_NAME_HINTS)

def smart_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df_p, report = df.copy(), []
    df_p.columns = [str(c).strip().lower().replace(" ", "_") for c in df_p.columns]
    report.append("Standardized column names")
    dups = int(df_p.duplicated().sum())
    if dups > 0: df_p = df_p.drop_duplicates(); report.append(f"Removed {dups} duplicate rows")
    else: report.append("No duplicates found")
    converted = []
    for col in df_p.columns:
        od = df_p[col].dtype
        if _is_likely_date_col(col):
            continue
        try: df_p[col] = pd.to_numeric(df_p[col], errors="raise")
        except: pass
        if df_p[col].dtype != od: converted.append(col)
    if converted: report.append(f"Converted to numeric: {', '.join(converted)}")
    for col in df_p.columns:
        mc = int(df_p[col].isnull().sum())
        if mc == 0: continue
        if is_numeric_dtype(df_p[col]):
            fv = df_p[col].median(); df_p[col] = df_p[col].fillna(fv)
            report.append(f"'{col}': {mc} nulls → median ({fv:.2f})")
        else:
            mv = df_p[col].mode()[0] if not df_p[col].mode().empty else "Unknown"
            df_p[col] = df_p[col].fillna(mv)
            report.append(f"'{col}': {mc} nulls → mode ('{mv}')")
    capped = []
    for col in df_p.select_dtypes(include=np.number).columns:
        q1, q3 = df_p[col].quantile(0.25), df_p[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0: continue
        lo, hi = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
        n = int(((df_p[col] < lo) | (df_p[col] > hi)).sum())
        if n > 0: df_p[col] = df_p[col].clip(lower=lo, upper=hi); capped.append(f"'{col}' ({n})")
    if capped: report.append(f"Outliers capped: {', '.join(capped)}")
    else: report.append("No outliers needed capping")
    report.append(f"Final: {df_p.shape[0]} rows x {df_p.shape[1]} cols, nulls: {int(df_p.isnull().sum().sum())}")
    return df_p, report


def compute_unified_outliers(datasets):
    result = {}
    for name, df in datasets.items():
        result[name] = {}
        for col in df.select_dtypes(include=np.number).columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0: continue
            lo = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            hi = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n > 0: result[name][col] = n
    return result


def _detect_time_col(df):
    # Pass 1: already a proper datetime column
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # Pass 2: column name strongly suggests a date
    date_name_hints = [
        "date", "time", "timestamp", "created", "updated",
        "year", "month", "day", "dt", "period"
    ]
    for col in df.columns:
        col_lower = col.lower().replace("_", "").replace(" ", "")
        if any(hint in col_lower for hint in date_name_hints):
            # Try to parse it regardless of dtype
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() >= 0.7:
                    return col
            except Exception:
                continue

    # Pass 3: object dtype — try parsing by content
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(50)
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    return col
            except Exception:
                continue

    # Pass 4: integer column that looks like YYYYMMDD or Unix timestamp
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            sample = df[col].dropna()
            if sample.empty:
                continue
            # YYYYMMDD format: values between 19000101 and 21001231
            if sample.between(19000101, 21001231).mean() >= 0.9:
                try:
                    parsed = pd.to_datetime(sample.astype(str), format="%Y%m%d", errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        return col
                except Exception:
                    pass

    return None


def _prophet_predict(df, time_col, value_col, horizon=30):
    try:
        from prophet import Prophet
    except Exception as e:
        print(f"Prophet Import Error: {e}")
        return None, 0, "unknown", pd.DataFrame()

    import warnings
    import pandas as pd
    import numpy as np

    # بداية الـ try الأساسية
    try:
        ts = df[[time_col, value_col]].copy()
        ts.columns = ["ds", "y"]
        ts["ds"] = pd.to_datetime(ts["ds"], errors="coerce")
        
        # تجميع الداتا لنفس التاريخ (لحل مشكلة الـ 175 Anomaly)
        ts = ts.groupby("ds")["y"].sum().reset_index()
        
        ts = ts.dropna().sort_values("ds").reset_index(drop=True)

        if len(ts) < 10:
            return None, 0, "unknown", pd.DataFrame()

        freq = pd.infer_freq(ts["ds"]) or "MS"

        m = Prophet(
            interval_width=0.85,
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(ts)

        # ── PART A: Score historical points ──────────────────────────────────
        historical_forecast = m.predict(ts[["ds"]])
        ts["yhat"] = historical_forecast["yhat"].values
        ts["yhat_lower"] = historical_forecast["yhat_lower"].values
        ts["yhat_upper"] = historical_forecast["yhat_upper"].values

        ts["is_anomaly"] = (
            (ts["y"] < ts["yhat_lower"]) |
            (ts["y"] > ts["yhat_upper"])
        )
        historical_anomaly_count = int(ts["is_anomaly"].sum())
        historical_anomalies = ts[ts["is_anomaly"]][["ds", "y", "yhat", "yhat_lower", "yhat_upper"]].copy()

        # ── PART B: Forward forecast ──────────────────────────────────────────
        future = m.make_future_dataframe(periods=horizon, freq=freq)
        forecast = m.predict(future)

        last_date = ts["ds"].max()
        future_fc = forecast[forecast["ds"] > last_date].copy()

        if not future_fc.empty:
            hist_mean = ts["y"].mean()
            hist_std = ts["y"].std()

            future_fc["predicted_anomaly"] = (
                (future_fc["yhat"] < hist_mean - 2.0 * hist_std) |
                (future_fc["yhat"] > hist_mean + 2.0 * hist_std)
            )
            future_anomaly_count = int(future_fc["predicted_anomaly"].sum())
        else:
            future_anomaly_count = 0

        total_anomaly_count = historical_anomaly_count + future_anomaly_count

        period_map = {
            "D": "days", "W": "weeks", "M": "months",
            "MS": "months", "H": "hours", "Q": "quarters"
        }
        period_label = f"next {horizon} {period_map.get(freq[:1], 'periods')}"

        return future_fc, total_anomaly_count, period_label, historical_anomalies

    # نهاية الـ try الأساسية (هذا الجزء كان مفقوداً لديك وسبب المشكلة)
    except Exception as e:
        print(f"Prophet Execution Error on column {value_col}: {e}")
        return None, 0, "unknown", pd.DataFrame()
    
def _bootstrap_predict(df, test_split=0.2):
    """
    Part A: Honest held-out anomaly detection.
    Trains IsolationForest on the first 80% of rows,
    scores the last 20% as genuinely unseen data.
    
    Returns:
        anomaly_count   : int   — anomalies found in the held-out set
        anomaly_rate    : float — anomaly_count / held-out size
        num_cols        : list  — numeric columns used
        iso             : trained IsolationForest model (reused for Part B)
        scaler          : trained StandardScaler (reused for Part B)
        X_test          : scaled held-out data (for visualization)
        test_indices    : original DataFrame indices of the held-out rows
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return 0, 0.0, [], None, None, None, []

    clean = df[num_cols].dropna()
    if len(clean) < 10:
        return 0, 0.0, num_cols, None, None, None, []

    scaler = StandardScaler()
    X = scaler.fit_transform(clean)

    # Honest train/test split — preserve row order (don't shuffle)
    split = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split], X[split:]
    test_indices = clean.index[split:].tolist()

    iso = IsolationForest(
        contamination=ANOMALY_CONTAMINATION,
        random_state=ANOMALY_RANDOM_STATE
    )
    iso.fit(X_train)

    preds = iso.predict(X_test)
    anomaly_count = int((preds == -1).sum())
    anomaly_rate = round(anomaly_count / len(X_test), 4)

    return anomaly_count, anomaly_rate, num_cols, iso, scaler, X_test, test_indices

def _simulate_future_anomalies(iso, scaler, X_historical, num_cols, n_simulated=300):
    """
    Part B: Optional simulation layer.
    Generates synthetic future-like rows by sampling from the learned
    distribution of historical data and adding realistic noise.
    Scores them with the already-trained IsolationForest from Part A.

    This is NOT a forecast — it simulates what anomalies might look like
    if new data arrives with similar statistical properties.

    Returns:
        sim_anomaly_count : int
        sim_anomaly_rate  : float
        sim_df            : DataFrame of simulated rows with Status column
    """
    if iso is None or scaler is None or X_historical is None:
        return 0, 0.0, pd.DataFrame()

    rng = np.random.default_rng(ANOMALY_RANDOM_STATE)

    # Sample rows from historical data as a base
    sampled_idx = rng.integers(0, len(X_historical), size=n_simulated)
    X_base = X_historical[sampled_idx]

    # Add realistic noise — std=0.15 is more meaningful than the original 0.05
    noise = rng.normal(0, 0.15, size=X_base.shape)
    X_simulated = X_base + noise

    # Score with the already-trained model — no refitting
    preds = iso.predict(X_simulated)
    anomaly_count = int((preds == -1).sum())
    anomaly_rate = round(anomaly_count / n_simulated, 4)

    # Build a readable DataFrame in original scale
    X_rescaled = scaler.inverse_transform(X_simulated)
    sim_df = pd.DataFrame(X_rescaled, columns=num_cols)
    sim_df["Status"] = np.where(preds == -1, "Anomaly", "Normal")
    sim_df["_source"] = "simulated"

    return anomaly_count, anomaly_rate, sim_df

def run_predictive_anomaly(datasets, horizon=30):
    per_dataset = []
    all_frames = []
    all_feature_cols = set()
    dominant_method = "bootstrap"

    for name, df in datasets.items():
        time_col = _detect_time_col(df)

        print(f"[{name}] time_col detected: {time_col}")
        print(f"[{name}] column dtypes:\n{df.dtypes}")


        if time_col is not None:
            dominant_method = "prophet"
            num_cols = [c for c in df.select_dtypes(include=np.number).columns if "rate" not in c.lower()]
            total_anomalies = 0
            col_details = []
            
            for col in num_cols:
                future_fc, n_anom, period_label, hist_anomalies = _prophet_predict(
                    df, time_col, col, horizon
                )
                
                if future_fc is None:
                    continue
                    
                total_anomalies += n_anom

                # --- NEW: Extract specific future anomalies for preview ---
                predicted_list = []
                if not future_fc.empty and "predicted_anomaly" in future_fc.columns:
                    anom_df = future_fc[future_fc["predicted_anomaly"]]
                    for _, r in anom_df.iterrows():
                        predicted_list.append({
                            "date": str(r["ds"]).split()[0], # Get YYYY-MM-DD
                            "expected": round(r["yhat"], 2)
                        })

                col_details.append({
                    "column": col,
                    "predicted_anomalies": n_anom,
                    "historical_anomalies": len(hist_anomalies) if hist_anomalies is not None and not hist_anomalies.empty else 0,
                    "predicted_list": predicted_list
                })
                all_feature_cols.add(col)
                
                # Create DataFrame for visualization
                ts = df[[time_col, col]].copy()
                ts.columns = ["ds", col]
                ts["ds"] = pd.to_datetime(ts["ds"], errors="coerce")
                ts["_source"] = name
                ts["Status"] = "Normal"
                
                # FIX: Correctly map the detected anomalies to the Status column
                if hist_anomalies is not None and not hist_anomalies.empty:
                    anomaly_dates = set(hist_anomalies["ds"])
                    ts["Status"] = ts["ds"].apply(lambda x: "Anomaly" if x in anomaly_dates else "Normal")
                
                all_frames.append(ts[[col, "_source", "Status"]])
                
            per_dataset.append({
                "dataset": name, "method": "prophet", "time_column": time_col,
                "predicted_anomalies": total_anomalies, "horizon": period_label, "columns": col_details,
            })
        else:
            # ── PART A: Honest held-out detection ────────────────────────────
            (anomaly_count, anomaly_rate, feat_cols,
            iso, scaler, X_test, test_indices) = _bootstrap_predict(df)

            clean = df[feat_cols].dropna().copy() if feat_cols else pd.DataFrame()

            if not clean.empty and iso is not None:
                # Score ALL rows for visualization (train + test)
                # but clearly mark which rows are held-out
                X_all = scaler.transform(clean)
                preds_all = iso.predict(X_all)
                clean["Status"] = np.where(preds_all == -1, "Anomaly", "Normal")
                clean["_source"] = name
                clean["_split"] = "train"
                split = int(len(clean) * 0.8)
                clean.iloc[split:, clean.columns.get_loc("_split")] = "held-out"
                all_frames.append(clean[feat_cols[:3] + ["_source", "Status", "_split"]])
                all_feature_cols.update(feat_cols)

            per_dataset.append({
                "dataset": name,
                "method": "isolation_forest",
                "time_column": None,
                "predicted_anomalies": anomaly_count,
                "anomaly_rate": anomaly_rate,
                # Honest label
                "horizon": (
                    f"held-out detection on last "
                    f"{int(len(clean) * 0.2) if not clean.empty else 0} rows "
                    f"({int((1 - ANOMALY_CONTAMINATION) * 100)}% confidence)"
                ),
                "columns": feat_cols,
            })

            # ── PART B: Optional simulation layer ────────────────────────────
            if iso is not None and scaler is not None and X_test is not None:
                sim_count, sim_rate, sim_df = _simulate_future_anomalies(
                    iso, scaler, X_test, feat_cols, n_simulated=300
                )
                # --- NEW: Extract simulated anomalies ---
                sim_list = []
                if sim_count > 0 and not sim_df.empty:
                    anoms = sim_df[sim_df["Status"] == "Anomaly"]
                    for _, r in anoms.iterrows():
                        # Grab up to 2 features to show in the preview
                        preview_vals = " | ".join([f"{c}: {round(r[c], 2)}" for c in feat_cols[:2]])
                        sim_list.append({"expected": preview_vals}) 
            # ----------------------------------------

                if not sim_df.empty:
                    all_frames.append(sim_df[feat_cols[:3] + ["_source", "Status"]])

                # Attach simulation results as a separate key — don't mix with Part A
                per_dataset[-1]["simulation"] = {
                    "simulated_rows": 300,
                    "simulated_anomalies": sim_count,
                    "simulated_anomaly_rate": sim_rate,
                    "predicted_list": sim_list,
                    "label": (
                        "Simulated anomaly risk based on learned data distribution. "
                        "Not a forecast — shows what anomalies might look like "
                        "if new data arrives with similar statistical properties."
                    )
                }

    total_predicted = sum(d["predicted_anomalies"] for d in per_dataset)
    horizon_label = per_dataset[0]["horizon"] if per_dataset else f"next {horizon} periods"
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else None
    return {
        "method": dominant_method,
        "predicted_anomalies": total_predicted,
        "horizon": horizon_label,
        "confidence": "95%" if dominant_method == "prophet" else f"{int((1 - ANOMALY_CONTAMINATION)*100)}%",
        "per_dataset": per_dataset,
        "_combined": combined,
        "_feature_cols": list(all_feature_cols),
    }

def run_similarity_recommender(df, row_index, top_n=RECOMMENDER_TOP_N):
    num_df = df.select_dtypes(include=np.number).dropna()
    if num_df.empty or row_index not in num_df.index: return None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(num_df)
    target_pos = num_df.index.get_loc(row_index)
    target_vector = scaled[target_pos].reshape(1, -1)
    sims = cosine_similarity(target_vector, scaled).flatten()
    order = sims.argsort()[::-1]
    order = [i for i in order if i != target_pos][:top_n]
    result = df.loc[num_df.index[order]].copy()
    result.insert(0, "similarity_score", [round(float(sims[i]), 4) for i in order])
    return result


def fig_to_json(fig):
    return json.loads(fig.to_json())

def fig_to_html(fig):
    return fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})


def _pdf_clean(text):
    """Strip markdown symbols and encode safely for FPDF latin-1."""
    import re
    text = str(text) if text else ""
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Convert bullet * or - at line start to dash
    text = re.sub(r'^\s*[\*\-]\s+', '  - ', text, flags=re.MULTILINE)
    # Remove === and --- section dividers
    text = re.sub(r'^[=\-]{3,}.*$', '', text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Encode to latin-1 safely (drops Arabic/special chars gracefully)
    text = text.encode("latin-1", "ignore").decode("latin-1")
    return text.strip()


def _pdf_section_header(pdf, number, title, color=(0, 86, 179)):
    """Draw a colored section header bar."""
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, f"  {number}. {title}", ln=True, fill=True)
    pdf.set_text_color(40, 40, 40)
    pdf.ln(4)


def _pdf_kv_line(pdf, label, value):
    """Print a bold label followed by normal value on the same line."""
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(0, 86, 179)
    pdf.cell(45, 6, label, ln=False)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, str(value)[:90], ln=True)

def _pdf_safe_cell(pdf, text, line_height=6):
    """Write text safely, splitting long words and catching FPDF errors."""
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    words = str(text).split()
    safe = " ".join(w[:50] for w in words)
    try:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_w, line_height, safe)
    except Exception:
        pass


def _pdf_body_text(pdf, text, line_height=6):
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            pdf.ln(2)
            continue
        if para.startswith("- ") or para.startswith("  - "):
            bullet = "  * " + para.lstrip("- ").strip()
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(40, 40, 40)
            _pdf_safe_cell(pdf, bullet, line_height)
        elif len(para) < 60 and para.endswith(":"):
            pdf.set_font("Arial", "B", 10)
            pdf.set_text_color(0, 86, 179)
            _pdf_safe_cell(pdf, para, line_height)
            pdf.set_text_color(40, 40, 40)
        else:
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(40, 40, 40)
            _pdf_safe_cell(pdf, para, line_height)



def _pdf_summary_box(text, max_chars=1200):
    import re
    summary_match = re.search(
        r'(?:executive summary|summary|conclusion|overview)[:\s]*(.*?)(?:\n\n|\Z)',
        text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        snippet = summary_match.group(1).strip()[:max_chars]
    else:
        paras = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 40]
        snippet = "\n\n".join(paras[:3])[:max_chars]
    return _pdf_clean(snippet)


def generate_pdf(insight_text, recommender_text, charts_dir):
    from datetime import date
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 15, 15)

    # COVER PAGE
    pdf.add_page()
    pdf.set_fill_color(10, 20, 50)
    pdf.rect(0, 0, 210, 55, "F")
    pdf.set_y(12)
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "Smart Business Analytics", ln=True, align="C")
    pdf.set_font("Arial", "I", 13)
    pdf.set_text_color(150, 200, 255)
    pdf.cell(0, 8, "Executive AI Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(180, 180, 200)
    pdf.cell(0, 7, f"Generated: {date.today().strftime('%B %d, %Y')}", ln=True, align="C")
    pdf.ln(30)
    pdf.set_draw_color(0, 120, 220)
    pdf.set_line_width(1)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)
    pdf.set_fill_color(240, 246, 255)
    pdf.set_draw_color(0, 86, 179)
    pdf.set_line_width(0.5)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(0, 86, 179)
    pdf.cell(0, 8, "  Executive Summary", ln=True, fill=True, border=1)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(40, 40, 40)
    summary = _pdf_summary_box(insight_text or "")
    if not summary and insight_text:
        clean_for_summary = _pdf_clean(insight_text)
        paras = [p.strip() for p in clean_for_summary.split("\n") if p.strip() and len(p.strip()) > 30]
        summary = " ".join(paras[:5])[:900]
    if not summary:
        summary = "No insight data available. Run the Agent Pipeline first."
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    try:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_w, 6, summary, border=1)
    except Exception:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_w, 6, summary[:300], border=1)
    pdf.ln(10)

    # SECTION 1: INSIGHTS
    pdf.add_page()
    _pdf_section_header(pdf, 1, "Strategic Insight Report")
    if insight_text:
        _pdf_body_text(pdf, _pdf_clean(insight_text)[:3500])
    else:
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 8, "No insight data available.", ln=True)

    # SECTION 2: RECOMMENDATIONS
    pdf.add_page()
    _pdf_section_header(pdf, 2, "Executive Recommendations", color=(0, 120, 80))
    if recommender_text:
        _pdf_body_text(pdf, _pdf_clean(recommender_text)[:3000])
    else:
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 8, "No recommendations available. Run the Agent Pipeline first.", ln=True)

    # SECTION 3: CHARTS
    pdf.add_page()
    _pdf_section_header(pdf, 3, "Key Data Visualizations", color=(120, 0, 120))
    charts_found = False
    for i in range(PDF_MAX_CHARTS):
        img_path = os.path.join(charts_dir, f"chart_{i}.png")
        if not os.path.exists(img_path):
            continue
        charts_found = True
        if pdf.get_y() > 220:
            pdf.add_page()
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 7, f"Figure {i+1}", ln=True)
        pdf.image(img_path, x=15, w=180)
        pdf.ln(6)
    if not charts_found:
        pdf.set_fill_color(255, 245, 230)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(160, 80, 0)
        pdf.multi_cell(0, 8,
            "No charts were saved during this session.\n"
            "To include charts: run the Agent Pipeline, then click Export PDF Report.",
            border=1, fill=True)

    pdf_path = os.path.join(charts_dir, "Executive_Report.pdf")
    pdf.output(pdf_path)
    return pdf_path

