const API = "/api";
let sessionId = null;
let currentTab = "data";
let viewKeys = ["unified"];
let viewTitles = {};
let tripleMode = false;
let sharedCols = [];

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

function showLoading(msg = "Processing...") {
  const el = document.getElementById("loadingOverlay");
  el.querySelector("p").textContent = msg;
  el.classList.remove("hidden");
}
function hideLoading() { document.getElementById("loadingOverlay").classList.add("hidden"); }

function showStatus(msg, type = "info") {
  const el = document.getElementById("statusMessage");
  el.className = `status-msg ${type}`;
  el.textContent = msg;
  el.classList.remove("hidden");
}
function clearStatus() { document.getElementById("statusMessage").classList.add("hidden"); }

// ─── Minimal, safe Markdown → HTML renderer (no external dependencies) ───
// The AI Insight / Recommender reports are Markdown. Previously they were dumped
// with a plain "\n -> <br>" replace, so headings (##), bold (**) and bullet lists
// showed as literal characters. This renders them properly and escapes HTML first.
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function renderInline(s) {
  // input is already HTML-escaped
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");          // `code`
  s = s.replace(/\*\*([^\n]+?)\*\*/g, "<strong>$1</strong>"); // **bold** (may wrap *italics*)
  s = s.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>"); // *italic*
  s = s.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");    // _italic_
  return s;
}
function renderMarkdown(md) {
  if (!md) return "";
  const lines = String(md).replace(/\r\n/g, "\n").split("\n");
  let html = "", listOpen = false;
  const closeList = () => { if (listOpen) { html += "</ul>"; listOpen = false; } };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) { closeList(); continue; }
    if (/^\s*---+\s*$/.test(line)) { closeList(); html += "<hr>"; continue; }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { closeList(); const lv = Math.min(h[1].length, 6);
      html += `<h${lv}>${renderInline(escapeHtml(h[2]))}</h${lv}>`; continue; }
    const bq = line.match(/^\s*>\s?(.*)$/);
    if (bq) { closeList(); html += `<blockquote>${renderInline(escapeHtml(bq[1]))}</blockquote>`; continue; }
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) { if (!listOpen) { html += "<ul>"; listOpen = true; }
      html += `<li>${renderInline(escapeHtml(li[1]))}</li>`; continue; }
    closeList();
    html += `<p>${renderInline(escapeHtml(line))}</p>`;
  }
  closeList();
  return html;
}
(function injectMarkdownStyles() {
  if (document.getElementById("md-style")) return;
  const el = document.createElement("style");
  el.id = "md-style";
  el.textContent = `
  .markdown-body h1,.markdown-body h2,.markdown-body h3,.markdown-body h4{
    font-family:'Space Grotesk',sans-serif;color:var(--accent-cyan,#00e5ff);margin:14px 0 8px;line-height:1.25;}
  .markdown-body h1{font-size:1.25rem;}
  .markdown-body h2{font-size:1.1rem;border-bottom:1px solid rgba(148,163,184,.18);padding-bottom:4px;}
  .markdown-body h3{font-size:1rem;}
  .markdown-body h4{font-size:.92rem;color:var(--text,#f8fafc);}
  .markdown-body p{margin:6px 0;line-height:1.6;}
  .markdown-body ul{margin:6px 0 10px;padding-left:20px;}
  .markdown-body li{margin:3px 0;line-height:1.55;}
  .markdown-body strong{color:var(--text,#f8fafc);font-weight:600;}
  .markdown-body em{color:var(--text-muted,#cbd5e1);font-style:italic;}
  .markdown-body code{font-family:'JetBrains Mono',monospace;font-size:.85em;background:rgba(148,163,184,.14);
    padding:1px 5px;border-radius:4px;color:var(--accent-cyan,#00e5ff);}
  .markdown-body blockquote{margin:8px 0;padding:8px 12px;border-left:3px solid var(--accent-cyan,#00e5ff);
    background:rgba(0,229,255,.07);border-radius:4px;}
  .markdown-body hr{border:none;border-top:1px solid rgba(148,163,184,.18);margin:14px 0;}`;
  document.head.appendChild(el);
})();

// ─── Tabs ───
function resizeCharts(container) {
  (container || document).querySelectorAll(".plotly-container .js-plotly-plot").forEach((el) => {
    Plotly.Plots.resize(el);
  });
}
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    $$(".tab-panel").forEach((p) => p.classList.add("hidden"));
    const panel = document.getElementById(`panel-${tab.dataset.tab}`);
    if (panel) panel.classList.remove("hidden");
    currentTab = tab.dataset.tab;
    setTimeout(() => resizeCharts(panel), 50);
  });
});

// ─── File Upload ───
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault(); uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFiles(fileInput.files);
});

async function handleFiles(files) {
  showLoading("Uploading files...");
  const form = new FormData();
  for (const f of files) form.append("files", f);
  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Upload failed (HTTP ${res.status})`);
    if (!data.session_id) throw new Error("The server did not return an upload session.");
    sessionId = data.session_id;
    renderFileBadges(data.files);
    await preprocess();
    await loadDataPreview();
    await loadChartColumns();
    hideLoading();
    showStatus("Files uploaded and preprocessed!", "success");
  } catch (err) {
    hideLoading();
    showStatus(`Upload failed: ${err.message}`, "error");
  }
}

function renderFileBadges(files) {
  const container = document.getElementById("fileBadges");
  container.replaceChildren();
  for (const fileName of files) {
    const badge = document.createElement("span");
    badge.className = "file-badge";
    badge.textContent = `📊 ${fileName}`;
    container.appendChild(badge);
  }
}

async function preprocess() {
  const form = new FormData();
  form.append("session_id", sessionId);
  const res = await fetch(`${API}/preprocess`, { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `Preprocessing failed (HTTP ${res.status})`);
  renderSidebarStats(data.summary);
  renderPrepReport(data.reports);

  tripleMode = data.triple_mode;
  sharedCols = data.shared_cols || [];
  viewKeys = Object.keys(data.views || { "unified": true });
  viewTitles = data.view_titles || {};

  // Show intersection / no-intersection message
  const msgContainer = document.getElementById("viewMessage");
  msgContainer.innerHTML = "";
  if (tripleMode) {
    if (sharedCols.length === 0) {
      msgContainer.innerHTML = `<div class="status-msg warning" style="margin-bottom:16px;">⚠️ No Common Columns Detected — The two datasets don't share column names. They will be analyzed independently as Dataset A and Dataset B.</div>`;
    } else {
      msgContainer.innerHTML = `<div class="status-msg info" style="margin-bottom:16px;">🔀 Triple-View Active — All tabs contain sub-tabs: Dataset A, Dataset B, and their Intersection on <strong>${sharedCols.length} shared column${sharedCols.length > 1 ? 's' : ''}</strong>: ${sharedCols.map(c => `<code style="color:var(--accent-cyan)">${escapeHtml(c)}</code>`).join(', ')}</div>`;
    }
  }

  // Render view sub-tabs
  renderViewSubtabs();
  // Initialize anomaly panel
  renderAnomalyPanel();
}

function renderSidebarStats(summary) {
  let totalRows = 0, totalCols = 0, dsCount = 0;
  for (const info of Object.values(summary)) {
    totalRows += info.rows; totalCols += info.cols; dsCount++;
  }
  document.getElementById("statDatasets").textContent = dsCount;
  document.getElementById("statColumns").textContent = totalCols;
  document.getElementById("statRecords").textContent = totalRows.toLocaleString();
}

function renderPrepReport(reports) {
  const container = document.getElementById("prepReport");
  if (!container) return;
  container.innerHTML = "";
  for (const [name, lines] of Object.entries(reports)) {
    const h = document.createElement("h4");
    h.style.cssText = "font-family:'Space Grotesk',sans-serif;margin:12px 0 6px;font-size:0.9rem;color:var(--accent-cyan);";
    h.textContent = `📂 ${name}`;
    container.appendChild(h);
    for (const line of lines) {
      const p = document.createElement("p");
      p.style.cssText = "font-size:0.82rem;color:var(--text-secondary);padding:2px 0;font-family:'JetBrains Mono',monospace;";
      p.textContent = line;
      container.appendChild(p);
    }
  }
}

// ─── View Sub-Tabs ───
function renderViewSubtabs() {
  const containers = $$(".view-subtabs");
  containers.forEach((container) => {
    container.innerHTML = "";
    if (viewKeys.length <= 1) return;

    const tabBar = document.createElement("div");
    tabBar.className = "tabs view-tabs";
    tabBar.style.marginBottom = "12px";

    viewKeys.forEach((vk, i) => {
      const label = viewTitles[vk] || vk;
      const short = label.length > 30 ? label.substring(0, 28) + "…" : label;
      const btn = document.createElement("button");
      btn.className = `tab view-tab ${i === 0 ? "active" : ""}`;
      btn.dataset.viewKey = vk;
      btn.textContent = short;
      btn.title = label;
      btn.addEventListener("click", () => {
        container.querySelectorAll(".view-tab").forEach(t => t.classList.remove("active"));
        btn.classList.add("active");
        // Hide all view panels in this group
        const parent = container.closest(".view-group") || container.parentElement;
        parent.querySelectorAll(".view-panel").forEach(p => p.classList.add("hidden"));
        const panel = parent.querySelector(`.view-panel[data-view="${vk}"]`);
        if (panel) panel.classList.remove("hidden");
      });
      tabBar.appendChild(btn);
    });
    container.appendChild(tabBar);
  });
}

function makeViewPanels(containerId, htmlFn) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  if (viewKeys.length <= 1) {
    // Single view — no sub-tabs
    container.innerHTML = `<div class="view-group">${htmlFn("unified")}</div>`;
    return;
  }
  // Triple view — add sub-tab bar and panels
  const subtabDiv = document.createElement("div");
  subtabDiv.className = "view-subtabs";
  container.appendChild(subtabDiv);

  const group = document.createElement("div");
  group.className = "view-group";
  viewKeys.forEach((vk, i) => {
    const panel = document.createElement("div");
    panel.className = `view-panel ${i === 0 ? "" : "hidden"}`;
    panel.dataset.view = vk;
    panel.innerHTML = htmlFn(vk);
    group.appendChild(panel);
  });
  container.appendChild(group);
  renderViewSubtabs();
}


// ─── Data Preview ───
async function loadDataPreview() {
  try {
    const res = await fetch(`${API}/data/${encodeURIComponent(sessionId)}/preview?rows=5`);
    if (!res.ok) throw new Error(`Preview request failed: ${res.status}`);
    const data = await res.json();
    const container = document.getElementById("dataPreview");
    container.replaceChildren();

    for (const [name, info] of Object.entries(data)) {
      const card = document.createElement("div");
      card.className = "card";

      const title = document.createElement("div");
      title.className = "card-title";
      title.append(document.createTextNode(`📂 ${name} `));
      const dimensions = document.createElement("span");
      dimensions.style.cssText = "margin-left:auto;font-size:0.75rem;color:var(--text-muted);font-family:'JetBrains Mono',monospace;";
      dimensions.textContent = `${info.shape[0]} rows × ${info.shape[1]} cols`;
      title.appendChild(dimensions);
      card.appendChild(title);

      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      const table = document.createElement("table");
      table.className = "data-table";
      const headers = Object.keys(info.data[0] || {});
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const header of headers) {
        const th = document.createElement("th");
        th.textContent = header;
        headerRow.appendChild(th);
      }
      thead.appendChild(headerRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (const row of info.data) {
        const tr = document.createElement("tr");
        for (const header of headers) {
          const td = document.createElement("td");
          td.textContent = row[header] == null ? "" : String(row[header]);
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      wrap.appendChild(table);
      card.appendChild(wrap);
      container.appendChild(card);
    }
    renderDatasetSelector(data);

    const firstDatasetName = Object.keys(data)[0];
    if (firstDatasetName) {
      fetchAndDisplayKPIs(firstDatasetName);
      const kpiTabBtn = document.querySelector('.tab[data-tab="kpi"]');
      if (kpiTabBtn) kpiTabBtn.click();
    }
  } catch (err) {
    console.error("Preview error:", err);
    showStatus(`Preview failed: ${err.message}`, "error");
  }
}

function renderDatasetSelector(datasets) {
  const sel = document.getElementById("dsSelect");
  sel.replaceChildren();
  for (const name of Object.keys(datasets)) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    sel.appendChild(option);
  }
}

// ─── Pipeline ───
async function runPipeline() {
  if (!sessionId) { showStatus("Upload files first!", "warning"); return; }
  showLoading("Running Multi-Agent Pipeline across all views...");
  clearStatus();

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("run_insight", "true");

  try {
    const res = await fetch(`${API}/pipeline/run`, { method: "POST", body: form });
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (parseErr) {
      hideLoading();
      showStatus(`Pipeline failed (HTTP ${res.status}). Check the server log for details.`, "error");
      return;
    }
    if (data.status === "complete") {
      await loadAllResults();
      hideLoading();
      showStatus("✅ Pipeline complete! Results loaded for all views.", "success");
    } else {
      hideLoading();
      showStatus("Pipeline failed. Check the server log for details.", "error");
    }
  } catch (err) {
    hideLoading();
    showStatus(`Pipeline network error: ${err.message}`, "error");
  }
}

async function loadAllResults() {
  try {
    const [resData, resCharts, resInsight, resRec] = await Promise.all([
      fetch(`${API}/results/${sessionId}`).then(r => r.json()),
      fetch(`${API}/charts/${sessionId}`).then(r => r.json()),
      fetch(`${API}/insight/${sessionId}`).then(r => r.json()),
      fetch(`${API}/recommender/${sessionId}`).then(r => r.json()),
    ]);

    if (resData.status !== "complete") return;
    viewKeys = resData.view_keys || viewKeys;
    viewTitles = resData.view_titles || viewTitles;
    tripleMode = resData.triple_mode || false;

    const views = resData.views || {};

    // KPIs
    makeViewPanels("kpiContent", (vk) => {
      const v = views[vk];
      if (!v) return "<p style='color:var(--text-muted)'>No results for this view.</p>";

      let html = '';
      for (const [dsName, kpi] of Object.entries(v.kpi || {})) {
        // ── Scalar summary cards ─────────────────────────────────────────────
        html += `<h3 style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;color:var(--accent-cyan);margin:16px 0 10px;">📂 ${escapeHtml(dsName)}</h3>`;
        html += '<div class="kpi-grid" style="grid-template-columns:repeat(5,1fr);margin-bottom:12px;">';
        const scalars = [
          { label: "Rows",            val: (kpi.rows ?? 0).toLocaleString(),   color: "var(--accent-cyan)"   },
          { label: "Columns",         val: (kpi.columns ?? kpi.cols ?? 0),      color: "var(--accent-violet)" },
          { label: "Numeric Cols",    val: (kpi.numeric_cols ?? 0),             color: "var(--accent-pink)"   },
          { label: "Null Values",     val: (kpi.nulls ?? 0).toLocaleString(),   color: "var(--accent-yellow)" },
          { label: "Null Rate",       val: kpi.null_rate ?? "—",               color: "var(--accent-red)"    },
        ];
        scalars.forEach(s => {
          html += `<div class="kpi-card">
            <div class="kpi-label">${s.label}</div>
            <div class="kpi-value" style="font-size:1.1rem;color:${s.color};-webkit-text-fill-color:${s.color};">${s.val}</div>
          </div>`;
        });
        html += '</div>';

        // ── Date range ───────────────────────────────────────────────────────
        if (kpi.date_range) {
          html += `<div style="background:var(--bg-elevated);border:1px solid var(--border-subtle);border-radius:8px;padding:8px 14px;margin-bottom:12px;font-size:0.82rem;">
            📅 <span style="color:var(--text-muted);">Date Range:</span>
            <span style="color:var(--accent-cyan);font-family:'JetBrains Mono',monospace;margin-left:8px;">${kpi.date_range.from} → ${kpi.date_range.to}</span>
          </div>`;
        }

        // ── Per-column stats table ───────────────────────────────────────────
        if (kpi.per_column && Object.keys(kpi.per_column).length > 0) {
          let rows = '';
          for (const [col, s] of Object.entries(kpi.per_column)) {
            const trend = kpi.time_based?.[col];
            const growthHtml = trend
              ? `<span style="color:${trend.period_growth_pct >= 0 ? '#22c55e' : '#ef4444'}">
                   ${trend.period_growth_pct >= 0 ? '▲' : '▼'} ${Math.abs(trend.period_growth_pct)}%
                 </span>`
              : '<span style="color:var(--text-muted)">—</span>';
            const peakHtml = trend
              ? `<span style="color:var(--accent-yellow);font-size:0.78rem;">${escapeHtml(trend.peak_period)}</span>`
              : '<span style="color:var(--text-muted)">—</span>';
            const volatility = s.cv_pct > 100 ? '🔴 High' : s.cv_pct > 50 ? '🟡 Medium' : '🟢 Low';
            rows += `<tr>
              <td style="font-family:'JetBrains Mono',monospace;color:var(--accent-cyan);">${escapeHtml(col)}</td>
              <td>${Number(s.total).toLocaleString()}</td>
              <td>${Number(s.mean).toLocaleString()}</td>
              <td>${Number(s.median).toLocaleString()}</td>
              <td>${Number(s.min).toLocaleString()}</td>
              <td>${Number(s.max).toLocaleString()}</td>
              <td>${volatility} (${s.cv_pct}%)</td>
              <td>${growthHtml}</td>
              <td>${peakHtml}</td>
            </tr>`;
          }
          html += `<div class="table-wrap">
            <table class="data-table">
              <thead><tr>
                <th>Column</th><th>Total</th><th>Mean</th><th>Median</th>
                <th>Min</th><th>Max</th><th>Volatility (CV)</th>
                <th>Period Growth</th><th>Peak Period</th>
              </tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>`;
        }
      }
      return html || "<p style='color:var(--text-muted)'>No KPI data available.</p>";
    });



    // Charts
    const chartsByView = resCharts.views || {};
    makeViewPanels("chartContent", (vk) => {
      const cv = chartsByView[vk];
      if (!cv || !cv.charts) return "";
      let html = "";
      for (const [dsName, cols] of Object.entries(cv.charts)) {
        html += `<h3 style="font-family:'Space Grotesk',sans-serif;font-size:1rem;margin:16px 0 12px;color:var(--accent-cyan);">📂 ${escapeHtml(dsName)}</h3><div class="chart-grid">`;
        let count = 0;
        for (const [col, chart] of Object.entries(cols)) {
          if (count >= 6) break;
          const overridden = chart.original_type !== chart.type;
          const badgeCls = overridden ? "corrected" : chart.guard_warning ? "warning" : "approved";
          const badgeText = overridden ? `🔄 ${chart.original_type} → ${chart.type}` : chart.guard_warning ? "🛡️ Guard" : `✅ ${chart.type}`;
          const chartId = `chart-${encodeURIComponent(vk)}-${encodeURIComponent(dsName)}-${encodeURIComponent(col)}`;
          html += `<div class="chart-container">
            <div class="chart-header"><span class="chart-title">${escapeHtml(col)}</span><span class="chart-badge ${badgeCls}">${escapeHtml(badgeText)}</span></div>
            <div class="plotly-container" id="${chartId}" style="width:100%;height:380px;"></div>
          </div>`;
          count++;
        }
        html += '</div>';
      }
      return html;
    });

    // Insights
    const insightByView = resInsight.views || {};
    const recByView = resRec.views || {};
    makeViewPanels("insightContent", (vk) => {
      const iv = insightByView[vk] || {};
      const insightText = iv.insight || "";

      // strip only a LEADING "[Agent] " marker — keep numeric ranges like [10 – 90] in the body
      const cleanInsight = insightText.replace(/^\s*\[[^\]]*\]\s*/, "").trim();
      const hasInsight = cleanInsight && !cleanInsight.startsWith("[") && cleanInsight.length > 20;

      let html = '<div class="card"><div class="card-title">📋 Strategic Report</div>';
      if (hasInsight) {
        if (cleanInsight.toLowerCase().includes("smaller model") || cleanInsight.toLowerCase().includes("ollama pull")) {
          html += `<div class="status-msg warning">⚠️ ${renderMarkdown(cleanInsight)}</div>`;
        } else {
          html += `<div class="insight-box markdown-body">${renderMarkdown(cleanInsight)}</div>`;
        }
      } else if (insightText.includes("Offline") || insightText.includes("offline")) {
        html += `<div class="status-msg warning">🤖 AI Engine Offline — Install and start <a href="https://ollama.ai" style="color:var(--accent-cyan);" target="_blank">Ollama</a>, then pull a model: <code>ollama pull llama3.2:1b</code></div>`;
      } else {
        html += "<p style='color:var(--text-muted);'>No insight available. Run the pipeline with AI enabled.</p>";
      }
      html += '</div>';
      return html;
    });

    // Anomaly panel
    renderAnomalyPanel();

    
    // Recommender tab - per-view sub-tabs (mirrors insightContent pattern)
    makeViewPanels("recContent", (vk) => {
      const rv = recByView[vk] || {};
      const recText = rv.recommender || "";
      const cleanRec = recText.replace(/^\s*\[[^\]]*\]\s*/, "").trim();
      const hasRec = cleanRec && !cleanRec.startsWith("[") && cleanRec.length > 20;

      let html = '<div class="card"><div class="card-title" style="color:var(--accent-yellow);">📋 AI Recommendations</div>';
      if (hasRec) {
        if (cleanRec.toLowerCase().includes("smaller model") || cleanRec.toLowerCase().includes("ollama pull")) {
          html += `<div class="status-msg warning">⚠️ ${renderMarkdown(cleanRec)}</div>`;
        } else {
          html += `<div class="insight-box markdown-body" style="border-left-color:var(--accent-yellow);">${renderMarkdown(cleanRec)}</div>`;
        }
      } else if (recText.includes("Offline") || recText.includes("offline")) {
        html += `<div class="status-msg warning">🤖 AI Engine Offline — Install and start <a href="https://ollama.ai" style="color:var(--accent-cyan);" target="_blank">Ollama</a>, then pull a model: <code>ollama pull llama3.2:1b</code></div>`;
      } else {
        html += "<p style='color:var(--text-muted);'>No recommendations available. Run the pipeline with AI enabled.</p>";
      }
      html += '</div>';
      return html;
    });

    // Render Plotly charts after DOM update
    setTimeout(() => {
      for (const [vk, cv] of Object.entries(chartsByView)) {
        if (!cv || !cv.charts) continue;
        for (const [dsName, cols] of Object.entries(cv.charts)) {
          for (const [col, chart] of Object.entries(cols)) {
            const chartId = `chart-${encodeURIComponent(vk)}-${encodeURIComponent(dsName)}-${encodeURIComponent(col)}`;
            const el = document.getElementById(chartId);
            if (el && chart.json) {
              try {
                const layout = Object.assign({}, chart.json.layout || {});
                layout.autosize = true;
                layout.width = el.clientWidth || undefined;
                layout.margin = layout.margin || {};
                layout.margin.l = Math.min(layout.margin.l || 50, 55);
                layout.margin.r = Math.min(layout.margin.r || 20, 25);
                Plotly.react(el, chart.json.data, layout, { displayModeBar: false, responsive: true });
              }
              catch (e) { console.error("Plotly error:", e); }
            }
          }
        }
      }
    }, 200);

    // Refresh chart builder columns
    await loadChartColumns();
  } catch (err) { console.error("Results error:", err); }
}

// ─── Interactive Chart Builder ───
let chartColumnCache = {};  // {dsName: [{name, type}, ...]}

function updateYVisibility(chartType) {
  const yEl    = document.getElementById("chartY");
  const yLabel = document.getElementById("yOptional");
  const aggRow = document.getElementById("chartAggFuncRow");
  const ctEl   = document.getElementById("chartType");

  if (!yEl || !ctEl) return; // DOM not ready yet

  if (!chartType) chartType = ctEl.value;
  const yVal    = yEl.value;
  const yRow    = yEl.closest("div");
  const needsY  = ["scatter", "line", "bar"].includes(chartType);

  if (yRow) {
    if (chartType === "box") {
      yRow.style.opacity = "1";
      if (yLabel) yLabel.style.display = "inline";
    } else if (needsY) {
      yRow.style.opacity = "1";
      if (yLabel) yLabel.style.display = "none";
    } else {
      yRow.style.opacity = "0.4";
      if (yLabel) yLabel.style.display = "inline";
    }
  }

  // Show agg selector only when chart type supports grouping AND a Y column is selected
  if (aggRow) {
    const aggSupported = ["bar", "line", "pie", "histogram"].includes(chartType);
    aggRow.style.display = (aggSupported && yVal) ? "" : "none";
  }
}

function onChartDatasetChange() {
  const dsSel = document.getElementById("chartDataset");
  const xSel  = document.getElementById("chartX");
  const ySel  = document.getElementById("chartY");
  if (!dsSel || !xSel || !ySel) return;

  const ds       = dsSel.value;
  const cols     = chartColumnCache[ds] || [];
  const currentX = xSel.value;
  const currentY = ySel.value;

  // Populate X — all columns available
  if (cols.length === 0) {
    xSel.innerHTML = `<option value="">— No columns loaded —</option>`;
  } else {
    xSel.innerHTML = cols.map(c =>
      `<option value="${escapeHtml(c.name)}" ${c.name === currentX ? "selected" : ""}>${escapeHtml(c.name)} (${escapeHtml(c.type)})</option>`
    ).join("");
  }

  // Populate Y — numeric first, then categorical, with "None" default
  const numCols = cols.filter(c => c.type === "numeric");
  const catCols = cols.filter(c => c.type !== "numeric");
  let yOpts = `<option value="">— None (count only) —</option>`;
  if (numCols.length) {
    yOpts += `<optgroup label="📊 Numeric">` +
      numCols.map(c => `<option value="${escapeHtml(c.name)}" ${c.name === currentY ? "selected" : ""}>${escapeHtml(c.name)}</option>`).join("") +
      `</optgroup>`;
  }
  if (catCols.length) {
    yOpts += `<optgroup label="🔤 Categorical">` +
      catCols.map(c => `<option value="${escapeHtml(c.name)}" ${c.name === currentY ? "selected" : ""}>${escapeHtml(c.name)}</option>`).join("") +
      `</optgroup>`;
  }
  ySel.innerHTML = yOpts;
  updateYVisibility();
}

function onChartTypeChange() {
  updateYVisibility(document.getElementById("chartType").value);
}

async function loadChartColumns() {
  if (!sessionId) return;
  const dsSel = document.getElementById("chartDataset");
  const ySel  = document.getElementById("chartY");
  if (!dsSel) return;

  try {
    const res = await fetch(`${API}/data/${sessionId}/columns`);
    if (!res.ok) { console.warn("loadChartColumns: server returned", res.status); return; }
    const data = await res.json();
    chartColumnCache = {};

    // Flatten all view datasets into a single lookup
    for (const [vk, info] of Object.entries(data.views || {})) {
      for (const [dsName, dsInfo] of Object.entries(info.datasets || {})) {
        if (!chartColumnCache[dsName]) {
          chartColumnCache[dsName] = dsInfo.columns || [];
        }
      }
    }

    const dsNames = Object.keys(chartColumnCache);
    const currentDs = dsSel.value;

    if (dsNames.length === 0) {
      dsSel.innerHTML = `<option value="">— No datasets loaded —</option>`;
      if (ySel) ySel.innerHTML = `<option value="">— None —</option>`;
      return;
    }

    dsSel.innerHTML = dsNames.map(n =>
      `<option value="${escapeHtml(n)}" ${n === currentDs ? "selected" : ""}>${escapeHtml(n)}</option>`
    ).join("");

    onChartDatasetChange(); // populates X and Y dropdowns
  } catch (err) {
    console.error("Chart columns error:", err);
  }
}

// Color palette per chart type for history badges
const CHART_TYPE_COLORS = {
  bar: "var(--accent-cyan)", line: "var(--accent-violet)", scatter: "var(--accent-pink)",
  pie: "var(--accent-yellow)", histogram: "var(--accent-green)", box: "#f97316"
};

// Swap X ↔ Y columns
function swapXY() {
  const xSel = document.getElementById("chartX");
  const ySel = document.getElementById("chartY");
  if (!xSel || !ySel) return;
  const xVal = xSel.value;
  const yVal = ySel.value;
  if (!yVal) return;
  // Try to set X to old Y value
  const xOpt = Array.from(xSel.options).find(o => o.value === yVal);
  if (xOpt) xSel.value = yVal;
  // Try to set Y to old X value
  const yOpt = Array.from(ySel.options).find(o => o.value === xVal);
  if (yOpt) ySel.value = xVal;
  updateYVisibility();
}

// Clear all saved charts from history
function clearChartHistory() {
  const h = document.getElementById("chartHistory");
  if (h) h.innerHTML = "";
  const d = document.getElementById("chartBuilderDisplay");
  if (d) d.innerHTML = `<p style="color:var(--text-muted);">Select your options above and click "Build Chart".</p>`;
}

async function buildInteractiveChart() {
  // ── Safely read all inputs ──────────────────────────────────────────
  const dsEl    = document.getElementById("chartDataset");
  const xEl     = document.getElementById("chartX");
  const yEl     = document.getElementById("chartY");
  const ctypeEl = document.getElementById("chartType");
  const titleEl = document.getElementById("chartTitle");
  const aggEl   = document.getElementById("chartAggFunc");

  if (!dsEl || !xEl || !ctypeEl) {
    showStatus("Chart builder controls not found — please reload the page.", "error");
    return;
  }

  const ds      = dsEl.value;
  const xCol    = xEl.value;
  const yCol    = yEl ? yEl.value : "";
  const ctype   = ctypeEl.value;
  const title   = titleEl ? titleEl.value : "";
  const aggFunc = aggEl ? aggEl.value : "mean";

  // ── Validation ──────────────────────────────────────────────────────
  if (!sessionId) {
    showStatus("Upload a file first!", "warning");
    return;
  }
  if (!ds) {
    showStatus("Please select a dataset.", "warning");
    return;
  }
  if (!xCol) {
    showStatus("Please select an X axis column.", "warning");
    return;
  }

  // ── Build request ───────────────────────────────────────────────────
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("dataset_name", ds);
  form.append("x_column", xCol);
  form.append("y_column", yCol || "");
  form.append("chart_type", ctype);
  form.append("agg_func", aggFunc);
  form.append("title", title);

  showLoading("Building chart...");
  try {
    const res = await fetch(`${API}/chart/build`, { method: "POST", body: form });

    // Handle non-OK responses gracefully
    if (!res.ok) {
      let errMsg = `Server error ${res.status}`;
      try {
        const errData = await res.json();
        errMsg = errData.detail || errMsg;
      } catch (_) {}
      throw new Error(errMsg);
    }

    const data = await res.json();
    hideLoading();

    // ── Validate response structure ────────────────────────────────────
    if (!data || !data.fig) {
      throw new Error("Server returned an empty chart response.");
    }
    const figData   = Array.isArray(data.fig.data)   ? data.fig.data   : [];
    const figLayout = (data.fig.layout && typeof data.fig.layout === "object")
                        ? data.fig.layout : {};

    // ── Build display labels ───────────────────────────────────────────
    const ctypeLabel = ctypeEl.options[ctypeEl.selectedIndex]
                         ? ctypeEl.options[ctypeEl.selectedIndex].text
                         : ctype;
    const yLabel   = yCol ? ` vs ${yCol}` : "";
    const aggLabel = (yCol && aggFunc && aggFunc !== "count") ? ` [${aggFunc}]` : "";
    const autoTitle = title || `${ctypeLabel}: ${xCol}${yLabel}${aggLabel}`;
    const badgeColor = CHART_TYPE_COLORS[ctype] || "var(--accent-cyan)";

    // Unique ID for this chart in history
    const chartId = `custom-chart-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    // ── Shared Plotly layout overrides ─────────────────────────────────
    const layout = Object.assign({}, figLayout);
    layout.autosize = true;
    layout.margin   = { l: 55, r: 20, t: 50, b: 60 };

    // ── Update the "latest chart" preview ─────────────────────────────
    const displayEl = document.getElementById("chartBuilderDisplay");
    if (displayEl) {
      displayEl.innerHTML = `
        <div class="chart-container">
          <div class="chart-header">
            <span class="chart-title">${escapeHtml(autoTitle)}</span>
            <span class="chart-badge approved" style="background:${badgeColor};color:#070912;">${escapeHtml(ctype)}</span>
          </div>
          <div class="plotly-container" id="latest-chart" style="width:100%;height:420px;"></div>
        </div>`;
      const latestEl = document.getElementById("latest-chart");
      if (latestEl && typeof Plotly !== "undefined") {
        Plotly.react(latestEl, figData, layout, { displayModeBar: true, responsive: true });
      }
    }

    // ── Prepend a copy to history ──────────────────────────────────────
    const historyContainer = document.getElementById("chartHistory");
    if (historyContainer) {
      const historyCard = document.createElement("div");
      historyCard.className = "chart-container";
      historyCard.style.marginBottom = "12px";
      historyCard.innerHTML = `
        <div class="chart-header">
          <span class="chart-title">${escapeHtml(autoTitle)}</span>
          <span style="display:flex;align-items:center;gap:6px;">
            <span class="chart-badge approved" style="background:${badgeColor};color:#070912;">${escapeHtml(ctype)}</span>
            <button onclick="this.closest('.chart-container').remove()"
              style="background:none;border:none;color:var(--accent-red);cursor:pointer;font-size:1.1rem;line-height:1;"
              title="Remove chart">✕</button>
          </span>
        </div>
        <div class="plotly-container" id="${chartId}" style="width:100%;height:360px;"></div>`;
      historyContainer.prepend(historyCard);

      // Render history card after DOM update
      setTimeout(() => {
        const histEl = document.getElementById(chartId);
        if (histEl && typeof Plotly !== "undefined") {
          Plotly.react(histEl, figData, layout, { displayModeBar: false, responsive: true });
        }
      }, 100);
    }

    showStatus(`✅ Chart built — ${autoTitle}`, "success");

  } catch (err) {
    hideLoading();
    showStatus(`Chart build failed: ${err.message}`, "error");
    console.error("buildInteractiveChart error:", err);
  }
}

// Set up chart builder event listeners (script runs at end of body, DOM is ready)
(function initChartBuilder() {
  const chartTypeSel = document.getElementById("chartType");
  const chartDsSel   = document.getElementById("chartDataset");
  const chartYSel    = document.getElementById("chartY");
  const chartXSel    = document.getElementById("chartX");

  if (chartTypeSel) {
    chartTypeSel.addEventListener("change", () => updateYVisibility(chartTypeSel.value));
    // Run once on load to set initial agg row visibility
    updateYVisibility(chartTypeSel.value);
  }
  if (chartDsSel) {
    chartDsSel.addEventListener("change", onChartDatasetChange);
  }
  if (chartYSel) {
    chartYSel.addEventListener("change", () => updateYVisibility());
  }
  // Allow pressing Enter in the title field to trigger build
  const titleEl = document.getElementById("chartTitle");
  if (titleEl) {
    titleEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") buildInteractiveChart();
    });
  }
})();

// ─── Anomaly Detection (per-view) ───
function renderAnomalyPanel() {
  makeViewPanels("anomalyContent", (vk) => {
    if (!vk) vk = viewKeys[0] || "unified";
    const resultsId = `anomaly-results-${vk}`;
    const explainId = `anomaly-explain-${vk}`;
    return `
      <div>
        <div style="display:flex;gap:12px;margin-bottom:8px;flex-wrap:wrap;align-items:center;">
          <button class="btn btn-primary" onclick="runAnomalyView('${vk}')">🚀 Run Anomaly Scanner</button>
          <button class="btn btn-secondary" onclick="explainAnomalyView('${vk}')">🧠 Explain with AI</button>
          <label style="color:var(--text-muted);font-size:0.85rem;display:flex;align-items:center;gap:6px;">
            Horizon:
            <select id="horizon-${vk}" style="background:var(--bg-card);color:var(--text-primary);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:0.85rem;">
              <option value="7">7 periods</option>
              <option value="30" selected>30 periods</option>
              <option value="60">60 periods</option>
              <option value="90">90 periods</option>
            </select>
          </label>
        </div>
        <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:16px;">
          Predicts future anomalies using Prophet for time-series data or bootstrapped IsolationForest for transactional data.
        </p>
        <div id="${resultsId}"><p style="color:var(--text-muted);">Click "Run Anomaly Scanner" to predict anomalies in this view.</p></div>
        <div id="${explainId}"></div>
      </div>`;
  });
}

async function runAnomalyView(vk) {
  if (!sessionId) return;
  showLoading("Running predictive anomaly detection...");
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("view_key", vk);
  const horizonEl = document.getElementById(`horizon-${vk}`);
  form.append("horizon", horizonEl ? horizonEl.value : "30");
  try {
    const res = await fetch(`${API}/anomaly/run`, { method: "POST", body: form });
    if (!res.ok) { hideLoading(); showStatus(`Anomaly failed: ${res.statusText}`, "error"); return; }
    const data = await res.json();
    hideLoading();
    const container = document.getElementById(`anomaly-results-${vk}`);
    if (!container) return;
    if (data.error) { container.innerHTML = `<div class="status-msg warning">${escapeHtml(data.error)}</div>`; return; }
    const methodLabel = data.method === "prophet" ? "📈 Prophet (Time-Series)" : "🔁 Bootstrap Simulation";
    container.innerHTML = `
      <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:16px;">
        <div class="kpi-card">
          <div class="kpi-label">PREDICTED ANOMALIES</div>
          <div class="kpi-value" style="font-size:1.3rem;color:var(--accent-red);-webkit-text-fill-color:var(--accent-red);">${escapeHtml(data.predicted_anomalies)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">HORIZON</div>
          <div class="kpi-value" style="font-size:1rem;">${escapeHtml(data.horizon)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">CONFIDENCE</div>
          <div class="kpi-value" style="font-size:1.3rem;">${escapeHtml(data.confidence)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">METHOD</div>
          <div class="kpi-value" style="font-size:0.85rem;">${methodLabel}</div>
        </div>
      </div>`;
    if (data.per_dataset && data.per_dataset.length > 0) {
      const rows = data.per_dataset.map(d => {
        const mIcon = d.method === "prophet" ? "📈" : "🔁";
        return `<tr>
          <td>${escapeHtml(d.dataset)}</td>
          <td>${mIcon} ${escapeHtml(d.method)}</td>
          <td style="color:var(--accent-red);font-weight:600;">${escapeHtml(d.predicted_anomalies)}</td>
          <td>${escapeHtml(d.horizon || "-")}</td>
          <td style="color:var(--text-muted);font-size:0.85rem;">${escapeHtml(d.time_column || "none")}</td>
        </tr>`;
      }).join("");
      container.innerHTML += `
        <div class="card" style="margin-top:4px;">
          <div class="card-title">📊 Per-Dataset Breakdown</div>
          <div class="table-wrap">
            <table class="data-table">
              <thead><tr><th>Dataset</th><th>Method</th><th>Predicted Anomalies</th><th>Horizon</th><th>Time Column</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>`;

        // --- MATCHED LAYOUT: Anomaly Highlight Preview Section ---
      let previewHtml = '';
      data.per_dataset.forEach(d => {
         if (d.method === "prophet" && d.columns) {
            d.columns.forEach(col => {
               if (col.predicted_list && col.predicted_list.length > 0) {
                  previewHtml += `
                    <div style="margin-top: 24px; font-family: 'Inter', sans-serif;">
                      <div style="background: rgba(239, 68, 68, 0.15); border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                          <h4 style="margin: 0; font-size: 0.95rem; color: #f8fafc; font-weight: 600;">⚠️ Horizon Anomalies Detected</h4>
                          <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94a3b8;">Dataset: <span style="color:#ef4444; font-weight:600;">${escapeHtml(d.dataset)}</span> | Column: <code>${escapeHtml(col.column)}</code></p>
                        </div>
                        <div style="background: #ef4444; color: #ffffff; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                          ${col.predicted_list.length} ANOMALIES
                        </div>
                      </div>

                      <div style="background: #0f172a; border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; max-height: 300px; overflow-y: auto;">
                        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.82rem;">
                          <thead>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02);">
                              <th style="padding: 10px 16px; color: #94a3b8; font-weight: 500;">Predicted Timestamp</th>
                              <th style="padding: 10px 16px; color: #94a3b8; font-weight: 500;">Expected Target Metric Value</th>
                              <th style="padding: 10px 16px; color: #ef4444; font-weight: 600; text-align: right;">Status Threshold</th>
                            </tr>
                          </thead>
                          <tbody>`;

                  col.predicted_list.forEach(p => {
                     previewHtml += `
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04); transition: background 0.2s;" onmouseover="this.style.background='rgba(239, 68, 68, 0.03)'" onmouseout="this.style.background='transparent'">
                              <td style="padding: 12px 16px; color: #cbd5e1; font-family: 'JetBrains Mono', monospace;">📅 ${escapeHtml(p.date)}</td>
                              <td style="padding: 12px 16px; color: #f1f5f9; font-weight: 500;">📈 ${escapeHtml(p.expected)}</td>
                              <td style="padding: 12px 16px; text-align: right;">
                                <span style="background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;">ANOMALY RISK</span>
                              </td>
                            </tr>`;
                  });

                  previewHtml += `
                          </tbody>
                        </table>
                      </div>
                    </div>`;
               }
            });
         } else if (d.simulation && d.simulation.predicted_list && d.simulation.predicted_list.length > 0) {
            // Reusable Layout block for Simulated Profile Data if available
            previewHtml += `
                    <div style="margin-top: 24px; font-family: 'Inter', sans-serif;">
                      <div style="background: rgba(239, 68, 68, 0.15); border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                          <h4 style="margin: 0; font-size: 0.95rem; color: #f8fafc; font-weight: 600;">⚠️ Simulated Profile Anomaly Signatures</h4>
                          <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94a3b8;">Dataset: <span style="color:#ef4444; font-weight:600;">${escapeHtml(d.dataset)}</span></p>
                        </div>
                      </div>
                      <div style="background: #0f172a; border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; max-height: 300px; overflow-y: auto;">
                        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.82rem;">
                          <thead>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02);">
                              <th style="padding: 10px 16px; color: #94a3b8; font-weight: 500;">Simulated Features Context Signature</th>
                              <th style="padding: 10px 16px; color: #ef4444; font-weight: 600; text-align: right;">Status</th>
                            </tr>
                          </thead>
                          <tbody>`;

            d.simulation.predicted_list.forEach(p => {
               previewHtml += `
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                              <td style="padding: 12px 16px; color: #cbd5e1; font-family: 'JetBrains Mono', monospace;">⚠️ ${escapeHtml(p.expected)}</td>
                              <td style="padding: 12px 16px; text-align: right;">
                                <span style="background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">CRITICAL</span>
                              </td>
                            </tr>`;
            });

            previewHtml += `
                          </tbody>
                        </table>
                      </div>
                    </div>`;
         }
      });

      if (previewHtml) {
         container.innerHTML += `
           <div class="card" style="margin-top:20px; border-top: 2px solid #ef4444;">
             <div class="card-title" style="color: #f8fafc; display: flex; align-items: center; gap: 8px;">
               <span style="color: #ef4444;">🚨</span> Horizon Execution Forecast Breakdown
             </div>
             ${previewHtml}
           </div>`;
      }
      // ---------------------------------------------------------
    }
  } catch (err) { hideLoading(); showStatus(`Anomaly failed: ${err.message}`, "error"); }
}

async function explainAnomalyView(vk) {
  if (!sessionId) return;
  showLoading("AI analyzing anomalies...");
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("view_key", vk);
  try {
    const res = await fetch(`${API}/anomaly/explain`, { method: "POST", body: form });
    const data = await res.json();
    hideLoading();
    const container = document.getElementById(`anomaly-explain-${vk}`);
    if (!container) return;
    if (data.explanation) container.innerHTML =
      `<div class="insight-box markdown-body" style="border-left-color:var(--accent-pink);margin-top:12px;">${renderMarkdown(data.explanation)}</div>`;
  } catch (err) { hideLoading(); showStatus(`Explain failed: ${err.message}`, "error"); }
}

// ─── Active View Helper ───
function getActiveView() {
  if (viewKeys.length <= 1) return viewKeys[0] || "unified";
  const activeTab = document.querySelector(".view-tab.active");
  return activeTab ? activeTab.dataset.viewKey : viewKeys[0];
}

// ─── Recommender ───
async function findSimilar() {
  if (!sessionId) { showStatus("Upload files first!", "warning"); return; }
  const ds = document.getElementById("dsSelect").value;
  const rowIdx = parseInt(document.getElementById("rowIndex").value) || 0;
  showLoading("Finding similar records...");
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("dataset", ds);
  form.append("row_index", rowIdx);
  form.append("top_n", 5);
  try {
    const res = await fetch(`${API}/recommender/similar`, { method: "POST", body: form });
    const data = await res.json();
    hideLoading();
    const container = document.getElementById("similarResults");
    container.replaceChildren();
    if (data.error) {
      const message = document.createElement("div");
      message.className = "status-msg warning";
      message.textContent = data.error;
      container.appendChild(message);
      return;
    }
    if (!data.matches || !data.matches.length) {
      const message = document.createElement("div");
      message.className = "status-msg info";
      message.textContent = "No similar records found.";
      container.appendChild(message);
      return;
    }

    const status = document.createElement("div");
    status.className = "status-msg success";
    status.style.marginTop = "12px";
    status.textContent = `✅ Top ${data.matches.length} matches for Row ${rowIdx}`;
    container.appendChild(status);

    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const table = document.createElement("table");
    table.className = "data-table";
    const headers = Object.keys(data.matches[0]);
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const header of headers) {
      const th = document.createElement("th");
      th.textContent = header;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (const record of data.matches) {
      const tr = document.createElement("tr");
      for (const header of headers) {
        const td = document.createElement("td");
        td.textContent = record[header] == null ? "" : String(record[header]);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    container.appendChild(wrap);
  } catch (err) {
    hideLoading();
    showStatus(`Similarity search failed: ${err.message}`, "error");
  }
}

// ─── PDF Export ───
async function downloadPDF() {
  if (!sessionId) { showStatus("Upload and run pipeline first!", "warning"); return; }
  const activeView = getActiveView();
  showLoading("Generating PDF...");
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("view_key", activeView);
  try {
    const res = await fetch(`${API}/pdf/generate`, { method: "POST", body: form });
    if (!res.ok) throw new Error("PDF generation failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "Executive_Report.pdf"; a.click();
    URL.revokeObjectURL(url);
    hideLoading();
    showStatus("✅ PDF downloaded!", "success");
  } catch (err) { hideLoading(); showStatus(`PDF failed: ${err.message}`, "error"); }
}

// ─── LLM Status ───
async function checkLLM() {
  const el = document.getElementById("llmStatus");
  if (!el) return;
  try {
    const res = await fetch(`${API}/llm/status`);
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    const isOnline = data.online === true;
    const statusDetail = String(data.status || (isOnline ? "online" : "offline"));
    el.className = `llm-status ${isOnline ? "online" : "offline"}`;
    let label = isOnline ? "ONLINE" : "OFFLINE";
    let detail = isOnline ? "Ollama Ready" : "Start Ollama";
    if (statusDetail.startsWith("error:")) {
      el.className = "llm-status offline";
      label = "ERROR";
      detail = statusDetail.replace("error:", "").trim();
    }
    el.replaceChildren();
    const dot = document.createElement("span"); dot.className = "dot"; el.appendChild(dot);
    el.appendChild(document.createTextNode(` ${label} `));
    const detailEl = document.createElement("span"); detailEl.style.cssText = "margin-left:auto;font-size:0.65rem;color:var(--text-muted);"; detailEl.textContent = detail; el.appendChild(detailEl);
  } catch {
    el.className = "llm-status offline";
    el.replaceChildren();
    const dot = document.createElement("span"); dot.className = "dot"; el.appendChild(dot);
    const detailEl = document.createElement("span"); detailEl.style.cssText = "margin-left:auto;font-size:0.65rem;color:var(--text-muted);"; detailEl.textContent = "Server unreachable"; el.appendChild(detailEl);
  }
}

// ─── Init ───
checkLLM();
setInterval(checkLLM, 15000);

// ─── Executive Business KPIs Logic ───
async function fetchAndDisplayKPIs(datasetName) {
  const container = document.getElementById('businessKpiContainer');
  if (!container) return;
  container.innerHTML = '<p style="color:var(--text-muted); font-size:0.9rem;">Calculating KPIs...</p>';
  try {
    const res = await fetch(`${API}/kpis?session_id=${encodeURIComponent(sessionId)}&dataset_name=${encodeURIComponent(datasetName)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const kpis = data.kpis;
    if (!kpis) { container.innerHTML = '<p style="color:var(--text-muted);">No KPI data returned.</p>'; return; }

    container.innerHTML = '';

    // ── Top-level scalar cards (rows, columns, nulls, null_rate) ─────────────
    const scalarKeys = ["rows", "columns", "numeric_cols", "nulls", "null_rate"];
    const scalarLabels = {
      rows: "Total Rows", columns: "Total Columns",
      numeric_cols: "Numeric Columns", nulls: "Null Values", null_rate: "Null Rate"
    };
    const scalarColors = ["var(--accent-cyan)", "var(--accent-violet)", "var(--accent-pink)", "var(--accent-yellow)", "var(--accent-red)"];
    scalarKeys.forEach((key, i) => {
      if (kpis[key] === undefined) return;
      const card = document.createElement('div');
      card.className = 'kpi-card';
      card.innerHTML = `
        <div class="kpi-label">${escapeHtml(scalarLabels[key] || key.toUpperCase())}</div>
        <div class="kpi-value" style="font-size:1.4rem;color:${scalarColors[i]};-webkit-text-fill-color:${scalarColors[i]};">
          ${escapeHtml(typeof kpis[key] === 'number' ? kpis[key].toLocaleString() : kpis[key])}
        </div>`;
      container.appendChild(card);
    });

    // ── Date range banner ────────────────────────────────────────────────────
    if (kpis.date_range) {
      const banner = document.createElement('div');
      banner.style.cssText = 'grid-column:1/-1;background:var(--bg-elevated);border:1px solid var(--border-subtle);border-radius:8px;padding:10px 16px;display:flex;gap:24px;align-items:center;';
      banner.innerHTML = `
        <span style="color:var(--text-muted);font-size:0.8rem;">📅 DATE RANGE</span>
        <span style="color:var(--accent-cyan);font-family:'JetBrains Mono',monospace;font-size:0.9rem;">
          ${escapeHtml(kpis.date_range.from)} → ${escapeHtml(kpis.date_range.to)}
        </span>`;
      container.appendChild(banner);
    }



  } catch (err) {
    console.error("KPI Error:", err);
    container.textContent = `Failed to load KPIs: ${err.message}`;
  }
}

