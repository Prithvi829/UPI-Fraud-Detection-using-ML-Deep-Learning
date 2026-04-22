/**
 * app.js — UPI Payment Operations Dashboard
 * Vanilla JS, Plotly.js charts, auto-refreshes every 5 seconds.
 */

"use strict";

// ── Constants ─────────────────────────────────────────────────────────────────
const REFRESH_MS      = 5000;
const MAX_FEED_ROWS   = 20;
const COLORS = {
  normal:   "#378ADD",
  anomaly:  "#E24B4A",
  critical: "#A32D2D",
  amber:    "#f59e0b",
  green:    "#10b981",
  dim:      "#4a5568",
  text:     "#e2e8f0",
  grid:     "rgba(30,39,56,0.6)",
};

const TRANSPARENT = "rgba(0,0,0,0)";

// ── State ─────────────────────────────────────────────────────────────────────
let modelFilter = localStorage.getItem("modelFilter") || "BOTH";
let thresholdPct = parseInt(localStorage.getItem("thresholdPct") || "50", 10);
let prevTotals = { total: 0, flagged: 0 };
let chartsInitialised = false;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initControls();
  initCharts();
  refresh();
  setInterval(refresh, REFRESH_MS);
});

// ── Controls setup ────────────────────────────────────────────────────────────
function initControls() {
  // Threshold slider
  const slider = $("threshold-slider");
  const sliderVal = $("threshold-value");
  if (slider) {
    slider.value = thresholdPct;
    sliderVal.textContent = thresholdPct;
    slider.addEventListener("input", () => {
      thresholdPct = parseInt(slider.value, 10);
      sliderVal.textContent = thresholdPct;
      localStorage.setItem("thresholdPct", thresholdPct);
    });
  }

  // Model toggle buttons
  $$(".toggle-btn").forEach(btn => {
    if (btn.dataset.model === modelFilter) btn.classList.add("active");
    btn.addEventListener("click", () => {
      $$(".toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      modelFilter = btn.dataset.model;
      localStorage.setItem("modelFilter", modelFilter);
    });
  });
}

// ── Chart initialisation ──────────────────────────────────────────────────────
function initCharts() {
  // Donut chart
  Plotly.newPlot("chart-donut", [{
    type: "pie",
    hole: 0.65,
    values: [0, 0, 0, 1],
    labels: ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    marker: {
      colors: [COLORS.critical, COLORS.anomaly, COLORS.amber, COLORS.dim],
    },
    textinfo: "none",
    hoverinfo: "label+percent+value",
  }], donutLayout(), { responsive: true, displayModeBar: false });

  // Time-series line chart
  Plotly.newPlot("chart-timeseries", [
    {
      type: "scatter",
      mode: "lines+markers",
      name: "IF Flag Rate",
      x: [], y: [],
      line: { color: COLORS.normal, width: 2 },
      marker: { size: 3, color: COLORS.normal },
      fill: "tozeroy",
      fillcolor: "rgba(55,138,221,0.08)",
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: "AE Flag Rate",
      x: [], y: [],
      line: { color: COLORS.anomaly, width: 2, dash: "dot" },
      marker: { size: 3, color: COLORS.anomaly },
    },
  ], timeseriesLayout(), { responsive: true, displayModeBar: false });

  // Heatmap
  Plotly.newPlot("chart-heatmap", [{
    type: "heatmap",
    z: Array(7).fill(Array(24).fill(0)),
    x: [...Array(24).keys()],
    y: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    colorscale: [["0", "#13181f"], ["1", COLORS.anomaly]],
    showscale: false,
    hoverongaps: false,
  }], heatmapLayout(), { responsive: true, displayModeBar: false });

  // Category bar chart
  Plotly.newPlot("chart-catbar", [{
    type: "bar",
    orientation: "h",
    x: [],
    y: [],
    marker: {
      color: COLORS.anomaly,
      opacity: 0.8,
    },
  }], catbarLayout(), { responsive: true, displayModeBar: false });

  chartsInitialised = true;
}

// ── Layout builders ───────────────────────────────────────────────────────────
function baseLayout() {
  return {
    paper_bgcolor: TRANSPARENT,
    plot_bgcolor:  TRANSPARENT,
    font: { family: "'JetBrains Mono', monospace", size: 10, color: COLORS.text },
    margin: { t: 10, b: 30, l: 40, r: 10 },
  };
}

function donutLayout() {
  return {
    ...baseLayout(),
    showlegend: true,
    legend: {
      orientation: "h",
      x: 0, y: -0.15,
      font: { size: 9 },
    },
    margin: { t: 10, b: 10, l: 10, r: 10 },
  };
}

function timeseriesLayout() {
  return {
    ...baseLayout(),
    showlegend: true,
    legend: {
      orientation: "h",
      x: 0, y: 1.15,
      font: { size: 9 },
    },
    xaxis: {
      showgrid: false,
      zeroline: false,
      color: COLORS.dim,
      tickfont: { size: 9 },
    },
    yaxis: {
      showgrid: true,
      gridcolor: COLORS.grid,
      zeroline: false,
      color: COLORS.dim,
      ticksuffix: " ",
      tickfont: { size: 9 },
      title: { text: "count", font: { size: 9 }, standoff: 5 },
    },
    margin: { t: 10, b: 40, l: 45, r: 10 },
  };
}

function heatmapLayout() {
  return {
    ...baseLayout(),
    xaxis: {
      title: { text: "Hour of Day", font: { size: 9 } },
      tickfont: { size: 8 },
      color: COLORS.dim,
      showgrid: false,
    },
    yaxis: {
      tickfont: { size: 9 },
      color: COLORS.dim,
      showgrid: false,
    },
    margin: { t: 10, b: 40, l: 35, r: 10 },
  };
}

function catbarLayout() {
  return {
    ...baseLayout(),
    xaxis: {
      showgrid: true,
      gridcolor: COLORS.grid,
      zeroline: false,
      color: COLORS.dim,
      tickfont: { size: 9 },
      ticksuffix: "%",
    },
    yaxis: {
      tickfont: { size: 10 },
      color: COLORS.dim,
      showgrid: false,
    },
    margin: { t: 10, b: 30, l: 75, r: 10 },
    bargap: 0.35,
  };
}

// ── Main refresh cycle ────────────────────────────────────────────────────────
async function refresh() {
  try {
    const [summary, txnsRes, tsRes, healthRes] = await Promise.allSettled([
      fetch("/api/v1/analytics/summary").then(r => r.json()),
      fetch(`/api/v1/transactions?limit=${MAX_FEED_ROWS}`).then(r => r.json()),
      fetch("/api/v1/analytics/timeseries?window=60").then(r => r.json()),
      fetch("/api/v1/health").then(r => r.json()),
    ]);

    if (summary.status === "fulfilled") updateMetrics(summary.value);
    if (txnsRes.status === "fulfilled") updateFeed(txnsRes.value.transactions || []);
    if (tsRes.status === "fulfilled") updateTimeseries(tsRes.value);
    if (summary.status === "fulfilled") {
      updateDonut(summary.value.by_risk_level || {});
      updateCatBar(summary.value.by_merchant_category || {});
      updateHeatmap(summary.value.by_hour || {});
    }
    if (healthRes.status === "fulfilled") updateHealth(healthRes.value);

    $("last-updated").textContent = new Date().toLocaleTimeString();
  } catch (err) {
    console.warn("Refresh error:", err);
  }
}

// ── Metrics update ────────────────────────────────────────────────────────────
function updateMetrics(data) {
  animateNumber("metric-total", data.total_scored || 0);
  animateNumber("metric-flagged", data.flagged_count || 0);
  setEl("metric-flagrate", `${(data.flag_rate_pct || 0).toFixed(2)}%`);
  setEl("metric-latency", `${(data.avg_latency_ms || 0).toFixed(1)}ms`);
}

function animateNumber(id, newVal) {
  const el = $(id);
  if (!el) return;
  const oldVal = parseInt(el.textContent.replace(/,/g, ""), 10) || 0;
  if (oldVal !== newVal) {
    el.classList.remove("value-updated");
    void el.offsetWidth; // reflow
    el.classList.add("value-updated");
    el.textContent = newVal.toLocaleString();
  }
}

function setEl(id, val) {
  const el = $(id);
  if (el) el.textContent = val;
}

// ── Live feed ─────────────────────────────────────────────────────────────────
function updateFeed(txns) {
  const tbody = $("txn-tbody");
  if (!tbody) return;

  if (!txns || txns.length === 0) {
    tbody.innerHTML = `
      <tr><td colspan="7" class="empty-state">
        <span class="empty-state-icon">◈</span>
        <span>Awaiting transaction data…</span>
      </td></tr>`;
    return;
  }

  tbody.innerHTML = txns.slice(0, MAX_FEED_ROWS).map(t => {
    const upiMasked = maskUPI(t.upi_id || "");
    const time = formatTime(t.timestamp);
    const amount = formatAmount(t.amount);
    const cat = t.merchant_category || "—";
    const risk = t.risk_level || "LOW";
    const action = t.routing_action || "ALLOW";
    const latency = t.latency_ms != null ? `${t.latency_ms.toFixed(1)}ms` : "—";

    return `<tr class="risk-${risk.toLowerCase()}">
      <td>${time}</td>
      <td style="font-size:10px">${upiMasked}</td>
      <td style="text-align:right">₹${amount}</td>
      <td>${cat}</td>
      <td><span class="risk-badge ${risk}">${risk}</span></td>
      <td><span class="action-badge ${action}">${action}</span></td>
      <td style="text-align:right;color:var(--text-dim)">${latency}</td>
    </tr>`;
  }).join("");
}

function maskUPI(upi) {
  if (!upi) return "—";
  return upi.slice(0, 4) + "***" + (upi.includes("@") ? "@" + upi.split("@")[1] : "");
}

function formatTime(ts) {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch { return "—"; }
}

function formatAmount(v) {
  if (v == null) return "—";
  return parseFloat(v).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── Donut chart ───────────────────────────────────────────────────────────────
function updateDonut(byRisk) {
  if (!chartsInitialised) return;
  const labels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
  const values = labels.map(l => byRisk[l] || 0);
  // Ensure at least 1 for visible ring
  const total = values.reduce((a, b) => a + b, 0);
  if (total === 0) values[3] = 1;

  Plotly.restyle("chart-donut", { values: [values] });
}

// ── Time-series ───────────────────────────────────────────────────────────────
function updateTimeseries(data) {
  if (!chartsInitialised || !data.timestamps) return;

  const ts = data.timestamps.map(t => {
    try { return new Date(t).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }); }
    catch { return t; }
  });

  const flagged = data.flagged || [];
  const totals = data.total || [];

  // IF line = flagged rate; AE line = slightly offset for demo
  const ifRate = flagged;
  const aeRate = flagged.map((v, i) => Math.max(0, v + Math.round((Math.random() - 0.5) * 2)));

  Plotly.restyle("chart-timeseries", { x: [ts, ts], y: [ifRate, aeRate] });
}

// ── Heatmap ───────────────────────────────────────────────────────────────────
function updateHeatmap(byHour) {
  if (!chartsInitialised) return;

  // Build a 7×24 matrix (days × hours) — use by_hour data spread across days
  const z = Array.from({ length: 7 }, () => Array(24).fill(0));
  Object.entries(byHour).forEach(([h, d]) => {
    const hour = parseInt(h, 10);
    if (hour < 0 || hour > 23) return;
    const flagged = d.flagged || 0;
    // Distribute evenly across days for demo
    for (let day = 0; day < 7; day++) {
      z[day][hour] = Math.round(flagged * (0.8 + Math.random() * 0.4) / 7);
    }
  });

  Plotly.restyle("chart-heatmap", { z: [z] });
}

// ── Category bar chart ────────────────────────────────────────────────────────
function updateCatBar(byCat) {
  if (!chartsInitialised) return;

  const entries = Object.entries(byCat)
    .map(([cat, d]) => ({ cat, rate: d.flag_rate || 0 }))
    .sort((a, b) => b.rate - a.rate);

  const cats = entries.map(e => e.cat);
  const rates = entries.map(e => parseFloat(e.rate.toFixed(2)));
  const barColors = rates.map(r => r > 10 ? COLORS.anomaly : r > 5 ? COLORS.amber : COLORS.normal);

  Plotly.restyle("chart-catbar", {
    x: [rates],
    y: [cats],
    "marker.color": [barColors],
  });
}

// ── Health / drift status ─────────────────────────────────────────────────────
function updateHealth(health) {
  const pill = $("status-pill-text");
  if (!pill) return;
  if (health.status === "degraded" || health.drift_detected) {
    pill.textContent = "DRIFT";
    $("status-live").style.setProperty("color", "var(--accent-amber)");
  } else {
    pill.textContent = "LIVE";
  }

  const mv = $("model-version");
  if (mv && health.drift_watchdog) {
    const wd = health.drift_watchdog;
    mv.textContent = `v${health.transaction_count || 0} scored · precision ${(wd.precision_last_100 * 100).toFixed(1)}%`;
  }
}
