/**
 * BB+ Cash Flow AI — app.js
 * Handles: file upload, API calls, Chart.js rendering, UI state
 */

const API_BASE = "http://127.0.0.1:5000";

// ─────────────────────────────────────────────
//  State
// ─────────────────────────────────────────────
let selectedFile = null;
let forecastChart = null;

// ─────────────────────────────────────────────
//  DOM refs
// ─────────────────────────────────────────────
const fileInput      = document.getElementById("file-input");
const uploadZone     = document.getElementById("upload-zone");
const browseBtn      = document.getElementById("browse-btn");
const demoBtn        = document.getElementById("demo-btn");
const runBtn         = document.getElementById("run-btn");
const sampleLink     = document.getElementById("sample-link");
const loadingEl      = document.getElementById("loading");
const dashboard      = document.getElementById("dashboard");
const uploadSection  = document.getElementById("upload-section");
const resetBtn       = document.getElementById("reset-btn");
const openingInput   = document.getElementById("opening-balance");
const horizonSelect  = document.getElementById("horizon-select");

// CRM DOM refs
const tabCrm          = document.getElementById("tab-crm");
const tabUpload       = document.getElementById("tab-upload");
const crmCard         = document.getElementById("crm-card");
const uploadCard      = document.getElementById("upload-card");
const crmRunBtn       = document.getElementById("crm-run-btn");
const crmHorizon      = document.getElementById("crm-horizon");
const toggleDrawerBtn = document.getElementById("toggle-drawer-btn");
const crmDrawer       = document.getElementById("crm-drawer");
const crmRefreshBtn   = document.getElementById("crm-refresh-btn");
const crmInvoicesCards= document.getElementById("crm-invoices-cards");
const crmPayablesCards= document.getElementById("crm-payables-cards");
const crmDashPanel    = document.getElementById("crm-dash-panel");
const crmDashBody     = document.getElementById("crm-dash-body");

let currentSource = "crm"; // 'crm' or 'upload'

// Source Switcher
if (tabCrm && tabUpload) {
  tabCrm.addEventListener("click", () => switchSource("crm"));
  tabUpload.addEventListener("click", () => switchSource("upload"));
}

function switchSource(source) {
  currentSource = source;
  if (source === "crm") {
    tabCrm.classList.add("active");
    tabUpload.classList.remove("active");
    if (crmCard) crmCard.hidden = false;
    if (uploadCard) uploadCard.hidden = true;
  } else {
    tabUpload.classList.add("active");
    tabCrm.classList.remove("active");
    if (uploadCard) uploadCard.hidden = false;
    if (crmCard) crmCard.hidden = true;
  }
}

// Drawer Toggle
if (toggleDrawerBtn && crmDrawer) {
  toggleDrawerBtn.addEventListener("click", () => {
    crmDrawer.hidden = !crmDrawer.hidden;
    const isHidden = crmDrawer.hidden;
    toggleDrawerBtn.querySelector("span").textContent = isHidden ? "📋 View Synced Ledger" : "▲ Hide Synced Ledger";
  });
}

// Sync Now button
if (crmRefreshBtn) {
  crmRefreshBtn.addEventListener("click", async () => {
    const span = crmRefreshBtn.querySelector("span");
    span.textContent = "Syncing…";
    await loadCrmPipeline();
    span.textContent = "Synced";
    setTimeout(() => { span.textContent = "Sync Now"; }, 2000);
  });
}

// CRM Forecast Run button
if (crmRunBtn) {
  crmRunBtn.addEventListener("click", runCrmForecast);
}

// ─────────────────────────────────────────────
//  File selection
// ─────────────────────────────────────────────
browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

uploadZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFileChosen(fileInput.files[0]);
});

// Drag & drop
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragging");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragging"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragging");
  if (e.dataTransfer.files.length) handleFileChosen(e.dataTransfer.files[0]);
});

function handleFileChosen(file) {
  selectedFile = file;

  // Show file name indicator
  const existing = uploadZone.querySelector(".file-chosen");
  if (existing) existing.remove();

  const chip = document.createElement("div");
  chip.className = "file-chosen";
  chip.innerHTML = `✅ <span>${file.name}</span>`;
  uploadZone.appendChild(chip);

  runBtn.disabled = false;
}

// ─────────────────────────────────────────────
//  Demo data direct load & download
// ─────────────────────────────────────────────
if (demoBtn) {
  demoBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    const originalText = demoBtn.textContent;
    demoBtn.textContent = "⏳ Loading…";
    try {
      const res = await fetch(`${API_BASE}/api/sample-data`);
      const blob = await res.blob();
      const file = new File([blob], "demo_statement_180d.csv", { type: "text/csv" });
      handleFileChosen(file);
      demoBtn.textContent = "✅ Loaded";
      setTimeout(() => { demoBtn.textContent = originalText; }, 2500);
    } catch {
      demoBtn.textContent = originalText;
      showError("Could not load demo statement. Ensure Flask server is running.");
    }
  });
}

sampleLink.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  try {
    const res = await fetch(`${API_BASE}/api/sample-data`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "sample_transactions.csv";
    a.click(); URL.revokeObjectURL(url);
  } catch {
    alert("Make sure the Flask server is running on port 5000.");
  }
});

// ─────────────────────────────────────────────
//  Run Forecast
// ─────────────────────────────────────────────
runBtn.addEventListener("click", runForecast);

async function runForecast() {
  if (!selectedFile) return;

  // Show loading
  showLoading();

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("opening_balance", openingInput.value || 0);
    formData.append("horizon", horizonSelect.value);

    const res = await fetch(`${API_BASE}/api/forecast`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || "Unknown server error");
    }

    renderDashboard(data);

  } catch (err) {
    hideLoading();
    showError(err.message);
  }
}

// ─────────────────────────────────────────────
//  Loading steps animation
// ─────────────────────────────────────────────
let stepTimer = null;

function showLoading() {
  uploadSection.hidden = true;
  loadingEl.hidden = false;
  dashboard.hidden = true;

  // Animate steps
  const steps = ["step-1", "step-2", "step-3"];
  let i = 0;
  steps.forEach(id => {
    document.getElementById(id).className = "step";
  });
  document.getElementById(steps[0]).className = "step active";

  stepTimer = setInterval(() => {
    if (i < steps.length - 1) {
      document.getElementById(steps[i]).className = "step done";
      i++;
      document.getElementById(steps[i]).className = "step active";
    }
  }, 900);
}

function hideLoading() {
  clearInterval(stepTimer);
  loadingEl.hidden = true;
}

// ─────────────────────────────────────────────
//  Render Dashboard
// ─────────────────────────────────────────────
function renderDashboard(data) {
  hideLoading();
  dashboard.hidden = false;

  const { summary, forecast_series, historical_series, explanation, model_used } = data;
  const horizonDays = forecast_series.length;

  renderKPIs(summary);
  renderChart(summary, forecast_series, horizonDays);
  renderAIPanel(explanation, summary);
  renderActionItems(explanation);
  renderCriticalPeriod(summary);

  document.getElementById("model-badge").textContent =
    `⚡ Model: ${model_used || "Ensemble ML"}`;

  const outlierEl = document.getElementById("outlier-badge");
  if (outlierEl) {
    if (data.outlier_info && data.outlier_info.count > 0) {
      outlierEl.textContent = `🛡️ ${data.outlier_info.count} outlier(s) capped`;
      outlierEl.title = `Largest capped anomaly: ₹${Math.round(data.outlier_info.max_magnitude).toLocaleString('en-IN')}`;
      outlierEl.hidden = false;
    } else {
      outlierEl.hidden = true;
    }
  }

  document.getElementById("data-points-label").textContent =
    `${summary.data_points} transactions analysed`;
  document.getElementById("chart-subtitle").textContent =
    `Next ${horizonDays} days`;
  document.getElementById("data-range-label").textContent =
    `Data: ${fmt(summary.date_range.start)} → ${fmt(summary.date_range.end)}`;

  // Synced CRM ledger breakdown on dashboard
  if (crmDashPanel) {
    if (data.is_crm_source && data.receivables && data.payables) {
      crmDashPanel.hidden = false;
      renderDashboardLedger(data);
    } else {
      crmDashPanel.hidden = true;
    }
  }
}

// ─────────────────────────────────────────────
//  KPI Cards
// ─────────────────────────────────────────────
function renderKPIs(s) {
  const curr = s.current_balance;
  const proj = s.projected_balance;
  const delta = proj - curr;
  const pct   = curr !== 0 ? ((delta / Math.abs(curr)) * 100).toFixed(1) : 0;

  animateCount("val-current", curr);
  animateCount("val-in", s.expected_inflow);
  animateCount("val-out", s.expected_outflow);
  animateCount("val-projected", proj);

  const projEl = document.getElementById("val-projected");
  projEl.className = "kpi-value " + (proj >= curr ? "green" : "red");

  document.getElementById("val-delta").textContent =
    `${delta >= 0 ? "▲" : "▼"} ${fmtINR(Math.abs(delta))} (${Math.abs(pct)}%)`;
  document.getElementById("val-delta").style.color =
    delta >= 0 ? "var(--green-l)" : "var(--red-l)";
}

function animateCount(id, target) {
  const el = document.getElementById(id);
  const duration = 1200;
  const start = performance.now();

  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = fmtINR(Math.round(eased * target));
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ─────────────────────────────────────────────
//  Chart.js — Forecast Chart
// ─────────────────────────────────────────────
function renderChart(summary, forecast_series, horizon) {
  const ctx = document.getElementById("forecast-chart").getContext("2d");

  if (forecastChart) forecastChart.destroy();

  const labels   = forecast_series.map(d => d.date);
  const values   = forecast_series.map(d => d.predicted);
  const uppers   = forecast_series.map(d => d.upper);
  const lowers   = forecast_series.map(d => d.lower);

  // Build running balance from current balance
  const currentBalance = summary.current_balance;
  let running = currentBalance;
  const balances = forecast_series.map(d => {
    running += d.predicted;
    return Math.round(running);
  });

  let runU = currentBalance;
  const balancesUpper = forecast_series.map(d => { runU += d.upper; return Math.round(runU); });
  let runL = currentBalance;
  const balancesLower = forecast_series.map(d => { runL += d.lower; return Math.round(runL); });

  // Gradient fill
  const gradientLine = ctx.createLinearGradient(0, 0, 0, 300);
  gradientLine.addColorStop(0, "rgba(59,130,246,0.5)");
  gradientLine.addColorStop(1, "rgba(59,130,246,0)");

  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Upper Bound",
          data: balancesUpper,
          borderColor: "transparent",
          backgroundColor: "rgba(59,130,246,0.08)",
          fill: "+1",
          tension: 0.4,
          pointRadius: 0,
        },
        {
          label: "Projected Balance",
          data: balances,
          borderColor: "#3b82f6",
          backgroundColor: gradientLine,
          fill: false,
          tension: 0.4,
          pointRadius: 3,
          pointBackgroundColor: "#3b82f6",
          pointHoverRadius: 6,
          borderWidth: 2.5,
        },
        {
          label: "Lower Bound",
          data: balancesLower,
          borderColor: "transparent",
          backgroundColor: "rgba(59,130,246,0.08)",
          fill: "-1",
          tension: 0.4,
          pointRadius: 0,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e293b",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          titleColor: "#f1f5f9",
          bodyColor: "#94a3b8",
          padding: 12,
          callbacks: {
            label: (ctx) => {
              if (ctx.dataset.label === "Projected Balance")
                return ` Balance: ${fmtINR(ctx.raw)}`;
              if (ctx.dataset.label === "Upper Bound")
                return ` Best case: ${fmtINR(ctx.raw)}`;
              if (ctx.dataset.label === "Lower Bound")
                return ` Worst case: ${fmtINR(ctx.raw)}`;
            },
            afterBody: (items) => {
              const idx = items[0].dataIndex;
              const evts = forecast_series[idx] && forecast_series[idx].crm_events;
              if (evts && evts.length > 0) {
                return "\nCRM Events:\n" + evts.map(e => " • " + e).join("\n");
              }
              return "";
            }
          }
        },
        annotation: {
          annotations: {
            zeroline: {
              type: "line",
              yMin: 0, yMax: 0,
              borderColor: "rgba(239,68,68,0.4)",
              borderWidth: 1,
              borderDash: [4, 4],
              label: {
                content: "Zero",
                display: true,
                color: "rgba(239,68,68,0.6)",
                font: { size: 10 },
                position: "start"
              }
            },
            worstDay: {
              type: "line",
              xMin: summary.worst_date,
              xMax: summary.worst_date,
              borderColor: "rgba(245,158,11,0.5)",
              borderWidth: 1,
              borderDash: [4, 4],
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: {
            color: "#64748b", font: { size: 11 },
            maxTicksLimit: 8,
            callback: (val, i) => {
              const d = new Date(labels[i]);
              return d.toLocaleDateString("en-IN", { day:"numeric", month:"short" });
            }
          },
          border: { color: "rgba(255,255,255,0.06)" }
        },
        y: {
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: {
            color: "#64748b", font: { size: 11 },
            callback: (v) => fmtINRShort(v)
          },
          border: { color: "rgba(255,255,255,0.06)" }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────
//  AI Panel
// ─────────────────────────────────────────────
function renderAIPanel(explanation, summary) {
  const container = document.getElementById("ai-body");
  container.innerHTML = "";

  const icons = ["📊", "📅", "⚖️", "💡", "📈"];

  explanation.insights.forEach((text, i) => {
    const block = document.createElement("div");
    block.className = "insight-block";
    block.style.animationDelay = `${i * 0.08}s`;
    block.innerHTML = `
      <div class="insight-icon">${icons[i % icons.length]}</div>
      <p class="insight-text">${text}</p>
    `;
    container.appendChild(block);
  });

  if (explanation.alerts.length > 0) {
    const hr = document.createElement("hr");
    hr.className = "divider";
    container.appendChild(hr);

    explanation.alerts.forEach(alert => {
      const banner = document.createElement("div");
      banner.className = `alert-banner alert-${alert.type}`;
      banner.textContent = alert.text;
      container.appendChild(banner);
    });
  }

  if (explanation.insights.length === 0) {
    container.innerHTML = `<p style="color:var(--text-3);font-size:0.875rem">No insights generated. Try adding more historical data.</p>`;
  }
}

// ─────────────────────────────────────────────
//  Action Items
// ─────────────────────────────────────────────
function renderActionItems(explanation) {
  const container = document.getElementById("action-items");
  container.innerHTML = "";

  if (explanation.recommendations.length === 0) {
    container.innerHTML = `
      <div class="alert-banner alert-success">
        ✅ Cash flow looks stable — no urgent actions required.
      </div>`;
    return;
  }

  explanation.recommendations.forEach((rec, i) => {
    const item = document.createElement("div");
    item.className = "action-item";
    item.innerHTML = `
      <div class="action-num">${i + 1}</div>
      <p class="action-text">${rec}</p>
    `;
    container.appendChild(item);
  });
}

// ─────────────────────────────────────────────
//  Critical Period
// ─────────────────────────────────────────────
function renderCriticalPeriod(summary) {
  const container = document.getElementById("critical-period");
  const isCritical = summary.worst_balance < summary.current_balance * 0.4
                  || summary.worst_balance < 100000;

  if (isCritical) {
    const d = new Date(summary.worst_date);
    const dateStr = d.toLocaleDateString("en-IN", { day:"numeric", month:"long", year:"numeric" });
    container.innerHTML = `
      <div class="critical-date-box">
        <span class="critical-label">Lowest projected balance on</span>
        <span class="critical-date">📅 ${dateStr}</span>
        <span class="critical-balance">${fmtINR(summary.worst_balance)}</span>
      </div>
      <p class="critical-hint">
        Cash is expected to be tightest around this date — likely due to outflow clustering
        before major receivables arrive. Plan ahead to ensure adequate liquidity.
      </p>
    `;
  } else {
    container.innerHTML = `
      <div class="good-box">
        <div class="good-icon">💚</div>
        <p class="good-text">No critical cash crunch detected</p>
        <p class="good-sub">Lowest projected balance: ${fmtINR(summary.worst_balance)}<br/>on ${fmt(summary.worst_date)}</p>
      </div>
    `;
  }
}

// ─────────────────────────────────────────────
//  CRM Forecast Runner & Data Pipeline
// ─────────────────────────────────────────────
async function runCrmForecast() {
  showLoading();
  try {
    const horizon = crmHorizon ? crmHorizon.value : 30;
    const res = await fetch(`${API_BASE}/api/crm/forecast`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ horizon: parseInt(horizon) })
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Server error while generating CRM forecast");
    }
    renderDashboard(data);
  } catch (err) {
    hideLoading();
    showError(err.message);
  }
}

async function loadCrmPipeline() {
  try {
    const res = await fetch(`${API_BASE}/api/crm/pipeline`);
    const data = await res.json();
    if (data.success) {
      renderCrmDrawerCards(data);
    }
  } catch (err) {
    console.warn("Could not load CRM pipeline:", err);
  }
}

function renderCrmDrawerCards(data) {
  if (!crmInvoicesCards || !crmPayablesCards) return;
  crmInvoicesCards.innerHTML = "";
  crmPayablesCards.innerHTML = "";

  (data.receivables || []).forEach(inv => {
    const el = document.createElement("div");
    el.className = "ledger-item";
    el.innerHTML = `
      <div class="ledger-item-left">
        <span class="ledger-item-title">${inv.client}</span>
        <span class="ledger-item-sub">#${inv.id} · ${inv.category}</span>
      </div>
      <div class="ledger-item-right">
        <span class="ledger-item-amt text-green">+${fmtINR(inv.amount)}</span>
        <span class="ledger-item-due">Due: ${fmt(inv.due_date)}</span>
      </div>
    `;
    crmInvoicesCards.appendChild(el);
  });

  const combinedPayables = [
    ...(data.payables || []).map(p => ({ title: p.vendor, sub: `#${p.id} · ${p.priority}`, amt: p.amount, due: p.due_date })),
    ...(data.commitments || []).map(c => ({ title: c.name, sub: `${c.id} · ${c.category}`, amt: c.amount, due: c.due_date }))
  ];

  combinedPayables.forEach(bill => {
    const el = document.createElement("div");
    el.className = "ledger-item";
    el.innerHTML = `
      <div class="ledger-item-left">
        <span class="ledger-item-title">${bill.title}</span>
        <span class="ledger-item-sub">${bill.sub}</span>
      </div>
      <div class="ledger-item-right">
        <span class="ledger-item-amt text-red">−${fmtINR(bill.amt)}</span>
        <span class="ledger-item-due">Due: ${fmt(bill.due_date)}</span>
      </div>
    `;
    crmPayablesCards.appendChild(el);
  });
}

function renderDashboardLedger(data) {
  if (!crmDashBody) return;
  crmDashBody.innerHTML = `
    <div class="drawer-grid">
      <div class="drawer-col">
        <h5 class="col-title text-green">📥 Customer Invoices (Expected Collections: +₹8.4L)</h5>
        <div class="ledger-cards">
          ${(data.receivables || []).map(r => `
            <div class="ledger-item">
              <div class="ledger-item-left">
                <span class="ledger-item-title">${r.client}</span>
                <span class="ledger-item-sub">Invoice #${r.id} · ${r.contact || r.category}</span>
              </div>
              <div class="ledger-item-right">
                <span class="ledger-item-amt text-green">+${fmtINR(r.amount)}</span>
                <span class="ledger-item-due">Due: ${fmt(r.due_date)} (${r.status})</span>
              </div>
            </div>
          `).join("")}
        </div>
      </div>
      <div class="drawer-col">
        <h5 class="col-title text-red">📤 Supplier Bills & Operational Commitments (−₹12.3L)</h5>
        <div class="ledger-cards">
          ${(data.payables || []).map(p => `
            <div class="ledger-item">
              <div class="ledger-item-left">
                <span class="ledger-item-title">${p.vendor}</span>
                <span class="ledger-item-sub">Bill #${p.id} · ${p.priority} priority</span>
              </div>
              <div class="ledger-item-right">
                <span class="ledger-item-amt text-red">−${fmtINR(p.amount)}</span>
                <span class="ledger-item-due">Due: ${fmt(p.due_date)}</span>
              </div>
            </div>
          `).join("")}
          ${(data.commitments || []).map(c => `
            <div class="ledger-item">
              <div class="ledger-item-left">
                <span class="ledger-item-title">${c.name}</span>
                <span class="ledger-item-sub">${c.type} · ${c.category}</span>
              </div>
              <div class="ledger-item-right">
                <span class="ledger-item-amt text-amber">−${fmtINR(c.amount)}</span>
                <span class="ledger-item-due">Due: ${fmt(c.due_date)}</span>
              </div>
            </div>
          `).join("")}
        </div>
      </div>
    </div>
  `;
}

// ─────────────────────────────────────────────
//  Reset
// ─────────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  dashboard.hidden = true;
  uploadSection.hidden = false;
  selectedFile = null;
  fileInput.value = "";
  runBtn.disabled = true;

  const chip = uploadZone.querySelector(".file-chosen");
  if (chip) chip.remove();

  if (forecastChart) { forecastChart.destroy(); forecastChart = null; }
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// Load initial CRM pipeline on page startup
loadCrmPipeline();

// ─────────────────────────────────────────────
//  Utilities
// ─────────────────────────────────────────────
function fmtINR(val) {
  if (val === null || val === undefined) return "—";
  const abs = Math.abs(val);
  const sign = val < 0 ? "−" : "";
  if (abs >= 1e7) return `${sign}₹${(abs / 1e7).toFixed(2)}Cr`;
  if (abs >= 1e5) return `${sign}₹${(abs / 1e5).toFixed(2)}L`;
  if (abs >= 1e3) return `${sign}₹${(abs / 1e3).toFixed(1)}K`;
  return `${sign}₹${abs.toFixed(0)}`;
}

function fmtINRShort(v) {
  if (Math.abs(v) >= 1e5) return `₹${(v / 1e5).toFixed(1)}L`;
  if (Math.abs(v) >= 1e3) return `₹${(v / 1e3).toFixed(0)}K`;
  return `₹${v.toFixed(0)}`;
}

function fmt(dateStr) {
  return new Date(dateStr).toLocaleDateString("en-IN", { day:"numeric", month:"short", year:"numeric" });
}

function showError(msg) {
  uploadSection.hidden = false;
  const err = document.createElement("div");
  err.className = "alert-banner alert-danger";
  err.style.cssText = "position:fixed;top:80px;right:2rem;z-index:300;max-width:420px";
  err.innerHTML = `⚠️ <strong>Error:</strong> ${msg}`;
  document.body.appendChild(err);
  setTimeout(() => err.remove(), 6000);
}
