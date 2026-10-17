# 📈 BB+ AI Cash Flow Forecast

> **Intelligent 30-Day Forward Cash Forecasting for Indian Small & Medium Businesses**  
> Powered by Facebook Prophet, XGBoost, and Deterministic CRM / ERP Forward Ledgers.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20Prophet%20%7C%20XGBoost-orange.svg)](https://flask.palletsprojects.com)
[![Status](https://img.shields.io/badge/Status-Active%20Prototype-success.svg)]()

---

## 🌟 Why BB+ Cash Flow AI?

Business owners often know their bank balance *today*, but not what will happen **30 days from now**. Traditional accounting reports (P&L, Balance Sheet) look in the rearview mirror. 

**BB+ provides forward clarity:**
- **Today**: Cash available: **₹6.2 Lakh**
- **Expected Next 30 Days**:
  - Collections (Verified CRM Invoices): **+₹8.4 Lakh**
  - Scheduled Payments (Supplier POs): **−₹10.1 Lakh**
  - Projected Operational Expenses: **−₹2.2 Lakh**
- **Projected Ending Balance**: **₹4.9 Lakh**
- **AI Explanation**: 
  > *"⚠️ Critical Liquidity Dip: Your cash position could fall significantly around **November 06** (dipping to ₹0.9 Lakh) because several major supplier payments (including ₹4.2L to Delta Cloud and ₹3.8L to Prime Logistics) are scheduled before key customer receivables arrive."*
- **Actionable Strategic Recommendation**:
  > *"Prioritize Collection: Follow up immediately on ₹3.1 Lakh of receivables (Invoice #INV-2026-081 — Apex Retail Solutions) due on October 29 to prevent balance dipping below safe operating buffers."*

---

## 🛡️ Solved Machine Learning Pitfalls

| Pitfall | Problem | Solution in BB+ |
|---|---|---|
| **1. Additive Bias with Outliers** | Sudden massive client payouts or bulk capital expenses distort Prophet's trendline. | Statistical **Z-score Winsorization** ($Z \ge 2.5$) caps extreme anomalies before model fitting; reduced `changepoint_prior_scale=0.05` prevents trend distortion. |
| **2. Lack of Exogenous Feature Depth** | Prophet assumes linear exogenous regressors and misses complex interactions (e.g. month-end payroll spikes on Fridays). | **XGBoost Regressor** trained on 20+ engineered features: lags (1, 2, 3, 7, 14, 21, 28), 7d/14d rolling stats, calendar features, and interaction terms (`dow_x_wom`). |
| **3. Aggregated Gaps & Sparse Data** | Zero-flow days cause models to hallucinate cyclical patterns. | Date reindexing with an explicit binary `has_transaction` flag fed to both models; automatic `gap_ratio` calculation disables weekly seasonality fitting when data is sparse ($> 40\%$ gaps). |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             BROWSER UI (Dark Glassmorphism)                 │
│   • Source Switcher: [ 🔗 Connected CRM / ERP ]  or  [ 📁 Upload Statement ]│
│   • Interactive Chart.js with 80% Confidence Interval Band & Event Tooltips │
│   • KPI Strip: Cash Today · Expected In · Expected Out · Projected Balance  │
│   • Action Items Panel & Critical Cash Crunch Warning                       │
│   • Expandable Synced CRM Forward Ledger                                    │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ HTTP REST API
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FLASK FORECASTING BACKEND                          │
│                                                                             │
│  ┌───────────────────────────────┐     ┌─────────────────────────────────┐  │
│  │    HISTORICAL TIME SERIES     │     │      FORWARD CRM LEDGER         │  │
│  │   Bank Statement / History    │     │   Verified Invoices & Bills     │  │
│  └──────────────┬────────────────┘     └────────────────┬────────────────┘  │
│                 │                                       │                   │
│                 ▼                                       ▼                   │
│  ┌───────────────────────────────┐     ┌─────────────────────────────────┐  │
│  │ 1. Winsorize Outliers         │     │ • Customer Receivables (+₹8.4L) │  │
│  │ 2. Prophet (Trend/Seasonality)│     │ • Supplier Payables (−₹10.1L)   │  │
│  │ 3. XGBoost (Non-linear lags)  │     │ • Fixed Commitments (−₹2.2L)    │  │
│  │ 4. Dynamic Weighted Ensemble  │     └────────────────┬────────────────┘  │
│  └──────────────┬────────────────┘                      │                   │
│                 │                                       │                   │
│                 └───────────────────┬───────────────────┘                   │
│                                     ▼                                       │
│                [ Hybrid Forward Cash Trajectory Engine ]                    │
│                                     │                                       │
│                                     ▼                                       │
│                [ Plain-English CFO Explanation & Action Items ]             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/HaarisBahjat/ai-cash-flow-forecast.git
cd ai-cash-flow-forecast
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```

### 4. Open in browser
Open [http://localhost:5000](http://localhost:5000) in your browser.
- Click **"⚡ Run AI Forecast from CRM"** for instant 1-click execution.
- Or switch to **"Upload Statement"** and drag-and-drop your bank statement CSV/Excel file.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/crm/status` | `GET` | Returns CRM connection health, account info, and pipeline metrics. |
| `/api/crm/pipeline` | `GET` | Returns itemized customer invoices, supplier payables, and commitments. |
| `/api/crm/forecast` | `POST` | Executes hybrid forecast (Prophet + XGBoost + CRM Forward Ledger). |
| `/api/forecast` | `POST` | Upload CSV/Excel statement file with opening balance and horizon. |
| `/api/sample-data` | `GET` | Generates a 180-day realistic sample transaction CSV for testing. |

---

## 📦 Project Structure

```
ai-cash-flow-forecast/
├── app.py               # Flask backend & Prophet + XGBoost ensemble engine
├── crm_synthetic.py     # Synthetic CRM live connector, forward ledger & NLG
├── index.html           # Dark glassmorphism dashboard UI
├── style.css            # Tailored design system, animations & badges
├── app.js               # Frontend controller, Chart.js logic & CRM sync
├── requirements.txt     # Python dependencies (flask, prophet, xgboost, etc.)
├── test_sample.csv      # Sample 180-day statement for testing
└── README.md            # Documentation & architecture guide
```

---

## 📄 License

MIT License. Developed for SME financial empowerment.
