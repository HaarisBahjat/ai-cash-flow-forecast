"""
Synthetic CRM Data & Integration Module for BB+ Cash Flow AI
Simulates a live connection to Zoho Books / HubSpot / Salesforce CRM.
Provides:
  1. Live CRM Connection Status & Metadata
  2. Synced Accounts Receivable (Customer Invoices)
  3. Synced Accounts Payable (Supplier Bills & POs)
  4. Scheduled Recurring Commitments (Payroll, Rent, GST)
  5. 180-day Historical Statement for Prophet + XGBoost model training
  6. Hybrid Forward Forecasting (Statistical Baseline + CRM Deterministic Ledger)
"""

from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
#  CRM CONFIGURATION & METADATA
# ─────────────────────────────────────────────────────────────

CRM_INFO = {
    "provider": "Zoho Books & CRM (Live Sandbox)",
    "company_name": "BharatTech Solutions Pvt Ltd",
    "gstin": "27AABCB1234M1Z5",
    "bank_name": "ICICI Bank — Current A/c (...4019)",
    "current_bank_balance": 620000.0,  # ₹6.2 Lakh
    "connected_at": "2026-10-17 09:30:00",
    "last_sync": "Just now (Automated 15m polling)",
    "status": "Healthy (Connected)",
}

def get_base_date():
    """Returns today's date normalized to midnight."""
    return datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

# ─────────────────────────────────────────────────────────────
#  SYNTHETIC CRM RECEIVABLES (Customer Invoices Pipeline)
# ─────────────────────────────────────────────────────────────

def generate_crm_receivables(base_date=None):
    """
    Simulates customer invoices synced from CRM/ERP.
    Total expected collections = ~₹8.4 Lakh.
    Includes the crucial ₹3.1L invoice from Apex Retail Solutions due around Day 12-14.
    """
    if base_date is None:
        base_date = get_base_date()

    return [
        {
            "id": "INV-2026-081",
            "client": "Apex Retail Solutions",
            "description": "Enterprise POS Software License (Q3)",
            "amount": 310000.0,
            "due_date": (base_date + timedelta(days=12)).strftime("%Y-%m-%d"),
            "status": "Due Soon",
            "probability": 0.90,
            "category": "Software Licensing",
            "contact": "Karthik Raja (VP Finance)",
            "days_away": 12
        },
        {
            "id": "INV-2026-084",
            "client": "TechNova Corp",
            "description": "Cloud Migration Milestone 2",
            "amount": 240000.0,
            "due_date": (base_date + timedelta(days=22)).strftime("%Y-%m-%d"),
            "status": "Pending",
            "probability": 0.85,
            "category": "Cloud Services",
            "contact": "Ananya Sharma (CTO)",
            "days_away": 22
        },
        {
            "id": "INV-2026-088",
            "client": "Zenith Logistics Ltd",
            "description": "Fleet Tracking API Integration",
            "amount": 165000.0,
            "due_date": (base_date + timedelta(days=6)).strftime("%Y-%m-%d"),
            "status": "Confirmed",
            "probability": 0.95,
            "category": "API Consulting",
            "contact": "Rohan Mehra (Director)",
            "days_away": 6
        },
        {
            "id": "INV-2026-092",
            "client": "Kavita Global Exports",
            "description": "Custom CRM Workflow Customization",
            "amount": 125000.0,
            "due_date": (base_date + timedelta(days=27)).strftime("%Y-%m-%d"),
            "status": "Awaiting Signoff",
            "probability": 0.75,
            "category": "Development",
            "contact": "Pooja Verma (Ops Head)",
            "days_away": 27
        },
    ]

# ─────────────────────────────────────────────────────────────
#  SYNTHETIC CRM PAYABLES (Supplier Bills & POs)
# ─────────────────────────────────────────────────────────────

def generate_crm_payables(base_date=None):
    """
    Simulates supplier bills and vendor purchase orders synced from CRM/ERP.
    Total expected payments = ~₹10.1 Lakh.
    Crucially: several large supplier bills hit around Day 8 to 18,
    BEFORE the major receivables arrive, causing a cash crunch around Day 14-20!
    """
    if base_date is None:
        base_date = get_base_date()

    return [
        {
            "id": "BILL-901",
            "vendor": "Delta Cloud & Server Infrastructure",
            "description": "AWS & Dedicated Hosting Monthly Invoice",
            "amount": 420000.0,
            "due_date": (base_date + timedelta(days=8)).strftime("%Y-%m-%d"),
            "priority": "High (Service Impact)",
            "category": "Cloud & Infra",
            "status": "Unpaid",
            "days_away": 8
        },
        {
            "id": "BILL-904",
            "vendor": "Prime Logistics & Hardware Distribution",
            "description": "Client IoT Hardware Inventory Dispatch",
            "amount": 380000.0,
            "due_date": (base_date + timedelta(days=14)).strftime("%Y-%m-%d"),
            "priority": "Critical",
            "category": "Hardware & Stock",
            "status": "Pending Approval",
            "days_away": 14
        },
        {
            "id": "BILL-907",
            "vendor": "Apex Raw Materials & Components",
            "description": "Packaging & Circuit Assemblies Batch 4",
            "amount": 210000.0,
            "due_date": (base_date + timedelta(days=20)).strftime("%Y-%m-%d"),
            "priority": "Medium",
            "category": "Raw Materials",
            "status": "Scheduled",
            "days_away": 20
        },
    ]

# ─────────────────────────────────────────────────────────────
#  SYNTHETIC RECURRING COMMITMENTS (Payroll, Rent, Tax)
# ─────────────────────────────────────────────────────────────

def generate_crm_commitments(base_date=None):
    """
    Recurring operational expenses:
    Total expected expenses = ~₹2.2 Lakh over the period.
    """
    if base_date is None:
        base_date = get_base_date()

    return [
        {
            "id": "REC-TAX-01",
            "name": "GST Payout (Form GSTR-3B)",
            "amount": 95000.0,
            "due_date": (base_date + timedelta(days=16)).strftime("%Y-%m-%d"),
            "category": "Government & Compliance",
            "type": "Mandatory",
            "days_away": 16
        },
        {
            "id": "REC-RENT-01",
            "name": "Commercial Office Lease & Utilities",
            "amount": 65000.0,
            "due_date": (base_date + timedelta(days=10)).strftime("%Y-%m-%d"),
            "category": "Facilities",
            "type": "Contractual",
            "days_away": 10
        },
        {
            "id": "REC-SAAS-01",
            "name": "Corporate Software Subscriptions & Tools",
            "amount": 60000.0,
            "due_date": (base_date + timedelta(days=24)).strftime("%Y-%m-%d"),
            "category": "Software Subscriptions",
            "type": "Operational",
            "days_away": 24
        },
    ]

# ─────────────────────────────────────────────────────────────
#  SYNTHETIC HISTORICAL DATA FOR MODEL TRAINING (180 DAYS)
# ─────────────────────────────────────────────────────────────

def generate_crm_historical_transactions(days=180, base_date=None):
    """
    Generates 180 days of realistic daily bank transaction history
    matching BharatTech Solutions' profile.
    This trains Prophet + XGBoost to learn day-of-week, bi-weekly and monthly cycles.
    """
    if base_date is None:
        base_date = get_base_date()

    r = random.Random(42)  # reproducible
    rows = []

    for i in range(days, 0, -1):
        d = base_date - timedelta(days=i)
        day_num = d.day

        # 1st of month: Major customer retainer inflows
        if day_num == 1:
            rows.append({
                "date": d.strftime("%d-%m-%Y"),
                "amount": round(r.gauss(260000, 15000), 2),
                "description": "Monthly Retainer Collections"
            })

        # 5th of month: Staff payroll & contractor settlements
        if day_num == 5:
            rows.append({
                "date": d.strftime("%d-%m-%Y"),
                "amount": -round(r.gauss(140000, 8000), 2),
                "description": "Salary & Contractor Payouts"
            })

        # 10th of month: Office rent & facility expenses
        if day_num == 10:
            rows.append({
                "date": d.strftime("%d-%m-%Y"),
                "amount": -65000.0,
                "description": "Office Lease & Maintenance"
            })

        # 15th & 25th: Supplier inventory / cloud disbursements
        if day_num in [15, 25]:
            rows.append({
                "date": d.strftime("%d-%m-%Y"),
                "amount": -round(r.gauss(85000, 6000), 2),
                "description": "Vendor & Hardware Settlement"
            })

        # Daily business sales (weekdays higher, weekends lower)
        is_weekend = d.weekday() >= 5
        base_sales = 8000 if is_weekend else 18000
        daily_amount = r.gauss(base_sales, 4000)
        if daily_amount > 2000:
            rows.append({
                "date": d.strftime("%d-%m-%Y"),
                "amount": round(daily_amount, 2),
                "description": "Daily Operational Receipts"
            })

        # Occasional operational petty cash / misc outflow
        if r.random() < 0.25:
            rows.append({
                "date": d.strftime("%d-%m-%Y"),
                "amount": -round(r.uniform(3000, 12000), 2),
                "description": "Petty Cash & Admin Expense"
            })

    df = pd.DataFrame(rows)
    return df

# ─────────────────────────────────────────────────────────────
#  CRM STATUS & PIPELINE OVERVIEW
# ─────────────────────────────────────────────────────────────

def get_crm_summary_stats(base_date=None):
    """Returns aggregated metrics from the connected CRM."""
    if base_date is None:
        base_date = get_base_date()

    receivables = generate_crm_receivables(base_date)
    payables = generate_crm_payables(base_date)
    commitments = generate_crm_commitments(base_date)

    total_receivables = sum(r["amount"] for r in receivables)
    expected_receivables_prob = sum(r["amount"] * r["probability"] for r in receivables)
    total_payables = sum(p["amount"] for p in payables)
    total_commitments = sum(c["amount"] for c in commitments)

    return {
        "crm_info": CRM_INFO,
        "receivables_count": len(receivables),
        "total_receivables": total_receivables,
        "expected_receivables_weighted": round(expected_receivables_prob, 2),
        "payables_count": len(payables),
        "total_payables": total_payables,
        "commitments_count": len(commitments),
        "total_commitments": total_commitments,
        "net_scheduled_flow": round(total_receivables - total_payables - total_commitments, 2),
        "current_balance": CRM_INFO["current_bank_balance"],
    }

# ─────────────────────────────────────────────────────────────
#  HYBRID FORECAST ENGINE (Statistical Ensemble + CRM Ledger)
# ─────────────────────────────────────────────────────────────

def build_crm_scheduled_calendar(horizon_days=30, base_date=None):
    """
    Creates a date-by-date schedule of verified CRM forward events:
      - Customer collections
      - Supplier payments
      - Mandatory commitments
    Returns a dict mapping 'YYYY-MM-DD' -> list of event objects.
    """
    if base_date is None:
        base_date = get_base_date()

    calendar = {}
    for i in range(horizon_days):
        d_str = (base_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        calendar[d_str] = {
            "inflows": [],
            "outflows": [],
            "net_scheduled": 0.0,
            "details": []
        }

    receivables = generate_crm_receivables(base_date)
    payables = generate_crm_payables(base_date)
    commitments = generate_crm_commitments(base_date)

    for item in receivables:
        due = item["due_date"]
        if due in calendar:
            calendar[due]["inflows"].append(item)
            calendar[due]["net_scheduled"] += item["amount"]
            calendar[due]["details"].append(f"Inflow: ₹{item['amount']/1e5:.1f}L ({item['client']})")

    for item in payables:
        due = item["due_date"]
        if due in calendar:
            calendar[due]["outflows"].append(item)
            calendar[due]["net_scheduled"] -= item["amount"]
            calendar[due]["details"].append(f"Payment: −₹{item['amount']/1e5:.1f}L ({item['vendor']})")

    for item in commitments:
        due = item["due_date"]
        if due in calendar:
            calendar[due]["outflows"].append(item)
            calendar[due]["net_scheduled"] -= item["amount"]
            calendar[due]["details"].append(f"Expense: −₹{item['amount']/1e5:.1f}L ({item['name']})")

    return calendar

def blend_crm_with_statistical_forecast(forecast_df, crm_calendar):
    """
    Merges statistical time-series prediction (Prophet + XGBoost baseline)
    with exact CRM verified forward commitments.

    Logic:
      - Prophet + XGBoost captures diffuse, recurring daily baseline sales & operational drift.
      - Scheduled CRM invoices and bills replace/augment large lumpy flow days with 100% precision.
    """
    augmented = forecast_df.copy()
    crm_net_col = []
    crm_events_col = []

    for _, row in augmented.iterrows():
        d_str = row["date"].strftime("%Y-%m-%d")
        day_crm = crm_calendar.get(d_str, {"net_scheduled": 0.0, "details": []})
        crm_net_col.append(day_crm["net_scheduled"])
        crm_events_col.append(day_crm["details"])

    augmented["crm_scheduled"] = crm_net_col
    augmented["crm_events"] = crm_events_col

    # Combined predicted flow:
    # Baseline statistical flow + Verified CRM scheduled commitments
    augmented["predicted_total"] = augmented["predicted"] + augmented["crm_scheduled"]
    augmented["lower_total"] = augmented["lower"] + augmented["crm_scheduled"]
    augmented["upper_total"] = augmented["upper"] + augmented["crm_scheduled"]

    return augmented


# ─────────────────────────────────────────────────────────────
#  CRM AI EXPLANATION ENGINE
# ─────────────────────────────────────────────────────────────

def generate_crm_ai_explanation(summary, crm_summary, receivables, payables, commitments, worst_info):
    """
    Generates intelligent, actionable insights merging CRM accounts ledger
    with ML predictive time-series patterns.
    Produces the exact executive advice described in the user's vision:
      - 'Your cash position could fall significantly around [Date] because several supplier payments...'
      - 'Consider following up on ₹3.1 lakh of receivables due before [Date]...'
    """
    insights = []
    recommendations = []
    alerts = []

    curr = summary["current_balance"]
    proj = summary["projected_balance"]
    worst_bal = worst_info["worst_balance"]
    worst_dt_str = pd.to_datetime(worst_info["worst_date"]).strftime("%B %d")

    tot_in = crm_summary["total_receivables"]
    tot_out = crm_summary["total_payables"]
    tot_exp = crm_summary["total_commitments"]

    # 1. Main Cash Trajectory Insight
    insights.append(
        f"Your cash position is projected to move from <strong>₹{curr/1e5:.1f} Lakh</strong> today to "
        f"<strong>₹{proj/1e5:.1f} Lakh</strong> over the next 30 days. "
        f"While net monthly flow is manageable, timing asymmetries create a significant mid-month dip."
    )

    # 2. Critical Cash Crunch Analysis (The exact user scenario!)
    insights.append(
        f"⚠️ <strong>Critical Liquidity Dip:</strong> Your cash position could fall significantly around "
        f"<strong>{worst_dt_str}</strong> (dipping to ₹{worst_bal/1e5:.1f} Lakh) because several major supplier payments "
        f"(including <strong>₹4.2L to Delta Cloud Infrastructure</strong> and <strong>₹3.8L to Prime Logistics</strong>) "
        f"are scheduled <em>before</em> key customer receivables arrive."
    )

    # 3. CRM Collections Breakdown
    insights.append(
        f"CRM pipeline shows <strong>+₹{tot_in/1e5:.1f} Lakh</strong> in expected collections across {len(receivables)} invoices, "
        f"countered by <strong>−₹{tot_out/1e5:.1f} Lakh</strong> in supplier bills and <strong>−₹{tot_exp/1e5:.1f} Lakh</strong> "
        f"in recurring commitments (GST, Rent & Software)."
    )

    # 4. Action Item 1: High-Priority Receivables Follow-Up (Matching User Vision!)
    key_invoice = next((r for r in receivables if r["id"] == "INV-2026-081"), receivables[0])
    inv_amt_str = f"₹{key_invoice['amount']/1e5:.1f} Lakh"
    inv_due_str = pd.to_datetime(key_invoice["due_date"]).strftime("%B %d")

    recommendations.append(
        f"<strong>Prioritize Collection:</strong> Follow up immediately on <strong>{inv_amt_str}</strong> of receivables "
        f"(Invoice #{key_invoice['id']} — <em>{key_invoice['client']}</em>) due on <strong>{inv_due_str}</strong>. "
        f"Securing this payment on or before the due date prevents your bank balance from dipping below ₹3.0 Lakh."
    )

    # 5. Action Item 2: Vendor Payment Restructuring
    recommendations.append(
        f"<strong>Restructure Supplier Disbursements:</strong> Contact <em>Delta Cloud & Server Infrastructure</em> "
        f"(Bill #BILL-901 for ₹4.2L) to split payment into two 50% tranches (₹2.1L due now, ₹2.1L Net 30). "
        f"This immediately preserves ₹2.1 Lakh in operational cushion during the mid-month crunch."
    )

    # 6. Action Item 3: Secondary Invoice Confirmation
    sec_invoice = next((r for r in receivables if r["id"] == "INV-2026-088"), None)
    if sec_invoice:
        sec_due_str = pd.to_datetime(sec_invoice["due_date"]).strftime("%B %d")
        recommendations.append(
            f"Confirm delivery milestones for <em>{sec_invoice['client']}</em> (₹{sec_invoice['amount']/1e5:.1f}L due {sec_due_str}) "
            f"to guarantee early clearance before the Prime Logistics inventory bill hits."
        )

    # Alerts
    if worst_bal < 250000:
        alerts.append({
            "type": "warning",
            "text": f"Liquidity Warning: Balance falls to ₹{worst_bal/1e5:.1f}L around {worst_dt_str}. Action required to preserve safe buffer."
        })

    return {
        "insights": insights,
        "recommendations": recommendations,
        "alerts": alerts
    }


# ─────────────────────────────────────────────────────────────
#  COMPLETE CRM PIPELINE FORECAST RUNNER
# ─────────────────────────────────────────────────────────────

def run_crm_full_forecast(horizon_days=30):
    """
    Executes the full CRM-powered hybrid forecast:
      1. Generates 180-day baseline transactions & runs Prophet+XGBoost ensemble
      2. Generates CRM forward schedule (invoices, payables, commitments)
      3. Blends statistical daily baseline with verified CRM forward ledger
      4. Calculates running balance from current bank balance (₹6.2 Lakh)
      5. Generates hyper-specific AI explanations with invoice & vendor names
    """
    import app as main_app

    base_date = get_base_date()

    # 1. Historical statement
    hist_raw = generate_crm_historical_transactions(180, base_date)
    # Parse through normalizer
    hist_raw.columns = ["date", "amount", "description"]
    hist_raw["ds"] = pd.to_datetime(hist_raw["date"], format="%d-%m-%Y")
    hist_clean = hist_raw[["ds", "amount", "description"]].sort_values("ds").reset_index(drop=True)

    # 2. Ensemble ML baseline
    forecast_df, model_used, outlier_info = main_app.run_ensemble_forecast(hist_clean, horizon_days=horizon_days)

    # 3. CRM Scheduled Calendar
    crm_calendar = build_crm_scheduled_calendar(horizon_days, base_date)
    augmented_df = blend_crm_with_statistical_forecast(forecast_df, crm_calendar)

    # 4. Running balances from current bank balance
    curr_balance = CRM_INFO["current_bank_balance"]
    running = curr_balance
    daily_balances = []

    for _, row in augmented_df.iterrows():
        running += row["predicted_total"]
        daily_balances.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "balance": round(running, 2),
            "events": row["crm_events"]
        })

    worst_item = min(daily_balances, key=lambda x: x["balance"])
    worst_info = {
        "worst_date": worst_item["date"],
        "worst_balance": worst_item["balance"]
    }

    crm_stats = get_crm_summary_stats(base_date)
    receivables = generate_crm_receivables(base_date)
    payables = generate_crm_payables(base_date)
    commitments = generate_crm_commitments(base_date)

    projected_final = daily_balances[-1]["balance"] if daily_balances else curr_balance

    summary = {
        "current_balance": curr_balance,
        "opening_balance": curr_balance,
        "expected_inflow": crm_stats["total_receivables"],
        "expected_outflow": crm_stats["total_payables"],
        "projected_expenses": crm_stats["total_commitments"],
        "projected_balance": round(projected_final, 2),
        "worst_date": worst_info["worst_date"],
        "worst_balance": round(worst_info["worst_balance"], 2),
        "data_points": len(hist_clean),
        "date_range": {
            "start": hist_clean["ds"].min().strftime("%Y-%m-%d"),
            "end": hist_clean["ds"].max().strftime("%Y-%m-%d")
        }
    }

    # 5. CRM AI Insights
    explanation = generate_crm_ai_explanation(
        summary, crm_stats, receivables, payables, commitments, worst_info
    )

    # 6. Format series for Chart.js
    forecast_series = []
    for _, row in augmented_df.iterrows():
        forecast_series.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "predicted": round(row["predicted_total"], 2),
            "lower": round(row["lower_total"], 2),
            "upper": round(row["upper_total"], 2),
            "crm_events": row["crm_events"]
        })

    hist_daily = hist_clean.groupby("ds")["amount"].sum().reset_index()
    hist_series = [
        {"date": row["ds"].strftime("%Y-%m-%d"), "amount": round(row["amount"], 2)}
        for _, row in hist_daily.iterrows()
    ]

    return {
        "success": True,
        "is_crm_source": True,
        "model_used": f"{model_used} + CRM Forward Ledger",
        "crm_info": CRM_INFO,
        "crm_stats": crm_stats,
        "receivables": receivables,
        "payables": payables,
        "commitments": commitments,
        "outlier_info": outlier_info,
        "summary": summary,
        "forecast_series": forecast_series,
        "historical_series": hist_series,
        "explanation": explanation
    }

