"""
BB+ Cash Flow Forecast — Real-World Evaluation & Benchmarking Suite
====================================================================
Tests the entire forecasting pipeline against:
  1. Realistic multi-pattern synthetic datasets (7 difficulty scenarios)
  2. Any user-provided Kaggle/real CSV dropped into the project directory
  3. Comprehensive metrics with improvement recommendations

Run:  python kaggle_evaluation.py
"""

import os
import sys
import json
import warnings
import traceback
from datetime import datetime, timedelta
from io import StringIO

# Fix Windows console encoding for UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import our main app pipeline
import app as main_app


# ═══════════════════════════════════════════════════════════════
#  SECTION 1: REALISTIC TEST DATA GENERATORS
#  7 scenarios covering real-world cash flow challenges
# ═══════════════════════════════════════════════════════════════

def generate_scenario_1_small_business_india(days=365, seed=42):
    """
    Scenario 1: Indian Small Business (Retail/Manufacturing)
    - Daily sales with weekend dips (Sat/Sun ~40% of weekday)
    - Monthly salary outflow on 1st (~₹4L)
    - Supplier payments on 5th, 15th, 25th (~₹1.2L each)
    - Quarterly GST payment (~₹2.5L)
    - Diwali/Festive season spike (Oct-Nov: +60% sales)
    - Monsoon slowdown (Jul-Aug: -25% sales)
    - Rent on 10th (~₹55K)
    - Random large B2B receivables (5-10 per year, ₹2-8L)
    - Occasional outlier (equipment purchase, one-time ₹15L)
    """
    np.random.seed(seed)
    today = datetime.today()
    start = today - timedelta(days=days)
    records = []

    for i in range(days):
        d = start + timedelta(days=i)
        dow = d.weekday()  # 0=Mon, 6=Sun
        month = d.month

        # --- Daily sales ---
        base_sales = 22000
        # Seasonal modulation
        if month in [10, 11]:  # Diwali/festive
            base_sales *= 1.60
        elif month in [7, 8]:  # Monsoon
            base_sales *= 0.75
        elif month in [3]:  # Year-end rush
            base_sales *= 1.30
        # Weekend dip
        if dow >= 5:
            base_sales *= 0.40
        # Add noise
        daily_sale = max(0, np.random.normal(base_sales, base_sales * 0.25))
        records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(daily_sale, 2),
                        "description": "Daily Sales Collection"})

        # --- Salary on 1st ---
        if d.day == 1:
            salary = np.random.normal(400000, 20000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-salary, 2),
                            "description": "Staff Salary Payment"})

        # --- Supplier payments on 5th, 15th, 25th ---
        if d.day in [5, 15, 25]:
            supplier = np.random.normal(120000, 15000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-supplier, 2),
                            "description": "Supplier Payment"})

        # --- Rent on 10th ---
        if d.day == 10:
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": -55000,
                            "description": "Office/Factory Rent"})

        # --- Quarterly GST (1st of Jan, Apr, Jul, Oct) ---
        if d.day == 20 and month in [1, 4, 7, 10]:
            gst = np.random.normal(250000, 30000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-gst, 2),
                            "description": "GST Payment"})

        # --- Random B2B receivable (large invoice clearance) ---
        if np.random.random() < 0.025:  # ~2.5% chance per day ≈ 9/year
            receivable = np.random.uniform(200000, 800000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(receivable, 2),
                            "description": "B2B Invoice Payment Received"})

    # --- One-time outlier: Equipment purchase (day ~120) ---
    outlier_date = start + timedelta(days=120)
    records.append({"date": outlier_date.strftime("%d-%m-%Y"), "amount": -1500000,
                    "description": "Capital Equipment Purchase"})

    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["date"], dayfirst=True)
    return df, "Indian Small Business (Retail/Mfg)"


def generate_scenario_2_freelancer_irregular(days=365, seed=123):
    """
    Scenario 2: Freelancer / Gig Economy (Highly Irregular)
    - Income arrives in bursts (project completions every 15-45 days)
    - Variable project sizes (₹30K - ₹3L)
    - Fixed monthly expenses (rent ₹25K, utilities ₹8K, subscriptions ₹5K)
    - Irregular expense spikes (travel, equipment)
    - Many zero-income days (tests gap handling)
    """
    np.random.seed(seed)
    today = datetime.today()
    start = today - timedelta(days=days)
    records = []

    # Pre-generate project payment dates
    payment_dates = []
    d = 0
    while d < days:
        d += np.random.randint(15, 45)
        if d < days:
            payment_dates.append(d)

    for i in range(days):
        d = start + timedelta(days=i)

        # --- Project payments (irregular bursts) ---
        if i in payment_dates:
            project_val = np.random.uniform(30000, 300000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(project_val, 2),
                            "description": "Project Payment Received"})

        # --- Fixed monthly expenses ---
        if d.day == 1:
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": -25000, "description": "Rent"})
        if d.day == 5:
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-np.random.normal(8000, 1500), 2),
                            "description": "Utilities"})
        if d.day == 15:
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": -5000, "description": "Subscriptions"})

        # --- Small daily expenses on ~60% of days ---
        if np.random.random() < 0.60:
            expense = np.random.uniform(200, 3000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-expense, 2),
                            "description": "Daily Expense"})

        # --- Occasional travel/equipment ---
        if np.random.random() < 0.015:
            big_expense = np.random.uniform(15000, 80000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-big_expense, 2),
                            "description": "Travel/Equipment"})

    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["date"], dayfirst=True)
    return df, "Freelancer / Gig Economy (Irregular)"


def generate_scenario_3_ecommerce_seasonal(days=365, seed=456):
    """
    Scenario 3: E-Commerce Business (Strong Seasonality)
    - High daily order volume with clear weekly/monthly patterns
    - Festival spikes: Diwali (Oct-Nov), End of Season Sale (Jan, Jul)
    - Steady growth trend (+0.1% daily)
    - Refund outflows (~8% of daily sales, delayed by 3-7 days)
    - Marketing spend spikes before festivals
    - Shipping costs proportional to sales
    """
    np.random.seed(seed)
    today = datetime.today()
    start = today - timedelta(days=days)
    records = []

    for i in range(days):
        d = start + timedelta(days=i)
        dow = d.weekday()
        month = d.month

        # Base daily revenue with growth trend
        base = 45000 * (1 + 0.001 * i)  # 0.1% daily growth

        # Weekly pattern: weekends are higher for e-commerce
        if dow in [5, 6]:
            base *= 1.35
        elif dow == 0:  # Monday dip
            base *= 0.85

        # Seasonal spikes
        if month in [10, 11]:
            base *= 1.80  # Diwali
        elif month in [1, 7]:
            base *= 1.40  # EOSS
        elif month in [2, 6]:
            base *= 0.70  # Slow months

        revenue = max(0, np.random.normal(base, base * 0.20))
        records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(revenue, 2),
                        "description": "Order Revenue"})

        # Refunds (~8%, delayed effect modeled as same-day for simplicity)
        refund = revenue * np.random.uniform(0.05, 0.12)
        records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-refund, 2),
                        "description": "Customer Refund"})

        # Shipping costs (~12% of revenue)
        shipping = revenue * np.random.uniform(0.08, 0.16)
        records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-shipping, 2),
                        "description": "Shipping/Logistics"})

        # Marketing spend (higher before festivals)
        if month in [9, 10]:  # Pre-Diwali marketing
            mkt = np.random.uniform(8000, 25000)
        else:
            mkt = np.random.uniform(3000, 10000)
        if d.day in [1, 15]:  # Bi-monthly marketing spend
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-mkt, 2),
                            "description": "Marketing Spend"})

        # Vendor/supplier payments (weekly)
        if dow == 2:  # Tuesday vendor payments
            vendor = np.random.uniform(50000, 150000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-vendor, 2),
                            "description": "Vendor Payment"})

    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["date"], dayfirst=True)
    return df, "E-Commerce (Strong Seasonality + Trend)"


def generate_scenario_4_saas_subscription(days=365, seed=789):
    """
    Scenario 4: SaaS / Subscription Business
    - Recurring monthly revenue (growing subscriber base)
    - Annual contracts renewing in clusters
    - Server/cloud costs (monthly, growing)
    - Payroll (bi-weekly)
    - Churn-related revenue dips
    - Quarter-end enterprise deal closings
    """
    np.random.seed(seed)
    today = datetime.today()
    start = today - timedelta(days=days)
    records = []

    base_mrr = 800000  # Starting MRR ₹8L
    monthly_growth = 0.04  # 4% MoM growth

    for i in range(days):
        d = start + timedelta(days=i)
        months_elapsed = i / 30.0
        current_mrr = base_mrr * (1 + monthly_growth) ** months_elapsed

        # Daily recognition of MRR (spread across working days)
        if d.weekday() < 5:
            daily_rev = current_mrr / 22  # ~22 working days
            daily_rev *= np.random.normal(1.0, 0.15)

            # Quarter-end enterprise closings boost
            if d.month in [3, 6, 9, 12] and d.day >= 25:
                daily_rev *= np.random.uniform(1.5, 3.0)

            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(max(0, daily_rev), 2),
                            "description": "Subscription Revenue"})

        # Annual contract renewals (clusters in Jan, Apr)
        if d.day == 1 and d.month in [1, 4]:
            annual = np.random.uniform(500000, 1500000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(annual, 2),
                            "description": "Annual Contract Renewal"})

        # Payroll (1st and 15th)
        if d.day in [1, 15]:
            payroll = np.random.normal(350000, 20000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-payroll, 2),
                            "description": "Payroll"})

        # Cloud/server costs (5th of month)
        if d.day == 5:
            cloud = current_mrr * 0.15 * np.random.normal(1.0, 0.05)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-cloud, 2),
                            "description": "Cloud Infrastructure"})

        # Churn refunds (random, ~2% chance)
        if np.random.random() < 0.02:
            churn = np.random.uniform(5000, 50000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-churn, 2),
                            "description": "Churn Refund"})

    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["date"], dayfirst=True)
    return df, "SaaS / Subscription Business"


def generate_scenario_5_construction_lumpy(days=365, seed=101):
    """
    Scenario 5: Construction / Project-Based Business
    - Extremely lumpy cash flows (milestone payments)
    - Large payments in (₹10-50L) every 30-60 days
    - Steady daily labor costs (₹15-25K)
    - Material purchases in bursts (₹5-20L)
    - Very challenging for time-series models
    """
    np.random.seed(seed)
    today = datetime.today()
    start = today - timedelta(days=days)
    records = []

    # Generate milestone payment dates (every 30-60 days)
    milestone_days = []
    d = 0
    while d < days:
        d += np.random.randint(30, 60)
        if d < days:
            milestone_days.append(d)

    # Generate material purchase dates
    material_days = []
    d = 5
    while d < days:
        d += np.random.randint(20, 45)
        if d < days:
            material_days.append(d)

    for i in range(days):
        d = start + timedelta(days=i)

        # Daily labor costs (weekdays only)
        if d.weekday() < 6:  # Mon-Sat
            labor = np.random.normal(20000, 3000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-labor, 2),
                            "description": "Daily Labor Wages"})

        # Milestone payments (large inflows)
        if i in milestone_days:
            milestone = np.random.uniform(1000000, 5000000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(milestone, 2),
                            "description": "Project Milestone Payment"})

        # Material purchases (large outflows)
        if i in material_days:
            material = np.random.uniform(500000, 2000000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-material, 2),
                            "description": "Material Purchase"})

        # Equipment rental (monthly)
        if d.day == 20:
            rental = np.random.normal(75000, 10000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-rental, 2),
                            "description": "Equipment Rental"})

        # Small miscellaneous
        if np.random.random() < 0.3:
            misc = np.random.uniform(1000, 8000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-misc, 2),
                            "description": "Misc Site Expense"})

    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["date"], dayfirst=True)
    return df, "Construction / Project-Based (Lumpy)"


def generate_scenario_6_restaurant_chain(days=365, seed=202):
    """
    Scenario 6: Restaurant Chain (Multi-Location)
    - Strong day-of-week patterns (Fri/Sat highest, Mon/Tue lowest)
    - Lunch vs dinner split doesn't matter for daily totals
    - Festival/holiday boosts
    - Steady ingredient costs (~35% of revenue)
    - Staff costs on 1st and 15th
    - Utility bills monthly
    """
    np.random.seed(seed)
    today = datetime.today()
    start = today - timedelta(days=days)
    records = []

    for i in range(days):
        d = start + timedelta(days=i)
        dow = d.weekday()
        month = d.month

        # Base daily revenue with strong weekly pattern
        dow_multipliers = {0: 0.65, 1: 0.70, 2: 0.85, 3: 0.95, 4: 1.30, 5: 1.45, 6: 1.10}
        base = 85000 * dow_multipliers[dow]

        # Festival boosts
        if month == 12:  # Dec party season
            base *= 1.40
        elif month in [10, 11]:  # Diwali
            base *= 1.25
        elif month in [1, 2]:  # Post-holiday dip
            base *= 0.80

        revenue = max(0, np.random.normal(base, base * 0.18))
        records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(revenue, 2),
                        "description": "Restaurant Revenue"})

        # Ingredient costs (~35% of revenue, purchased daily)
        ingredients = revenue * np.random.uniform(0.30, 0.40)
        records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-ingredients, 2),
                        "description": "Ingredient Purchase"})

        # Staff costs
        if d.day in [1, 15]:
            staff = np.random.normal(250000, 15000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-staff, 2),
                            "description": "Staff Salary"})

        # Utility bills (10th)
        if d.day == 10:
            utility = np.random.normal(45000, 5000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(-utility, 2),
                            "description": "Utility Bills"})

        # Rent (1st)
        if d.day == 1:
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": -120000,
                            "description": "Restaurant Rent"})

    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["date"], dayfirst=True)
    return df, "Restaurant Chain (Weekly Patterns)"


def generate_scenario_7_stress_test(days=365, seed=303):
    """
    Scenario 7: Adversarial Stress Test
    - Sudden trend reversal at day 180
    - Multiple outliers (5 extreme transactions)
    - Missing data gaps (30-day gap in middle)
    - Mixed positive/negative near-zero values
    - Designed to break naive models
    """
    np.random.seed(seed)
    today = datetime.today()
    start = today - timedelta(days=days)
    records = []

    for i in range(days):
        d = start + timedelta(days=i)

        # Skip 30-day gap (days 100-130)
        if 100 <= i <= 130:
            continue

        # Trend reversal at day 180
        if i < 180:
            base = 15000 + 100 * i  # Growing trend
        else:
            base = 15000 + 100 * 180 - 150 * (i - 180)  # Declining trend

        base = max(base, 2000)
        flow = np.random.normal(base, base * 0.30)

        # Mix in negative days (~35%)
        if np.random.random() < 0.35:
            flow = -abs(flow) * np.random.uniform(0.5, 1.5)

        records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(flow, 2),
                        "description": "Transaction"})

    # Add 5 extreme outliers
    outlier_days = [50, 160, 220, 280, 340]
    for od in outlier_days:
        if od < days:
            d = start + timedelta(days=od)
            sign = np.random.choice([-1, 1])
            magnitude = np.random.uniform(500000, 2000000)
            records.append({"date": d.strftime("%d-%m-%Y"), "amount": round(sign * magnitude, 2),
                            "description": "OUTLIER Transaction"})

    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["date"], dayfirst=True)
    return df, "Adversarial Stress Test"


# ═══════════════════════════════════════════════════════════════
#  SECTION 2: LOAD USER-PROVIDED KAGGLE / REAL CSV
# ═══════════════════════════════════════════════════════════════

def load_kaggle_csv(filepath):
    """
    Auto-detects column mapping from any financial CSV.
    Handles: date, amount, credit/debit, transaction_amount, cash_flow etc.
    Returns standardized DataFrame with [ds, amount, description] columns.
    """
    df = pd.read_csv(filepath, encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # --- Find date column ---
    date_candidates = ["date", "transaction_date", "txn_date", "value_date", "trans_date",
                       "posting_date", "entry_date", "datetime", "timestamp"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        # Try to find any column with 'date' in name
        date_col = next((c for c in df.columns if "date" in c), None)
    if date_col is None:
        raise ValueError(f"No date column found. Available: {list(df.columns)}")

    # --- Find amount column ---
    amount_candidates = ["amount", "transaction_amount", "net_amount", "value",
                         "cash_flow", "debit_credit", "total"]
    amount_col = next((c for c in amount_candidates if c in df.columns), None)

    credit_candidates = ["credit", "credit_amount", "inflow", "receipts", "income"]
    debit_candidates = ["debit", "debit_amount", "outflow", "payments", "expense", "expenditure"]
    credit_col = next((c for c in credit_candidates if c in df.columns), None)
    debit_col = next((c for c in debit_candidates if c in df.columns), None)

    # --- Find description column ---
    desc_candidates = ["description", "narration", "particulars", "remarks", "note",
                       "category", "subcategory", "account_type", "type"]
    desc_col = next((c for c in desc_candidates if c in df.columns), None)

    # --- Find income/expense indicator ---
    ie_candidates = ["income/expense", "income_expense", "type", "txn_type", "cr/dr", "cr_dr"]
    ie_col = next((c for c in ie_candidates if c in df.columns), None)

    # Parse dates
    df["ds"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    # Try alternate parsing if too many NaTs
    nat_ratio = df["ds"].isna().mean()
    if nat_ratio > 0.5:
        df["ds"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["ds"])

    if len(df) < 10:
        raise ValueError(f"Too few valid date rows ({len(df)}). Check date format.")

    # Parse amounts
    if amount_col:
        df["amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
        # If there's an income/expense indicator, use it to set sign
        if ie_col and ie_col != amount_col:
            ie_vals = df[ie_col].astype(str).str.strip().str.lower()
            expense_mask = ie_vals.str.contains("expense|debit|dr|outflow|payment", na=False)
            df.loc[expense_mask & (df["amount"] > 0), "amount"] *= -1
    elif credit_col and debit_col:
        df["credit_val"] = pd.to_numeric(df[credit_col], errors="coerce").fillna(0)
        df["debit_val"] = pd.to_numeric(df[debit_col], errors="coerce").fillna(0)
        df["amount"] = df["credit_val"] - df["debit_val"]
    else:
        raise ValueError(f"No amount column found. Available: {list(df.columns)}")

    df["description"] = df[desc_col].astype(str) if desc_col else "Transaction"

    result = df[["ds", "amount", "description"]].sort_values("ds").reset_index(drop=True)
    print(f"  Date range: {result['ds'].min().date()} to {result['ds'].max().date()}")
    print(f"  Total transactions: {len(result)}, Net flow: Rs.{result['amount'].sum():,.0f}")
    return result


def scan_for_kaggle_csvs():
    """Scans project directory for any CSV files that could be Kaggle datasets."""
    csvs = []
    for f in os.listdir(BASE_DIR):
        if f.lower().endswith(".csv") and f != "test_sample.csv":
            fpath = os.path.join(BASE_DIR, f)
            try:
                size_kb = os.path.getsize(fpath) / 1024
                csvs.append((f, fpath, size_kb))
            except Exception:
                pass

    # Also check uploads/ folder
    uploads = os.path.join(BASE_DIR, "uploads")
    if os.path.isdir(uploads):
        for f in os.listdir(uploads):
            if f.lower().endswith(".csv"):
                fpath = os.path.join(uploads, f)
                try:
                    size_kb = os.path.getsize(fpath) / 1024
                    csvs.append((f, fpath, size_kb))
                except Exception:
                    pass

    return csvs


# ═══════════════════════════════════════════════════════════════
#  SECTION 3: EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════

def compute_all_metrics(y_true, y_pred, y_train, lower=None, upper=None):
    """Computes comprehensive metrics for a single model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    medae = float(np.median(np.abs(y_true - y_pred)))

    # WAPE
    tot_act = float(np.sum(np.abs(y_true)))
    wape = float((np.sum(np.abs(y_true - y_pred)) / tot_act) * 100.0) if tot_act > 0 else 0.0

    # sMAPE
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-6
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)

    # MASE
    naive_mae = np.mean(np.abs(np.diff(y_train)))
    mase = float(mae / naive_mae) if naive_mae > 0 else 1.0

    # MDA (directional accuracy)
    mda = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100.0)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Cumulative balance tracking
    opening = 500000  # Rs.5L default opening
    act_bal = opening + np.cumsum(y_true)
    pred_bal = opening + np.cumsum(y_pred)
    cum_bal_mae = float(np.mean(np.abs(act_bal - pred_bal)))
    final_bal_err = float(np.abs(act_bal[-1] - pred_bal[-1]))

    # Coverage (if CI provided)
    coverage = None
    interval_width = None
    if lower is not None and upper is not None:
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        in_bounds = (y_true >= lower) & (y_true <= upper)
        coverage = float(np.mean(in_bounds) * 100.0)
        interval_width = float(np.mean(upper - lower))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "medae": round(medae, 2),
        "wape": round(wape, 2),
        "smape": round(smape, 2),
        "mase": round(mase, 3),
        "mda": round(mda, 1),
        "r2": round(r2, 3),
        "cum_bal_mae": round(cum_bal_mae, 2),
        "final_bal_err": round(final_bal_err, 2),
        "coverage": round(coverage, 1) if coverage is not None else None,
        "interval_width": round(interval_width, 2) if interval_width is not None else None,
    }


def evaluate_single_dataset(df, scenario_name, horizon_days=30):
    """
    Runs full holdout evaluation on a single dataset.
    Returns comprehensive results dict.
    """
    print(f"\n{'─'*70}")
    print(f"  EVALUATING: {scenario_name}")
    print(f"{'─'*70}")

    # Prepare daily series
    daily, gap_ratio = main_app.prepare_daily_series(df[["ds", "amount"]])
    total_days = len(daily)
    print(f"  Daily series: {total_days} days, gap ratio: {gap_ratio:.1%}")

    if total_days < horizon_days + 60:
        print(f"  [SKIP] Not enough data ({total_days} days, need {horizon_days + 60})")
        return None

    # Train/test split
    train_daily = daily.iloc[:-horizon_days].copy()
    test_daily = daily.iloc[-horizon_days:].copy()
    y_train = train_daily["y"].values
    y_test = test_daily["y"].values

    results = {
        "scenario": scenario_name,
        "total_days": total_days,
        "train_days": len(train_daily),
        "test_days": horizon_days,
        "gap_ratio": round(gap_ratio, 3),
        "train_mean": round(float(y_train.mean()), 2),
        "train_std": round(float(y_train.std()), 2),
        "test_mean": round(float(y_test.mean()), 2),
        "test_std": round(float(y_test.std()), 2),
        "models": {},
        "series": {
            "dates": test_daily["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "actual": [round(float(v), 2) for v in y_test],
        }
    }

    # --- Baseline 1: Naive Persistence ---
    print("  [1/6] Naive Persistence...", end=" ")
    last_val = float(y_train[-1])
    naive_pred = np.repeat(last_val, horizon_days)
    results["models"]["Naive Persistence"] = compute_all_metrics(y_test, naive_pred, y_train)
    print(f"MAE={results['models']['Naive Persistence']['mae']:,.0f}")

    # --- Baseline 2: 14d Rolling Mean ---
    print("  [2/6] Rolling Mean (14d)...", end=" ")
    mean_val = float(pd.Series(y_train).iloc[-14:].mean())
    std_val = float(pd.Series(y_train).iloc[-14:].std() or 0.0)
    mean_pred = np.repeat(mean_val, horizon_days)
    mean_lo = mean_pred - 1.28 * std_val
    mean_hi = mean_pred + 1.28 * std_val
    results["models"]["14d Rolling Mean"] = compute_all_metrics(y_test, mean_pred, y_train, mean_lo, mean_hi)
    print(f"MAE={results['models']['14d Rolling Mean']['mae']:,.0f}")

    # --- Baseline 3: Seasonal Naive (7d) ---
    print("  [3/6] Seasonal Naive (7d)...", end=" ")
    last_week = y_train[-7:]
    repeats = int(np.ceil(horizon_days / 7))
    snaive_pred = np.tile(last_week, repeats)[:horizon_days]
    results["models"]["Seasonal Naive (7d)"] = compute_all_metrics(y_test, snaive_pred, y_train)
    print(f"MAE={results['models']['Seasonal Naive (7d)']['mae']:,.0f}")

    # --- Model 4: Prophet ---
    print("  [4/6] Facebook Prophet...", end=" ", flush=True)
    try:
        train_clean, outlier_info = main_app.winsorize_outliers(train_daily)
        prophet_df = main_app.run_prophet(train_clean, gap_ratio, horizon_days)
        p_pred = prophet_df["predicted"].values
        p_lo = prophet_df["lower"].values
        p_hi = prophet_df["upper"].values
        results["models"]["Facebook Prophet"] = compute_all_metrics(y_test, p_pred, y_train, p_lo, p_hi)
        results["series"]["prophet"] = [round(float(v), 2) for v in p_pred]
        print(f"MAE={results['models']['Facebook Prophet']['mae']:,.0f}")
    except Exception as e:
        print(f"FAILED: {e}")
        p_pred = None
        results["models"]["Facebook Prophet"] = {"error": str(e)}

    # --- Model 5: XGBoost ---
    print("  [5/7] XGBoost Regressor...", end=" ", flush=True)
    try:
        if "train_clean" not in dir():
            train_clean, outlier_info = main_app.winsorize_outliers(train_daily)
        xgb_df = main_app.run_xgboost(train_clean, horizon_days)
        x_pred = xgb_df["predicted"].values
        x_lo = xgb_df["lower"].values
        x_hi = xgb_df["upper"].values
        results["models"]["XGBoost Regressor"] = compute_all_metrics(y_test, x_pred, y_train, x_lo, x_hi)
        results["series"]["xgboost"] = [round(float(v), 2) for v in x_pred]
        print(f"MAE={results['models']['XGBoost Regressor']['mae']:,.0f}")
    except Exception as e:
        print(f"FAILED: {e}")
        x_pred = None
        results["models"]["XGBoost Regressor"] = {"error": str(e)}

    # --- Model 6: LightGBM ---
    print("  [6/7] LightGBM Regressor...", end=" ", flush=True)
    try:
        lgb_df = main_app.run_lightgbm(train_clean, horizon_days)
        l_pred = lgb_df["predicted"].values
        l_lo = lgb_df["lower"].values
        l_hi = lgb_df["upper"].values
        results["models"]["LightGBM Regressor"] = compute_all_metrics(y_test, l_pred, y_train, l_lo, l_hi)
        results["series"]["lightgbm"] = [round(float(v), 2) for v in l_pred]
        print(f"MAE={results['models']['LightGBM Regressor']['mae']:,.0f}")
    except Exception as e:
        print(f"FAILED: {e}")
        l_pred = None
        results["models"]["LightGBM Regressor"] = {"error": str(e)}

    # --- Model 7: Previous Baseline Ensemble (Fixed 55/45 Blend) ---
    if p_pred is not None and x_pred is not None:
        xgb_w = 0.55 if gap_ratio < 0.20 else 0.40
        prophet_w = 1.0 - xgb_w
        base_pred = prophet_w * p_pred + xgb_w * x_pred
        base_lo = prophet_w * p_lo + xgb_w * x_lo
        base_hi = prophet_w * p_hi + xgb_w * x_hi
        results["models"]["Baseline Ensemble (Fixed 55/45)"] = compute_all_metrics(y_test, base_pred, y_train, base_lo, base_hi)
        results["series"]["baseline_ensemble"] = [round(float(v), 2) for v in base_pred]

    # --- Model 8: NEW Stacking Ensemble (Prophet + XGBoost + LightGBM) ---
    print("  [7/7] Stacking Ensemble (Prophet + XGBoost + LightGBM)...", end=" ", flush=True)
    active_models = {}
    if p_pred is not None:
        active_models["Prophet"] = prophet_df
    if x_pred is not None:
        active_models["XGBoost"] = xgb_df
    if l_pred is not None:
        active_models["LightGBM"] = lgb_df

    if len(active_models) >= 2:
        weights, conf_half_width, fallback_info = main_app.calculate_stacking_weights(
            train_clean, gap_ratio, horizon_days, active_models, return_conformal=True
        )
        results["learned_weights"] = {k: round(v, 3) for k, v in weights.items()}
        
        stack_pred = np.zeros(horizon_days)
        for name, res in active_models.items():
            w = weights[name]
            stack_pred += w * res["predicted"].values

        # Adaptive Conformal Inference (ACI): scales with horizon drift and predicted volatility surge
        steps = np.arange(horizon_days)
        horizon_factors = 1.0 + 0.40 * np.sqrt(steps / float(horizon_days))
        train_mean = float(np.mean(train_clean["y"]))
        train_std = float(np.std(train_clean["y"]))
        volatility_factors = 1.0 + 0.50 * np.clip(np.abs(stack_pred - train_mean) / (train_std + 1e-5), 0, 2.0)
        dynamic_half_width = conf_half_width * horizon_factors * volatility_factors

        # Runtime Model Selector: If fallback triggered, route prediction to winning baseline
        if fallback_info.get("is_fallback", False):
            stack_pred = fallback_info["forecast"]
            results["fallback_triggered"] = {
                "model": fallback_info["model"],
                "reason": fallback_info["reason"],
                "cv": fallback_info.get("cv"),
                "r2_val": fallback_info.get("r2_val")
            }

        stack_lo = stack_pred - dynamic_half_width
        stack_hi = stack_pred + dynamic_half_width

        stack_metrics = compute_all_metrics(y_test, stack_pred, y_train, stack_lo, stack_hi)
        results["models"]["Stacking Ensemble"] = stack_metrics
        results["models"]["Ensemble"] = stack_metrics  # Primary model for downstream reporting
        results["series"]["ensemble"] = [round(float(v), 2) for v in stack_pred]
        results["series"]["ensemble_lower"] = [round(float(v), 2) for v in stack_lo]
        results["series"]["ensemble_upper"] = [round(float(v), 2) for v in stack_hi]

        # Calculate improvement vs baseline
        if "Baseline Ensemble (Fixed 55/45)" in results["models"]:
            b_mae = results["models"]["Baseline Ensemble (Fixed 55/45)"]["mae"]
            s_mae = stack_metrics["mae"]
            imp_pct = ((b_mae - s_mae) / b_mae) * 100.0 if b_mae > 0 else 0.0
            results["stacking_improvement"] = {
                "mae_reduction_pct": round(imp_pct, 2),
                "mae_difference": round(b_mae - s_mae, 2),
                "mase_reduction": round(results["models"]["Baseline Ensemble (Fixed 55/45)"]["mase"] - stack_metrics["mase"], 3),
                "r2_gain": round(stack_metrics["r2"] - results["models"]["Baseline Ensemble (Fixed 55/45)"]["r2"], 3),
            }
        w_str = ", ".join([f"{k}: {int(round(v*100))}%" for k, v in weights.items()])
        if fallback_info.get("is_fallback", False):
            print(f"MAE={stack_metrics['mae']:,.0f} [ROUTED TO FALLBACK: {fallback_info['model']} ({fallback_info['reason']})]")
        else:
            print(f"MAE={stack_metrics['mae']:,.0f} [Weights: {w_str}]")
    elif len(active_models) == 1:
        name = next(iter(active_models.keys()))
        results["models"]["Ensemble"] = results["models"][f"{name} Regressor" if name != "Prophet" else "Facebook Prophet"].copy()
        print(f"({name} only)")
    else:
        print("ALL MODELS FAILED")
        results["models"]["Ensemble"] = {"error": "All models failed"}

    return results


# ═══════════════════════════════════════════════════════════════
#  SECTION 4: IMPROVEMENT ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════

def analyze_improvements(all_results):
    """
    Analyzes evaluation results across all scenarios and produces
    actionable improvement recommendations with priority ratings.
    """
    improvements = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
        "strengths": [],
    }

    mase_values = []
    mda_values = []
    wape_values = []
    coverage_values = []
    r2_values = []
    ensemble_wins = 0
    total_scenarios = 0

    for res in all_results:
        if res is None:
            continue
        total_scenarios += 1
        ens = res["models"].get("Ensemble", {})
        if "error" in ens:
            improvements["critical"].append(
                f"[{res['scenario']}] Ensemble model failed entirely: {ens['error']}"
            )
            continue

        mase_values.append(ens.get("mase", 2.0))
        mda_values.append(ens.get("mda", 50.0))
        wape_values.append(ens.get("wape", 100.0))
        if ens.get("coverage") is not None:
            coverage_values.append(ens["coverage"])
        r2_values.append(ens.get("r2", 0.0))

        # Check if ensemble beats all baselines
        baselines = ["Naive Persistence", "14d Rolling Mean", "Seasonal Naive (7d)"]
        ens_mae = ens.get("mae", float("inf"))
        baseline_maes = [res["models"].get(b, {}).get("mae", float("inf")) for b in baselines]
        if ens_mae < min(baseline_maes):
            ensemble_wins += 1

        # Scenario-specific issues
        if ens.get("mase", 2.0) > 1.0:
            improvements["high"].append(
                f"[{res['scenario']}] MASE={ens['mase']:.3f} > 1.0 — "
                f"ensemble is WORSE than naive persistence. The model is not adding value here."
            )
        if ens.get("mda", 50.0) < 55.0:
            improvements["high"].append(
                f"[{res['scenario']}] MDA={ens['mda']:.1f}% — "
                f"directional accuracy near coin-flip. Model can't reliably predict inflow vs outflow direction."
            )
        if ens.get("wape", 100.0) > 80.0:
            improvements["medium"].append(
                f"[{res['scenario']}] WAPE={ens['wape']:.1f}% — "
                f"weighted error very high. Predictions deviate massively from actuals."
            )
        if ens.get("coverage") is not None and ens["coverage"] < 60.0:
            improvements["medium"].append(
                f"[{res['scenario']}] CI Coverage={ens['coverage']:.1f}% (target: 80%) — "
                f"confidence intervals are too narrow, underestimating uncertainty."
            )
        if ens.get("r2", 0.0) < 0.0:
            improvements["high"].append(
                f"[{res['scenario']}] R²={ens['r2']:.3f} < 0 — "
                f"model explains LESS variance than predicting the mean. Actively harmful."
            )

    # --- Aggregate analysis ---
    if mase_values:
        avg_mase = np.mean(mase_values)
        avg_mda = np.mean(mda_values)
        avg_wape = np.mean(wape_values)
        avg_r2 = np.mean(r2_values)

        if avg_mase < 0.8:
            improvements["strengths"].append(
                f"Avg MASE={avg_mase:.3f} < 0.8 — ensemble consistently outperforms naive baselines. Strong foundation."
            )
        elif avg_mase < 1.0:
            improvements["strengths"].append(
                f"Avg MASE={avg_mase:.3f} < 1.0 — ensemble is better than naive on average, but marginal on some scenarios."
            )

        if avg_mda > 70:
            improvements["strengths"].append(
                f"Avg MDA={avg_mda:.1f}% — good directional accuracy across scenarios."
            )

        if ensemble_wins == total_scenarios:
            improvements["strengths"].append(
                f"Ensemble beats ALL baselines in every scenario ({ensemble_wins}/{total_scenarios}). Solid model selection."
            )

    # --- General improvement recommendations ---
    improvements["recommended_improvements"] = []

    if any(m > 1.0 for m in mase_values):
        improvements["recommended_improvements"].append({
            "priority": "CRITICAL",
            "area": "Feature Engineering",
            "recommendation": "Add domain-specific features: month-end payment cycle indicators, "
                              "payroll day flags, GST quarter flags. Current lag/calendar features miss "
                              "these structured business patterns.",
            "expected_impact": "10-30% MAE reduction on business cash flow data"
        })

    if coverage_values and np.mean(coverage_values) < 75:
        improvements["recommended_improvements"].append({
            "priority": "HIGH",
            "area": "Uncertainty Quantification",
            "recommendation": "Replace residual-based CI with conformal prediction or quantile regression. "
                              "Current approach assumes Gaussian errors which is wrong for cash flow data.",
            "expected_impact": "Coverage from ~60% to target 80% without excessive interval width"
        })

    improvements["recommended_improvements"].extend([
        {
            "priority": "HIGH",
            "area": "Model Architecture",
            "recommendation": "Add LightGBM as a third ensemble member. Often outperforms XGBoost on "
                              "tabular data with less tuning. Use stacking instead of fixed-weight blending.",
            "expected_impact": "5-15% improvement in MAE through model diversity"
        },
        {
            "priority": "HIGH",
            "area": "Hyperparameter Optimization",
            "recommendation": "Implement Optuna-based hyperparameter search for XGBoost (n_estimators, max_depth, "
                              "learning_rate, subsample, colsample_bytree). Current fixed params are generic.",
            "expected_impact": "5-20% improvement depending on dataset characteristics"
        },
        {
            "priority": "MEDIUM",
            "area": "Feature Engineering - Fourier Terms",
            "recommendation": "Add Fourier terms for monthly and quarterly seasonality to capture "
                              "salary cycles and GST payment patterns that Prophet's built-in seasonality misses.",
            "expected_impact": "Better monthly/quarterly pattern capture"
        },
        {
            "priority": "MEDIUM",
            "area": "Ensemble Weighting",
            "recommendation": "Replace fixed 55/45 weights with learned weights using validation set performance. "
                              "Use inverse-MAE weighting or meta-learning (stacking with ridge regression).",
            "expected_impact": "2-8% improvement by adapting to each dataset's characteristics"
        },
        {
            "priority": "MEDIUM",
            "area": "Data Augmentation",
            "recommendation": "For sparse/irregular data (freelancer, construction), use bootstrap aggregation "
                              "or generate synthetic training examples to stabilize predictions.",
            "expected_impact": "Significant on irregular cash flow patterns"
        },
        {
            "priority": "LOW",
            "area": "Deep Learning",
            "recommendation": "Consider adding N-BEATS or Temporal Fusion Transformer for datasets with 2+ years "
                              "of history. Overkill for <1 year but powerful for longer series.",
            "expected_impact": "Potential 10-25% improvement on long series with complex patterns"
        },
        {
            "priority": "LOW",
            "area": "External Features",
            "recommendation": "Incorporate calendar holidays (Indian holidays API), weather data for "
                              "retail/restaurant scenarios, and economic indicators for B2B businesses.",
            "expected_impact": "Dataset-dependent, significant for retail and hospitality"
        },
    ])

    return improvements


# ═══════════════════════════════════════════════════════════════
#  SECTION 5: VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def generate_comprehensive_dashboard(all_results, output_dir=None):
    """
    Generates a multi-page evaluation dashboard with:
    1. Per-scenario actual vs predicted plots
    2. Cross-scenario metric comparison heatmap
    3. Model ranking summary
    """
    if output_dir is None:
        output_dir = BASE_DIR

    plt.style.use("dark_background")

    # --- Chart 1: Cross-Scenario Metrics Heatmap ---
    valid_results = [r for r in all_results if r is not None and "Ensemble" in r["models"] and "error" not in r["models"]["Ensemble"]]

    if not valid_results:
        print("[WARN] No valid results to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    accent_blue = "#38bdf8"
    accent_purple = "#a855f7"
    accent_green = "#22c55e"
    accent_red = "#ef4444"
    accent_amber = "#f59e0b"
    accent_pink = "#ec4899"

    for ax in axes.flat:
        ax.set_facecolor("#111827")
        ax.grid(True, linestyle="--", alpha=0.25, color="#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#1f2937")

    # ──── Panel 1: MAE Comparison Across Scenarios ────
    ax1 = axes[0, 0]
    scenario_names = [r["scenario"][:25] for r in valid_results]
    model_names = ["Naive Persistence", "Baseline Ensemble (Fixed 55/45)", "LightGBM Regressor", "Stacking Ensemble"]
    colors = [accent_red, accent_amber, accent_purple, accent_green]

    x = np.arange(len(scenario_names))
    width = 0.18
    for idx, (mname, color) in enumerate(zip(model_names, colors)):
        maes = []
        for r in valid_results:
            m = r["models"].get(mname, {})
            maes.append(m.get("mae", 0) / 1000)
        offset = (idx - 1.5) * width
        ax1.bar(x + offset, maes, width, color=color, alpha=0.85, label=mname)

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, rotation=30, ha="right", fontsize=8, color="#94a3b8")
    ax1.set_title("1. MAE Comparison Across Scenarios (Rs. Thousands)", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax1.set_ylabel("MAE (Rs. K)", color="#94a3b8")
    ax1.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155")

    # ──── Panel 2: MASE < 1 Threshold ────
    ax2 = axes[0, 1]
    mase_vals = [r["models"]["Ensemble"].get("mase", 2.0) for r in valid_results]
    bar_colors = [accent_green if m < 1.0 else accent_red for m in mase_vals]
    bars = ax2.bar(x, mase_vals, color=bar_colors, alpha=0.85, width=0.5)
    ax2.axhline(y=1.0, color="#f59e0b", linestyle="--", lw=2, alpha=0.8, label="MASE = 1.0 (Naive Baseline)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, rotation=30, ha="right", fontsize=8, color="#94a3b8")
    ax2.set_title("2. Ensemble MASE (< 1.0 = Better than Naive)", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax2.set_ylabel("MASE", color="#94a3b8")
    ax2.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155")
    for bar, val in zip(bars, mase_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="#cbd5e1")

    # ──── Panel 3: Directional Accuracy (MDA) ────
    ax3 = axes[1, 0]
    mda_vals = [r["models"]["Ensemble"].get("mda", 50.0) for r in valid_results]
    mda_colors = [accent_green if m > 65 else accent_amber if m > 55 else accent_red for m in mda_vals]
    bars3 = ax3.bar(x, mda_vals, color=mda_colors, alpha=0.85, width=0.5)
    ax3.axhline(y=50.0, color=accent_red, linestyle=":", lw=1.5, alpha=0.6, label="Random (50%)")
    ax3.axhline(y=65.0, color=accent_amber, linestyle="--", lw=1.5, alpha=0.6, label="Acceptable (65%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenario_names, rotation=30, ha="right", fontsize=8, color="#94a3b8")
    ax3.set_title("3. Directional Accuracy (MDA %) — Inflow vs Outflow", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax3.set_ylabel("MDA %", color="#94a3b8")
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155")
    for bar, val in zip(bars3, mda_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=9, color="#cbd5e1")

    # ──── Panel 4: R² Score ────
    ax4 = axes[1, 1]
    r2_vals = [r["models"]["Ensemble"].get("r2", 0.0) for r in valid_results]
    r2_colors = [accent_green if v > 0.3 else accent_amber if v > 0 else accent_red for v in r2_vals]
    bars4 = ax4.bar(x, r2_vals, color=r2_colors, alpha=0.85, width=0.5)
    ax4.axhline(y=0.0, color="#f8fafc", linestyle="--", lw=1, alpha=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenario_names, rotation=30, ha="right", fontsize=8, color="#94a3b8")
    ax4.set_title("4. R² Score (Variance Explained by Ensemble)", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax4.set_ylabel("R²", color="#94a3b8")
    ax4.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155")
    for bar, val in zip(bars4, r2_vals):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.02 if val >= 0 else -0.06),
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#cbd5e1")

    fig.suptitle("BB+ Cash Flow AI — Multi-Scenario Evaluation Dashboard",
                 color="#f8fafc", fontsize=16, fontweight="700", y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    path1 = os.path.join(output_dir, "kaggle_eval_metrics.png")
    fig.savefig(path1, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"\n[CHART] Metrics dashboard saved: {path1}")

    # --- Chart 2: Per-Scenario Actual vs Predicted Plots ---
    n_plots = len(valid_results)
    if n_plots == 0:
        return

    cols = 2
    rows = (n_plots + 1) // 2
    fig2, axes2 = plt.subplots(rows, cols, figsize=(20, 5 * rows), dpi=120)
    fig2.patch.set_facecolor("#0b0f19")
    if rows == 1:
        axes2 = axes2.reshape(1, -1)

    for idx, res in enumerate(valid_results):
        ax = axes2[idx // cols, idx % cols]
        ax.set_facecolor("#111827")
        ax.grid(True, linestyle="--", alpha=0.25, color="#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#1f2937")

        dates = [datetime.strptime(d, "%Y-%m-%d") for d in res["series"]["dates"]]
        actual = np.array(res["series"]["actual"])

        ax.plot(dates, actual / 1e3, label="Actual", color="#ffffff", lw=2, marker="o", markersize=3, zorder=5)

        if "ensemble" in res["series"]:
            ens = np.array(res["series"]["ensemble"])
            ax.plot(dates, ens / 1e3, label="Ensemble", color=accent_blue, lw=1.8, zorder=4)

            if "ensemble_lower" in res["series"]:
                lo = np.array(res["series"]["ensemble_lower"])
                hi = np.array(res["series"]["ensemble_upper"])
                ax.fill_between(dates, lo / 1e3, hi / 1e3, color=accent_blue, alpha=0.15)

        if "baseline_ensemble" in res["series"]:
            ax.plot(dates, np.array(res["series"]["baseline_ensemble"]) / 1e3,
                    label="Baseline Ens (Fixed)", color=accent_amber, lw=1.2, linestyle="--", alpha=0.6)
        if "lightgbm" in res["series"]:
            ax.plot(dates, np.array(res["series"]["lightgbm"]) / 1e3,
                    label="LightGBM", color=accent_pink, lw=1, linestyle=":", alpha=0.6)

        # Show key metrics in corner
        ens_m = res["models"].get("Ensemble", {})
        fb_note = f" | [Fallback: {res['fallback_triggered']['model']}]" if "fallback_triggered" in res else ""
        metric_text = f"MAE: Rs.{ens_m.get('mae', 0)/1e3:.1f}K | MASE: {ens_m.get('mase', 0):.2f} | MDA: {ens_m.get('mda', 0):.0f}%{fb_note}"
        ax.text(0.02, 0.95, metric_text, transform=ax.transAxes, fontsize=8,
                color="#94a3b8", verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="#1e293b", edgecolor="#334155", alpha=0.9))

        ax.set_title(f"{res['scenario']}", color="#f8fafc", fontsize=11, fontweight="600", pad=8)
        ax.set_ylabel("Net Flow (Rs. K)", color="#94a3b8", fontsize=9)
        ax.legend(fontsize=7, facecolor="#1e293b", edgecolor="#334155", loc="lower right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    # Hide empty subplots
    for idx in range(n_plots, rows * cols):
        axes2[idx // cols, idx % cols].set_visible(False)

    fig2.suptitle("BB+ Cash Flow AI — Actual vs Predicted (30-Day Holdout)",
                  color="#f8fafc", fontsize=15, fontweight="700", y=1.0)
    plt.tight_layout()

    path2 = os.path.join(output_dir, "kaggle_eval_forecasts.png")
    fig2.savefig(path2, facecolor=fig2.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig2)
    print(f"[CHART] Forecast comparison saved: {path2}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 6: REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════

def print_summary_report(all_results, improvements):
    """Prints a beautifully formatted terminal report."""
    print("\n")
    print("=" * 80)
    print("  BB+ CASH FLOW AI — COMPREHENSIVE EVALUATION REPORT")
    print("=" * 80)

    valid = [r for r in all_results if r is not None]
    print(f"\n  Scenarios evaluated: {len(valid)}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Per-Scenario Summary Table ---
    print("\n" + "-" * 80)
    print("  SCENARIO-WISE ENSEMBLE PERFORMANCE")
    print("-" * 80)

    header = f"{'Scenario':<28} {'MAE (Rs.)':<12} {'WAPE%':<8} {'MASE':<7} {'MDA%':<7} {'R2':<8} {'Coverage':<10}"
    print(f"  {header}")
    print(f"  {'─'*78}")

    for res in valid:
        ens = res["models"].get("Ensemble", {})
        if "error" in ens:
            print(f"  {res['scenario']:<28} FAILED: {ens['error'][:45]}")
            continue
        cov = f"{ens.get('coverage', 0):.0f}%" if ens.get("coverage") is not None else "N/A"
        print(f"  {res['scenario'][:27]:<28} "
              f"Rs.{ens.get('mae', 0):>8,.0f} "
              f"{ens.get('wape', 0):>6.1f}% "
              f"{ens.get('mase', 0):>5.3f} "
              f"{ens.get('mda', 0):>5.1f}% "
              f"{ens.get('r2', 0):>6.3f} "
              f"{cov:>8}")

    # --- Baseline Comparison ---
    print("\n" + "-" * 80)
    print("  ENSEMBLE vs BASELINES (MASE < 1.0 = Better than Naive)")
    print("-" * 80)
    beats_naive = 0
    total = 0
    for res in valid:
        ens = res["models"].get("Ensemble", {})
        if "error" in ens:
            continue
        total += 1
        mase = ens.get("mase", 2.0)
        status = "BETTER" if mase < 1.0 else "WORSE"
        emoji = "[+]" if mase < 1.0 else "[-]"
        if mase < 1.0:
            beats_naive += 1
        print(f"  {emoji} {res['scenario'][:35]:<36} MASE={mase:.3f}  ({status} than naive)")

    if total > 0:
        print(f"\n  Result: Ensemble beats naive in {beats_naive}/{total} scenarios ({beats_naive/total*100:.0f}%)")

    # --- Improvement Recommendations ---
    print("\n" + "=" * 80)
    print("  IMPROVEMENT RECOMMENDATIONS (Prioritized)")
    print("=" * 80)

    if improvements.get("strengths"):
        print("\n  [STRENGTHS]")
        for s in improvements["strengths"]:
            print(f"    [+] {s}")

    if improvements.get("critical"):
        print("\n  [CRITICAL ISSUES]")
        for s in improvements["critical"]:
            print(f"    [!!!] {s}")

    if improvements.get("high"):
        print("\n  [HIGH PRIORITY ISSUES]")
        for s in improvements["high"]:
            print(f"    [!!] {s}")

    if improvements.get("medium"):
        print("\n  [MEDIUM PRIORITY ISSUES]")
        for s in improvements["medium"][:5]:
            print(f"    [!] {s}")

    print("\n  [RECOMMENDED IMPROVEMENTS]")
    for imp in improvements.get("recommended_improvements", []):
        print(f"\n    [{imp['priority']}] {imp['area']}")
        print(f"      {imp['recommendation']}")
        print(f"      Expected impact: {imp['expected_impact']}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 7: MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def run_full_kaggle_evaluation():
    """
    Main entry point. Runs the complete evaluation suite:
    1. Generates 7 realistic scenarios
    2. Scans for any user-provided CSVs
    3. Evaluates all datasets
    4. Produces charts, JSON report, and improvement analysis
    """
    print("=" * 80)
    print("  BB+ CASH FLOW AI — REAL-WORLD EVALUATION SUITE")
    print("  Testing pipeline against 7 realistic business scenarios")
    print("=" * 80)

    all_results = []
    generators = [
        generate_scenario_1_small_business_india,
        generate_scenario_2_freelancer_irregular,
        generate_scenario_3_ecommerce_seasonal,
        generate_scenario_4_saas_subscription,
        generate_scenario_5_construction_lumpy,
        generate_scenario_6_restaurant_chain,
        generate_scenario_7_stress_test,
    ]

    # --- Phase 1: Synthetic Scenarios ---
    print("\n[PHASE 1] Evaluating on 7 Realistic Business Scenarios...")
    for gen_func in generators:
        try:
            df, name = gen_func(days=365)
            result = evaluate_single_dataset(df, name, horizon_days=30)
            all_results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] {gen_func.__name__}: {e}")
            traceback.print_exc()
            all_results.append(None)

    # --- Phase 2: User-provided CSVs ---
    print("\n\n[PHASE 2] Scanning for user-provided CSV files...")
    kaggle_csvs = scan_for_kaggle_csvs()
    if kaggle_csvs:
        print(f"  Found {len(kaggle_csvs)} CSV file(s):")
        for fname, fpath, size in kaggle_csvs:
            print(f"    - {fname} ({size:.1f} KB)")
            try:
                df = load_kaggle_csv(fpath)
                result = evaluate_single_dataset(df, f"User CSV: {fname}", horizon_days=30)
                all_results.append(result)
            except Exception as e:
                print(f"    [ERROR] Could not process {fname}: {e}")
                all_results.append(None)
    else:
        print("  No additional CSV files found.")
        print("  TIP: Drop any Kaggle CSV into the project directory to include it in evaluation.")

    # --- Phase 3: Analysis ---
    print("\n\n[PHASE 3] Analyzing results and generating recommendations...")
    improvements = analyze_improvements(all_results)

    # --- Phase 4: Visualization ---
    print("\n[PHASE 4] Generating evaluation dashboards...")
    generate_comprehensive_dashboard(all_results)

    # --- Phase 5: Report ---
    print_summary_report(all_results, improvements)

    # --- Phase 6: Save JSON ---
    json_results = {
        "timestamp": datetime.now().isoformat(),
        "scenarios_evaluated": len([r for r in all_results if r is not None]),
        "results": [],
        "improvements": improvements,
    }
    for res in all_results:
        if res is not None:
            # Remove large series data for JSON (keep it lightweight)
            res_copy = {k: v for k, v in res.items() if k != "series"}
            json_results["results"].append(res_copy)

    json_path = os.path.join(BASE_DIR, "kaggle_evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Full results: {json_path}")

    print("\n" + "=" * 80)
    print("  EVALUATION COMPLETE")
    print("=" * 80)

    return json_results


if __name__ == "__main__":
    run_full_kaggle_evaluation()
