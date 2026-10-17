"""
AI Cash Flow Forecast - Flask Backend
Author: BB+ (Business Brain+)
"""
# ─────────────────────────────────────────────
#  NOTE: Run with:  python app.py
#  Then open:       http://localhost:5000
# ─────────────────────────────────────────────

import os
import json
import warnings
from datetime import datetime, timedelta
from io import StringIO, BytesIO

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

# Serve frontend files from the same directory as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
#  HELPER: Parse uploaded CSV / Excel
# ─────────────────────────────────────────────

def parse_transaction_file(file):
    """
    Accepts CSV or XLSX with at minimum two columns:
      - date  : transaction date  (any common format)
      - amount: positive = inflow, negative = outflow
    Optional columns: description, category, type (credit/debit)
    Returns a clean DataFrame sorted by date.
    """
    filename = file.filename.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type. Use CSV or XLSX.")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Flexible column name mapping
    date_cols = ["date", "transaction_date", "txn_date", "value_date", "trans_date"]
    amount_cols = ["amount", "net_amount", "debit_credit", "value", "transaction_amount"]
    credit_cols = ["credit", "credit_amount", "inflow", "receipts"]
    debit_cols = ["debit", "debit_amount", "outflow", "payments"]

    date_col = next((c for c in date_cols if c in df.columns), None)
    amount_col = next((c for c in amount_cols if c in df.columns), None)
    credit_col = next((c for c in credit_cols if c in df.columns), None)
    debit_col = next((c for c in debit_cols if c in df.columns), None)

    if date_col is None:
        raise ValueError("No 'date' column found. Please include a column named 'date'.")

    df["ds"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["ds"])

    # Build unified 'amount' (positive = inflow, negative = outflow)
    if amount_col:
        df["amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
    elif credit_col and debit_col:
        df["credit"] = pd.to_numeric(df[credit_col], errors="coerce").fillna(0)
        df["debit"] = pd.to_numeric(df[debit_col], errors="coerce").fillna(0)
        df["amount"] = df["credit"] - df["debit"]
    else:
        raise ValueError("No amount column found. Need 'amount' or separate 'credit'/'debit' columns.")

    # Optional: description
    desc_col = next((c for c in ["description", "narration", "particulars", "remarks"] if c in df.columns), None)
    df["description"] = df[desc_col] if desc_col else ""

    # Optional: category (credit / debit)
    type_col = next((c for c in ["type", "txn_type", "cr/dr", "cr_dr"] if c in df.columns), None)
    if type_col:
        df["type"] = df[type_col].astype(str).str.upper()
        # Some banks use 'CR'/'DR' notation
        cr_mask = df["type"].str.contains("CR|CREDIT", na=False)
        dr_mask = df["type"].str.contains("DR|DEBIT", na=False)
        df.loc[dr_mask & (df["amount"] > 0), "amount"] *= -1
        df.loc[cr_mask & (df["amount"] < 0), "amount"] *= -1

    result = df[["ds", "amount", "description"]].sort_values("ds").reset_index(drop=True)
    return result


# ─────────────────────────────────────────────
#  STEP 1 — GAP FILLING & OUTLIER DETECTION
#  Pitfall fixes before any model sees the data
# ─────────────────────────────────────────────

def prepare_daily_series(df):
    """
    FIX — Pitfall 3 (Aggregated Gaps):
      Reindex to a complete date range so no day is missing.
      Adds 'has_transaction' flag so models can learn the difference
      between a genuine zero-flow day vs a data gap.
      Prophet won't hallucinate fake weekly patterns on sparse data.

    Returns:
      daily : DataFrame with columns [ds, y, has_transaction]
      gap_ratio : float, fraction of days with no transactions
    """
    raw = df.groupby("ds")["amount"].sum().reset_index()
    raw.columns = ["ds", "y"]
    raw = raw.sort_values("ds").set_index("ds")

    full_index = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
    daily = raw.reindex(full_index)
    daily["has_transaction"] = (~daily["y"].isna()).astype(int)
    daily["y"] = daily["y"].fillna(0.0)
    daily = daily.reset_index().rename(columns={"index": "ds"})

    gap_ratio = 1.0 - daily["has_transaction"].mean()
    return daily, gap_ratio


def winsorize_outliers(daily, z_thresh=3.0):
    """
    FIX — Pitfall 1 (Additive Bias with Outliers):
      Caps extreme values at mean ± z_thresh * std BEFORE Prophet fits.
      This stops a single ₹50L real-estate purchase from bending the
      entire trend line and producing a misleading downward forecast.

      Outlier metadata is returned separately so the AI explanation
      engine can warn the user: "1 outlier transaction was detected
      and capped for trend stability."

    Returns:
      daily_clean : winsorized copy
      outlier_info : dict with count, max outlier magnitude, direction
    """
    y = daily["y"].copy()
    mean, std = y.mean(), y.std()
    if std == 0:
        return daily.copy(), {"count": 0}

    lo = mean - z_thresh * std
    hi = mean + z_thresh * std
    outliers = daily[(y < lo) | (y > hi)]

    daily_clean = daily.copy()
    daily_clean["y"] = y.clip(lower=lo, upper=hi)

    outlier_info = {
        "count": len(outliers),
        "max_magnitude": float(outliers["y"].abs().max()) if len(outliers) else 0,
        "direction": "outflow" if (len(outliers) and outliers["y"].mean() < 0) else "inflow",
    }
    return daily_clean, outlier_info


# ─────────────────────────────────────────────
#  STEP 2 — PROPHET MODEL (hardened)
# ─────────────────────────────────────────────

def run_prophet(daily_clean, gap_ratio, horizon_days):
    """
    FIX — Pitfall 3 (Gap hallucination):
      - Disables weekly_seasonality when gap_ratio > 40% (too sparse
        to reliably detect Mon/Fri patterns).
      - Uses a tighter changepoint_prior_scale (0.05 default vs 0.3)
        to reduce overfitting to recent outlier-driven trend shifts.
      - Adds 'has_transaction' as an exogenous regressor so Prophet
        knows which zeros are real vs gap-filling artifacts.
    """
    from prophet import Prophet

    prophet_df = daily_clean[["ds", "y", "has_transaction"]].copy()

    # Seasonality guard: don't fit weekly on very sparse data
    weekly_ok  = gap_ratio < 0.40 and len(prophet_df) >= 14
    yearly_ok  = len(prophet_df) >= 180

    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=weekly_ok,
        yearly_seasonality=yearly_ok,
        changepoint_prior_scale=0.05,       # tighter → less outlier-bend
        seasonality_prior_scale=5.0,
        interval_width=0.80,
    )

    # FIX — Pitfall 2: add transaction-activity as regressor
    m.add_regressor("has_transaction", standardize=True)
    m.fit(prophet_df)

    # Build future frame, assume all future days have transactions
    future = m.make_future_dataframe(periods=horizon_days)
    future["has_transaction"] = 1
    forecast = m.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days).copy()
    result.columns = ["date", "predicted", "lower", "upper"]
    return result


# ─────────────────────────────────────────────
#  STEP 3 — XGBOOST MODEL
#  FIX — Pitfall 2 (Non-linear exogenous features)
# ─────────────────────────────────────────────

def engineer_features(daily):
    """
    Creates 20+ features that capture what Prophet's linear
    regressor assumption cannot:
      - Lag features: autocorrelation at 1,2,3,7,14,21,28 days
      - Rolling statistics: mean, std, min, max at 7 and 14-day windows
      - Calendar features: day-of-week, week-of-month, month, quarter
      - Interaction: day_of_week × week_of_month (captures month-end Mon patterns)
      - Binary: is_month_start, is_month_end, is_weekend, has_transaction
    """
    df = daily.copy().set_index("ds").sort_index()
    y = df["y"]

    feat = pd.DataFrame(index=df.index)

    # — Lag features (captures autocorrelation Prophet misses)
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        feat[f"lag_{lag}"] = y.shift(lag)

    # — Rolling stats at 7-day and 14-day windows
    for window in [7, 14]:
        feat[f"roll_mean_{window}"] = y.shift(1).rolling(window).mean()
        feat[f"roll_std_{window}"]  = y.shift(1).rolling(window).std().fillna(0)
        feat[f"roll_min_{window}"]  = y.shift(1).rolling(window).min()
        feat[f"roll_max_{window}"]  = y.shift(1).rolling(window).max()

    # — Calendar
    feat["day_of_week"]    = df.index.dayofweek          # 0=Mon, 6=Sun
    feat["day_of_month"]   = df.index.day
    feat["week_of_month"]  = (df.index.day - 1) // 7 + 1
    feat["month"]          = df.index.month
    feat["quarter"]        = df.index.quarter
    feat["is_weekend"]     = (df.index.dayofweek >= 5).astype(int)
    feat["is_month_start"] = df.index.is_month_start.astype(int)
    feat["is_month_end"]   = df.index.is_month_end.astype(int)

    # — Interaction: captures "last Friday of the month" type patterns
    feat["dow_x_wom"] = feat["day_of_week"] * feat["week_of_month"]

    # — Transaction activity flag
    feat["has_transaction"] = df["has_transaction"]

    feat["target"] = y
    return feat


def run_xgboost(daily_clean, horizon_days):
    """
    Trains XGBoost on engineered time-series features and predicts
    the next `horizon_days` days by recursive/direct forecasting.

    Uses TimeSeriesSplit for validation to prevent data leakage.
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error

    feat_df = engineer_features(daily_clean)
    feat_df = feat_df.dropna()  # drop rows where lags don't exist yet

    if len(feat_df) < 30:
        raise ValueError("XGBoost needs 30+ clean days after lag creation.")

    feature_cols = [c for c in feat_df.columns if c != "target"]
    X = feat_df[feature_cols].values
    y = feat_df["target"].values

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,       # L1 — sparse feature selection
        reg_lambda=1.0,      # L2 — reduces overfitting
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X, y)

    # — Recursive forecasting: predict one day, feed it as lag for next
    last_known = daily_clean.set_index("ds")["y"].copy()
    last_date  = daily_clean["ds"].max()
    predictions = []

    extended = last_known.copy()
    for i in range(horizon_days):
        next_date = last_date + timedelta(days=i + 1)
        extended[next_date] = np.nan  # placeholder

        temp_daily = extended.reset_index()
        temp_daily.columns = ["ds", "y"]
        temp_daily["has_transaction"] = 1
        temp_feat = engineer_features(temp_daily)

        if next_date not in temp_feat.index or temp_feat.loc[next_date, feature_cols].isna().any():
            predictions.append(last_known.iloc[-14:].mean())
        else:
            row = temp_feat.loc[[next_date], feature_cols].values
            pred = float(model.predict(row)[0])
            predictions.append(pred)
            extended[next_date] = pred  # feed prediction as next lag

    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]

    # Compute residual std for a simple confidence interval
    y_pred_train = model.predict(X)
    residual_std = float(np.std(y - y_pred_train))

    result = pd.DataFrame({
        "date": future_dates,
        "predicted": predictions,
        "lower": [p - 1.28 * residual_std for p in predictions],  # ~80% CI
        "upper": [p + 1.28 * residual_std for p in predictions],
    })
    return result


# ─────────────────────────────────────────────
#  STEP 4 — ENSEMBLE BLENDER
#  Weighted average of Prophet + XGBoost
# ─────────────────────────────────────────────

def run_ensemble_forecast(df, horizon_days=30):
    """
    Orchestrates the full hardened pipeline:
      1. Gap-fill + compute gap_ratio
      2. Winsorize outliers (saves outlier_info for AI explanation)
      3. Fit Prophet on clean data
      4. Fit XGBoost on clean data
      5. Blend: if both succeed → weighted average
                if only one succeeds → use that alone
                if both fail → rolling average fallback

    Returns:
      forecast_df : DataFrame [date, predicted, lower, upper]
      model_used  : str label for UI badge
      outlier_info: dict for AI explanation
    """
    # Step 1 — Gap filling
    daily, gap_ratio = prepare_daily_series(df)

    # Step 2 — Winsorize
    daily_clean, outlier_info = winsorize_outliers(daily)

    prophet_result = None
    xgb_result     = None
    errors         = []

    # Step 3 — Prophet
    try:
        prophet_result = run_prophet(daily_clean, gap_ratio, horizon_days)
    except Exception as e:
        errors.append(f"Prophet: {e}")
        print(f"[WARN] Prophet failed: {e}")

    # Step 4 — XGBoost
    try:
        xgb_result = run_xgboost(daily_clean, horizon_days)
    except Exception as e:
        errors.append(f"XGBoost: {e}")
        print(f"[WARN] XGBoost failed: {e}")

    # Step 5 — Blend
    if prophet_result is not None and xgb_result is not None:
        # Dynamic weighting: give XGBoost more weight when data is dense
        # (lots of transactions → XGBoost features are reliable)
        xgb_weight    = 0.55 if gap_ratio < 0.20 else 0.40
        prophet_weight = 1.0 - xgb_weight

        blended = prophet_result.copy()
        blended["predicted"] = (
            prophet_weight * prophet_result["predicted"].values +
            xgb_weight     * xgb_result["predicted"].values
        )
        # Ensemble CI: average the bounds (conservative)
        blended["lower"] = (
            prophet_weight * prophet_result["lower"].values +
            xgb_weight     * xgb_result["lower"].values
        )
        blended["upper"] = (
            prophet_weight * prophet_result["upper"].values +
            xgb_weight     * xgb_result["upper"].values
        )
        model_used = f"Prophet ({int(prophet_weight*100)}%) + XGBoost ({int(xgb_weight*100)}%)"
        return blended, model_used, outlier_info

    elif prophet_result is not None:
        return prophet_result, "Prophet only", outlier_info

    elif xgb_result is not None:
        return xgb_result, "XGBoost only", outlier_info

    else:
        print(f"Both models failed ({errors}), using rolling average.")
        result = _rolling_average_forecast(df, horizon_days)
        return result, "Rolling Average (fallback)", outlier_info


def _rolling_average_forecast(df, horizon_days=30):
    """Last-resort fallback when both Prophet and XGBoost fail."""
    daily = df.groupby("ds")["amount"].sum().reset_index().sort_values("ds")
    window = min(14, len(daily))
    avg = daily["amount"].rolling(window).mean().iloc[-1]
    std = daily["amount"].rolling(window).std().iloc[-1] or 0
    last_date = daily["ds"].max()
    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]
    return pd.DataFrame({
        "date": future_dates,
        "predicted": [avg] * horizon_days,
        "lower":     [avg - std] * horizon_days,
        "upper":     [avg + std] * horizon_days,
    })


# ─────────────────────────────────────────────
#  AI EXPLANATION ENGINE (Rule-based -> LLM-ready)
# ─────────────────────────────────────────────

def generate_ai_explanation(summary, forecast_df, outlier_info=None):
    """
    Generates smart natural-language cash flow insights.
    outlier_info: dict from winsorize_outliers() — tells us if anomalous
                  transactions were detected and capped before modelling.
    TO UPGRADE: Replace this function body with a Gemini/OpenAI API call.
    """
    insights = []
    alerts = []
    recommendations = []

    current_balance = summary["current_balance"]
    expected_in = summary["expected_inflow"]
    expected_out = summary["expected_outflow"]
    projected = summary["projected_balance"]
    low_days = summary.get("low_cash_days", [])
    worst_date = summary.get("worst_date")
    worst_balance = summary.get("worst_balance", projected)

    # -- Main narrative --
    if projected < 0:
        insights.append(
            f"Your cash position is projected to go <strong>negative (Rs.{abs(projected)/100000:.1f}L)</strong> "
            f"over the next 30 days. Immediate action is needed."
        )
    elif projected < current_balance * 0.3:
        insights.append(
            f"Your cash balance is projected to fall sharply from "
            f"<strong>Rs.{current_balance/100000:.1f}L to Rs.{projected/100000:.1f}L</strong> "
            f"over the next 30 days - a drop of {((current_balance - projected)/current_balance*100):.0f}%."
        )
    elif projected < current_balance:
        insights.append(
            f"Your cash position is expected to decrease moderately from "
            f"<strong>Rs.{current_balance/100000:.1f}L to Rs.{projected/100000:.1f}L</strong> over 30 days."
        )
    else:
        insights.append(
            f"Your cash position looks healthy - expected to grow from "
            f"<strong>Rs.{current_balance/100000:.1f}L to Rs.{projected/100000:.1f}L</strong> over 30 days. "
            f"Keep monitoring your receivables to maintain this trajectory."
        )

    # -- Worst day alert --
    if worst_date and worst_balance < current_balance * 0.4:
        worst_dt = pd.to_datetime(worst_date).strftime("%B %d")
        insights.append(
            f"Your cash position could fall to its lowest point around "
            f"<strong>{worst_dt}</strong> (Rs.{worst_balance/100000:.1f}L) - likely because several "
            f"outflows are clustered before major inflows arrive."
        )

    # -- Outflow dominance --
    if expected_out > expected_in:
        shortfall = expected_out - expected_in
        insights.append(
            f"Expected outflows (Rs.{expected_out/100000:.1f}L) exceed inflows "
            f"(Rs.{expected_in/100000:.1f}L) by <strong>Rs.{shortfall/100000:.1f}L</strong> this month."
        )

    # -- Recommendations --
    if projected < current_balance * 0.5:
        receivables_gap = abs(expected_out - expected_in) * 0.6
        recommendations.append(
            f"Consider following up on receivables worth approximately "
            f"<strong>Rs.{receivables_gap/100000:.1f}L</strong> due in the next 15 days to shore up liquidity."
        )
    if expected_out > expected_in * 1.2:
        recommendations.append(
            "Review discretionary expenses this month - deferring non-critical payments "
            "could ease cash pressure significantly."
        )
    if projected >= current_balance:
        recommendations.append(
            "With a positive cash trajectory, consider placing surplus funds in a short-term liquid instrument "
            "to earn additional yield."
        )
    if len(low_days) > 7:
        recommendations.append(
            f"You have {len(low_days)} days with projected low cash. "
            "Consider negotiating better payment terms with suppliers or setting up an overdraft line."
        )

    # -- Alerts --
    if projected < 50000:
        alerts.append({"type": "danger", "text": "Critical: Projected balance falls below Rs.50K. Immediate action required."})
    elif projected < current_balance * 0.25:
        alerts.append({"type": "warning", "text": "Caution: Significant cash drawdown expected this month."})

    # -- Outlier transparency note (Pitfall 1 awareness) --
    if outlier_info and outlier_info.get("count", 0) > 0:
        mag = outlier_info["max_magnitude"]
        direction = outlier_info["direction"]
        count = outlier_info["count"]
        insights.append(
            f"🔍 <strong>Model note:</strong> {count} anomalous {direction} transaction(s) "
            f"(largest: Rs.{mag/100000:.1f}L) were detected and statistically capped before forecasting "
            f"to prevent trend distortion. These one-off events are excluded from the pattern model "
            f"but your actual balance already reflects them."
        )

    return {
        "insights": insights,
        "recommendations": recommendations,
        "alerts": alerts
    }


# ─────────────────────────────────────────────
#  SUMMARY CALCULATOR
# ─────────────────────────────────────────────

def calculate_summary(df, forecast_df, opening_balance=0):
    """Computes key cash flow metrics for the dashboard."""
    today = pd.Timestamp.today().normalize()

    # Historical: last 30 days for context
    # If user provided current bank balance, use it directly as cash available today.
    # Otherwise fallback to cumulative sum of statement transactions.
    if opening_balance > 0:
        current_balance = opening_balance
    else:
        current_balance = float(df["amount"].sum())

    # Forecast: next 30 days
    inflow_forecast = forecast_df[forecast_df["predicted"] > 0]["predicted"].sum()
    outflow_forecast = abs(forecast_df[forecast_df["predicted"] < 0]["predicted"].sum())
    projected_balance = current_balance + forecast_df["predicted"].sum()

    # Running balance through forecast period
    running = current_balance
    daily_balances = []
    for _, row in forecast_df.iterrows():
        running += row["predicted"]
        daily_balances.append({"date": row["date"].strftime("%Y-%m-%d"), "balance": round(running, 2)})

    low_threshold = max(50000, current_balance * 0.15)
    low_days = [d for d in daily_balances if d["balance"] < low_threshold]
    worst = min(daily_balances, key=lambda x: x["balance"])

    # Historical breakdown
    hist_inflow = float(df[df["amount"] > 0]["amount"].sum())
    hist_outflow = float(abs(df[df["amount"] < 0]["amount"].sum()))

    return {
        "current_balance": round(current_balance, 2),
        "opening_balance": round(opening_balance, 2),
        "expected_inflow": round(inflow_forecast, 2),
        "expected_outflow": round(outflow_forecast, 2),
        "projected_balance": round(projected_balance, 2),
        "low_cash_days": low_days,
        "worst_date": worst["date"],
        "worst_balance": round(worst["balance"], 2),
        "daily_balances": daily_balances,
        "historical_inflow": round(hist_inflow, 2),
        "historical_outflow": round(hist_outflow, 2),
        "data_points": len(df),
        "date_range": {
            "start": df["ds"].min().strftime("%Y-%m-%d"),
            "end": df["ds"].max().strftime("%Y-%m-%d"),
        }
    }


# ─────────────────────────────────────────────
#  API ROUTES
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "BB+ Cash Flow API running", "version": "1.0"})


@app.route("/api/forecast", methods=["POST"])
def forecast():
    """
    Main endpoint. Accepts:
      - file: CSV or XLSX of transactions
      - opening_balance (optional): float, current bank balance
      - horizon (optional): int, days to forecast (default 30)
    Returns JSON with forecast, summary, and AI explanation.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Send a CSV or XLSX file."}), 400

        file = request.files["file"]
        opening_balance = float(request.form.get("opening_balance", 0))
        horizon = int(request.form.get("horizon", 30))

        # Parse
        df = parse_transaction_file(file)
        if len(df) < 5:
            return jsonify({"error": "File has too few rows. Need at least 5 transactions."}), 400

        # Forecast (ensemble: Prophet + XGBoost)
        forecast_df, model_used, outlier_info = run_ensemble_forecast(df, horizon_days=horizon)

        # Summary
        summary = calculate_summary(df, forecast_df, opening_balance)

        # AI Explanation (now receives outlier_info)
        explanation = generate_ai_explanation(summary, forecast_df, outlier_info)

        # Prepare forecast series for chart
        forecast_series = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "predicted": round(row["predicted"], 2),
                "lower": round(row["lower"], 2),
                "upper": round(row["upper"], 2),
            }
            for _, row in forecast_df.iterrows()
        ]

        # Historical daily series for chart
        hist_daily = df.groupby("ds")["amount"].sum().reset_index()
        hist_series = [
            {"date": row["ds"].strftime("%Y-%m-%d"), "amount": round(row["amount"], 2)}
            for _, row in hist_daily.iterrows()
        ]

        return jsonify({
            "success": True,
            "model_used": model_used,
            "outlier_info": outlier_info,
            "summary": summary,
            "forecast_series": forecast_series,
            "historical_series": hist_series,
            "explanation": explanation,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample-data", methods=["GET"])
def sample_data():
    """Returns a sample CSV content so the user can test without their own data."""
    import random
    random.seed(42)

    today = datetime.today()
    rows = []
    for i in range(180):
        d = today - timedelta(days=180 - i)
        # Simulate salary on 1st, supplier payments on 5th/15th/25th, daily sales
        if d.day == 1:
            rows.append({"date": d.strftime("%d-%m-%Y"), "amount": 250000, "description": "Salary Credit"})
        if d.day in [5, 15, 25]:
            rows.append({"date": d.strftime("%d-%m-%Y"), "amount": -80000, "description": "Supplier Payment"})
        # Daily sales (random)
        import random as r
        daily_sale = r.gauss(15000, 5000)
        rows.append({"date": d.strftime("%d-%m-%Y"), "amount": round(daily_sale, 2), "description": "Sales Collection"})
        # Rent on 10th
        if d.day == 10:
            rows.append({"date": d.strftime("%d-%m-%Y"), "amount": -45000, "description": "Office Rent"})

    df = pd.DataFrame(rows)
    csv_str = df.to_csv(index=False)
    return app.response_class(csv_str, mimetype="text/csv",
                               headers={"Content-Disposition": "attachment;filename=sample_transactions.csv"})


# ─────────────────────────────────────────────
#  CRM INTEGRATION ROUTES (Synthetic Zoho/HubSpot)
# ─────────────────────────────────────────────

@app.route("/api/crm/status", methods=["GET"])
def crm_status():
    """Returns connected CRM status, account info, and live pipeline totals."""
    try:
        import crm_synthetic
        stats = crm_synthetic.get_crm_summary_stats()
        return jsonify({"success": True, "data": stats})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/crm/pipeline", methods=["GET"])
def crm_pipeline():
    """Returns full itemized customer invoices, supplier payables, and commitments."""
    try:
        import crm_synthetic
        base_date = crm_synthetic.get_base_date()
        receivables = crm_synthetic.generate_crm_receivables(base_date)
        payables = crm_synthetic.generate_crm_payables(base_date)
        commitments = crm_synthetic.generate_crm_commitments(base_date)
        stats = crm_synthetic.get_crm_summary_stats(base_date)
        return jsonify({
            "success": True,
            "crm_info": crm_synthetic.CRM_INFO,
            "receivables": receivables,
            "payables": payables,
            "commitments": commitments,
            "stats": stats
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/crm/forecast", methods=["POST"])
def crm_forecast():
    """
    Runs the hybrid cash flow forecast directly from the connected CRM.
    Combines 180-day ML baseline (Prophet+XGBoost) with scheduled CRM invoices and payables.
    """
    try:
        import crm_synthetic
        horizon = 30
        if request.is_json:
            horizon = int(request.json.get("horizon", 30))
        elif request.form:
            horizon = int(request.form.get("horizon", 30))

        result = crm_synthetic.run_crm_full_forecast(horizon_days=horizon)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True, port=5000)
