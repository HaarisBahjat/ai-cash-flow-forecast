"""
Evaluation and Benchmarking Engine for BB+ AI Cash Flow Forecast
Author: BB+ (Business Brain+)

This module implements a comprehensive, domain-tailored evaluation suite for
daily cash flow forecasting with suitable regression, percentage, directional,
liquidity-balance, and uncertainty-coverage metrics.
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta

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

import app as main_app
import crm_synthetic

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  1. SUITABLE METRICS DEFINITIONS
# ─────────────────────────────────────────────────────────────

def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error in Rupees (₹)."""
    return float(np.mean(np.abs(y_true - y_pred)))

def root_mean_squared_error(y_true, y_pred):
    """Root Mean Squared Error in Rupees (₹). Penalizes large cash surprises heavily."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def median_absolute_error(y_true, y_pred):
    """Median Absolute Error in Rupees (₹). Robust to extreme outlier transaction days."""
    return float(np.median(np.abs(y_true - y_pred)))

def weighted_absolute_percentage_error(y_true, y_pred):
    """
    Weighted Absolute Percentage Error (WAPE %):
    WAPE = sum(|y - y_hat|) / sum(|y|) * 100%
    Essential for cash flow where net daily flow can be 0 or cross zero,
    preventing the division-by-zero catastrophe of standard MAPE.
    """
    tot_act = float(np.sum(np.abs(y_true)))
    if tot_act == 0:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / tot_act) * 100.0)

def symmetric_mape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE %):
    Bounded between 0% and 200%. Handles small values symmetrically.
    """
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-6
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """
    Mean Absolute Scaled Error (MASE):
    MASE = MAE / (in-sample 1-step naive MAE)
    Hyndman & Koehler (2006). MASE < 1.0 proves model is statistically
    superior to naive persistence. Scale-free and robust.
    """
    mae_model = np.mean(np.abs(y_true - y_pred))
    naive_in_sample_mae = np.mean(np.abs(np.diff(y_train)))
    if naive_in_sample_mae == 0:
        return 1.0
    return float(mae_model / naive_in_sample_mae)

def mean_directional_accuracy(y_true, y_pred):
    """
    Mean Directional Accuracy (MDA %):
    Percentage of days where the sign/direction of cash flow (Inflow > 0 vs Outflow < 0)
    was correctly identified.
    """
    signs_match = (np.sign(y_true) == np.sign(y_pred))
    return float(np.mean(signs_match) * 100.0)

def r_squared_score(y_true, y_pred):
    """Coefficient of Determination (R²): Fraction of net flow variance explained."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

def calculate_liquidity_balance_metrics(y_true, y_pred, opening_balance=620000.0):
    """
    Evaluates the Cumulative Bank Balance Trajectory:
    - Cumulative Balance Path MAE: Average rupee deviation across the 30-day balance curve.
    - Final Balance Error: Net liquidity deviation at Day 30.
    - Liquidity Dip (Minimum Balance) Error: Rupee difference between actual and predicted minimum balance.
    - Dip Timing Error: Deviation in days between predicted crunch date and actual crunch date.
    """
    act_balance = opening_balance + np.cumsum(y_true)
    pred_balance = opening_balance + np.cumsum(y_pred)

    cum_bal_mae = float(np.mean(np.abs(act_balance - pred_balance)))
    final_bal_err = float(np.abs(act_balance[-1] - pred_balance[-1]))

    act_min_bal = float(np.min(act_balance))
    pred_min_bal = float(np.min(pred_balance))
    dip_bal_err = float(np.abs(act_min_bal - pred_min_bal))

    act_min_idx = int(np.argmin(act_balance))
    pred_min_idx = int(np.argmin(pred_balance))
    dip_days_err = int(abs(act_min_idx - pred_min_idx))

    return {
        "cum_balance_mae": cum_bal_mae,
        "final_balance_error": final_bal_err,
        "dip_balance_error": dip_bal_err,
        "dip_timing_error_days": dip_days_err,
        "actual_min_balance": act_min_bal,
        "predicted_min_balance": pred_min_bal,
        "actual_dip_day": act_min_idx + 1,
        "predicted_dip_day": pred_min_idx + 1,
        "actual_balance_trajectory": act_balance.tolist(),
        "predicted_balance_trajectory": pred_balance.tolist(),
    }

def calculate_prediction_interval_metrics(y_true, lower, upper):
    """
    Prediction Interval Coverage Probability (PICP) and Mean Interval Width (MPIW)
    for the nominal 80% confidence interval.
    """
    if lower is None or upper is None:
        return {"coverage_percent": None, "mean_interval_width": None}

    lower = np.asarray(lower)
    upper = np.asarray(upper)
    in_bounds = (y_true >= lower) & (y_true <= upper)
    coverage = float(np.mean(in_bounds) * 100.0)
    mpiw = float(np.mean(upper - lower))
    return {
        "coverage_percent": coverage,
        "mean_interval_width": mpiw,
    }

def evaluate_predictions(y_true, y_pred, y_train, lower=None, upper=None, opening_balance=620000.0):
    """Aggregates all suitable metrics for a single model's predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    wape = weighted_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mape(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred, y_train)
    mda = mean_directional_accuracy(y_true, y_pred)
    r2 = r_squared_score(y_true, y_pred)

    bal_metrics = calculate_liquidity_balance_metrics(y_true, y_pred, opening_balance)
    pi_metrics = calculate_prediction_interval_metrics(y_true, lower, upper)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "medae": round(medae, 2),
        "wape": round(wape, 2),
        "smape": round(smape, 2),
        "mase": round(mase, 3),
        "mda": round(mda, 2),
        "r2": round(r2, 3),
        "cum_balance_mae": round(bal_metrics["cum_balance_mae"], 2),
        "final_balance_error": round(bal_metrics["final_balance_error"], 2),
        "dip_balance_error": round(bal_metrics["dip_balance_error"], 2),
        "dip_timing_error_days": bal_metrics["dip_timing_error_days"],
        "actual_min_balance": bal_metrics["actual_min_balance"],
        "predicted_min_balance": bal_metrics["predicted_min_balance"],
        "coverage_percent": round(pi_metrics["coverage_percent"], 2) if pi_metrics["coverage_percent"] is not None else None,
        "mean_interval_width": round(pi_metrics["mean_interval_width"], 2) if pi_metrics["mean_interval_width"] is not None else None,
        "balance_trajectory": bal_metrics["predicted_balance_trajectory"],
        "actual_balance_trajectory": bal_metrics["actual_balance_trajectory"],
    }


# ─────────────────────────────────────────────────────────────
#  2. BASELINE AND MODEL FORECAST GENERATORS
# ─────────────────────────────────────────────────────────────

def generate_naive_persistence_forecast(y_train, horizon_days=30):
    """Predicts the last observed daily flow for all future days."""
    last_val = float(y_train.iloc[-1] if isinstance(y_train, pd.Series) else y_train[-1])
    preds = np.repeat(last_val, horizon_days)
    return preds, None, None

def generate_rolling_mean_forecast(y_train, horizon_days=30, window=14):
    """Predicts the rolling average of the last 14 days."""
    series = pd.Series(y_train)
    mean_val = float(series.iloc[-window:].mean())
    std_val = float(series.iloc[-window:].std() or 0.0)
    preds = np.repeat(mean_val, horizon_days)
    lower = preds - 1.28 * std_val
    upper = preds + 1.28 * std_val
    return preds, lower, upper

def generate_seasonal_naive_forecast(y_train, horizon_days=30, period=7):
    """Predicts the same day of the week from the previous week."""
    series = pd.Series(y_train).values
    last_period = series[-period:]
    repeats = int(np.ceil(horizon_days / period))
    preds = np.tile(last_period, repeats)[:horizon_days]
    return preds, None, None


# ─────────────────────────────────────────────────────────────
#  3. EVALUATION WORKFLOWS
# ─────────────────────────────────────────────────────────────

def run_holdout_evaluation(df=None, horizon_days=30, opening_balance=620000.0):
    """
    Protocol 1: 30-Day Holdout Test Horizon Evaluation.
    Takes the dataset, splits into Train (all prior days) and Test (last 30 days).
    Fits all baselines and ML models, produces side-by-side metric comparison.
    """
    if df is None:
        csv_path = os.path.join(main_app.BASE_DIR, "test_sample.csv")
        raw_df = pd.read_csv(csv_path)
        raw_df["ds"] = pd.to_datetime(raw_df["date"], dayfirst=True)
        daily, gap_ratio = main_app.prepare_daily_series(raw_df)
    else:
        daily, gap_ratio = main_app.prepare_daily_series(df)

    if len(daily) < horizon_days + 45:
        raise ValueError(f"Need at least {horizon_days + 45} daily records for evaluation.")

    train_daily = daily.iloc[:-horizon_days].copy()
    test_daily = daily.iloc[-horizon_days:].copy()

    y_train = train_daily["y"].values
    y_test = test_daily["y"].values
    dates_test = test_daily["ds"].dt.strftime("%Y-%m-%d").tolist()

    # Fit and forecast baselines
    naive_pred, _, _ = generate_naive_persistence_forecast(y_train, horizon_days)
    mean_pred, mean_lo, mean_hi = generate_rolling_mean_forecast(y_train, horizon_days, window=14)
    s_naive_pred, _, _ = generate_seasonal_naive_forecast(y_train, horizon_days, period=7)

    # Winsorize training data for ML models
    train_clean, outlier_info = main_app.winsorize_outliers(train_daily)

    # Fit Prophet
    prophet_df = main_app.run_prophet(train_clean, gap_ratio, horizon_days)
    p_pred = prophet_df["predicted"].values
    p_lo = prophet_df["lower"].values
    p_hi = prophet_df["upper"].values

    # Fit XGBoost
    xgb_df = main_app.run_xgboost(train_clean, horizon_days)
    x_pred = xgb_df["predicted"].values
    x_lo = xgb_df["lower"].values
    x_hi = xgb_df["upper"].values

    # Fit LightGBM
    lgb_df = main_app.run_lightgbm(train_clean, horizon_days)
    l_pred = lgb_df["predicted"].values
    l_lo = lgb_df["lower"].values
    l_hi = lgb_df["upper"].values

    # Baseline Fixed Blend Ensemble (55/45)
    xgb_weight = 0.55 if gap_ratio < 0.20 else 0.40
    prophet_weight = 1.0 - xgb_weight
    base_pred = prophet_weight * p_pred + xgb_weight * x_pred
    base_lo = prophet_weight * p_lo + xgb_weight * x_lo
    base_hi = prophet_weight * p_hi + xgb_weight * x_hi

    # Stacking Ensemble (Prophet + XGBoost + LightGBM)
    models_dict = {"Prophet": prophet_df, "XGBoost": xgb_df, "LightGBM": lgb_df}
    stack_weights = main_app.calculate_stacking_weights(train_clean, gap_ratio, horizon_days, models_dict)
    
    stack_pred = np.zeros(horizon_days)
    stack_lo = np.zeros(horizon_days)
    stack_hi = np.zeros(horizon_days)
    for name, res in models_dict.items():
        w = stack_weights[name]
        stack_pred += w * res["predicted"].values
        stack_lo += w * res["lower"].values
        stack_hi += w * res["upper"].values

    # Evaluate each
    results = {
        "metadata": {
            "total_days": len(daily),
            "train_days": len(train_daily),
            "test_days": horizon_days,
            "train_range": f"{train_daily['ds'].min().strftime('%Y-%m-%d')} to {train_daily['ds'].max().strftime('%Y-%m-%d')}",
            "test_range": f"{test_daily['ds'].min().strftime('%Y-%m-%d')} to {test_daily['ds'].max().strftime('%Y-%m-%d')}",
            "gap_ratio": round(gap_ratio, 3),
            "outliers_detected": outlier_info.get("count", 0),
            "opening_balance": opening_balance,
            "learned_weights": {k: round(v, 3) for k, v in stack_weights.items()},
        },
        "models": {
            "Naive (Persistence)": evaluate_predictions(y_test, naive_pred, y_train, opening_balance=opening_balance),
            "14d Rolling Mean": evaluate_predictions(y_test, mean_pred, y_train, mean_lo, mean_hi, opening_balance=opening_balance),
            "Seasonal Naive (7d)": evaluate_predictions(y_test, s_naive_pred, y_train, opening_balance=opening_balance),
            "Facebook Prophet": evaluate_predictions(y_test, p_pred, y_train, p_lo, p_hi, opening_balance=opening_balance),
            "XGBoost Regressor": evaluate_predictions(y_test, x_pred, y_train, x_lo, x_hi, opening_balance=opening_balance),
            "LightGBM Regressor": evaluate_predictions(y_test, l_pred, y_train, l_lo, l_hi, opening_balance=opening_balance),
            "Baseline Ensemble (Fixed 55/45)": evaluate_predictions(y_test, base_pred, y_train, base_lo, base_hi, opening_balance=opening_balance),
            "Stacking Ensemble (Prophet + XGBoost + LightGBM)": evaluate_predictions(y_test, stack_pred, y_train, stack_lo, stack_hi, opening_balance=opening_balance),
            "Ensemble (Prophet + XGBoost)": evaluate_predictions(y_test, stack_pred, y_train, stack_lo, stack_hi, opening_balance=opening_balance),
        },
        "series": {
            "dates": dates_test,
            "actual": [round(float(v), 2) for v in y_test],
            "naive": [round(float(v), 2) for v in naive_pred],
            "seasonal_naive": [round(float(v), 2) for v in s_naive_pred],
            "prophet": [round(float(v), 2) for v in p_pred],
            "xgboost": [round(float(v), 2) for v in x_pred],
            "lightgbm": [round(float(v), 2) for v in l_pred],
            "baseline_ensemble": [round(float(v), 2) for v in base_pred],
            "ensemble": [round(float(v), 2) for v in stack_pred],
            "ensemble_lower": [round(float(v), 2) for v in stack_lo],
            "ensemble_upper": [round(float(v), 2) for v in stack_hi],
        }
    }

    return results


def run_walk_forward_cv(df=None, horizon_days=30, n_splits=3, opening_balance=620000.0):
    """
    Protocol 2: Expanding Window Walk-Forward Time-Series Cross Validation (Backtesting).
    Evaluates model stability across multiple historical rolling windows.
    Returns mean and standard deviation for all suitable metrics.
    """
    if df is None:
        csv_path = os.path.join(main_app.BASE_DIR, "test_sample.csv")
        raw_df = pd.read_csv(csv_path)
        raw_df["ds"] = pd.to_datetime(raw_df["date"], dayfirst=True)
        daily, gap_ratio = main_app.prepare_daily_series(raw_df)
    else:
        daily, gap_ratio = main_app.prepare_daily_series(df)

    total_len = len(daily)
    min_train = 90
    step = (total_len - min_train - horizon_days) // (n_splits - 1) if n_splits > 1 else 0

    model_names = [
        "Naive (Persistence)",
        "Seasonal Naive (7d)",
        "Facebook Prophet",
        "XGBoost Regressor",
        "Ensemble (Prophet + XGBoost)"
    ]

    cv_folds_data = {m: [] for m in model_names}

    for fold_idx in range(n_splits):
        train_end_idx = min_train + fold_idx * step
        test_end_idx = train_end_idx + horizon_days
        if test_end_idx > total_len:
            test_end_idx = total_len
            train_end_idx = test_end_idx - horizon_days

        train_fold = daily.iloc[:train_end_idx].copy()
        test_fold = daily.iloc[train_end_idx:test_end_idx].copy()

        y_train = train_fold["y"].values
        y_test = test_fold["y"].values

        # Baselines
        naive_pred, _, _ = generate_naive_persistence_forecast(y_train, horizon_days)
        s_naive_pred, _, _ = generate_seasonal_naive_forecast(y_train, horizon_days, period=7)

        # ML models
        train_clean, _ = main_app.winsorize_outliers(train_fold)
        p_df = main_app.run_prophet(train_clean, gap_ratio, horizon_days)
        p_pred = p_df["predicted"].values
        p_lo = p_df["lower"].values
        p_hi = p_df["upper"].values

        x_df = main_app.run_xgboost(train_clean, horizon_days)
        x_pred = x_df["predicted"].values
        x_lo = x_df["lower"].values
        x_hi = x_df["upper"].values

        xgb_weight = 0.55 if gap_ratio < 0.20 else 0.40
        prophet_weight = 1.0 - xgb_weight
        ens_pred = prophet_weight * p_pred + xgb_weight * x_pred
        ens_lo = prophet_weight * p_lo + xgb_weight * x_lo
        ens_hi = prophet_weight * p_hi + xgb_weight * x_hi

        cv_folds_data["Naive (Persistence)"].append(evaluate_predictions(y_test, naive_pred, y_train, opening_balance=opening_balance))
        cv_folds_data["Seasonal Naive (7d)"].append(evaluate_predictions(y_test, s_naive_pred, y_train, opening_balance=opening_balance))
        cv_folds_data["Facebook Prophet"].append(evaluate_predictions(y_test, p_pred, y_train, p_lo, p_hi, opening_balance=opening_balance))
        cv_folds_data["XGBoost Regressor"].append(evaluate_predictions(y_test, x_pred, y_train, x_lo, x_hi, opening_balance=opening_balance))
        cv_folds_data["Ensemble (Prophet + XGBoost)"].append(evaluate_predictions(y_test, ens_pred, y_train, ens_lo, ens_hi, opening_balance=opening_balance))

    # Aggregate Mean and Std
    cv_summary = {}
    for m in model_names:
        folds = cv_folds_data[m]
        metrics = ["mae", "rmse", "wape", "smape", "mase", "mda", "cum_balance_mae", "final_balance_error", "dip_timing_error_days"]
        cv_summary[m] = {}
        for met in metrics:
            vals = [f[met] for f in folds if f[met] is not None]
            cv_summary[m][f"{met}_mean"] = round(float(np.mean(vals)), 2)
            cv_summary[m][f"{met}_std"] = round(float(np.std(vals)), 2)

    return {
        "n_splits": n_splits,
        "cv_summary": cv_summary,
        "raw_folds": cv_folds_data,
    }


def run_crm_evaluation(horizon_days=30):
    """
    Protocol 3: Synthetic CRM Dataset Evaluation.
    Evaluates Statistical Baseline vs CRM Deterministic Forward Ledger augmentation.
    """
    base_date = crm_synthetic.get_base_date()
    # Generate 210 days: 180 for train, 30 for test
    full_hist = crm_synthetic.generate_crm_historical_transactions(180 + horizon_days, base_date)
    full_hist["ds"] = pd.to_datetime(full_hist["date"], format="%d-%m-%Y")
    daily, gap_ratio = main_app.prepare_daily_series(full_hist[["ds", "amount"]])

    train_daily = daily.iloc[:-horizon_days].copy()
    test_daily = daily.iloc[-horizon_days:].copy()

    y_train = train_daily["y"].values
    y_test = test_daily["y"].values
    opening_bal = crm_synthetic.CRM_INFO["current_bank_balance"]

    train_clean, _ = main_app.winsorize_outliers(train_daily)
    p_df = main_app.run_prophet(train_clean, gap_ratio, horizon_days)
    x_df = main_app.run_xgboost(train_clean, horizon_days)

    xgb_weight = 0.55 if gap_ratio < 0.20 else 0.40
    prophet_weight = 1.0 - xgb_weight
    ens_pred = prophet_weight * p_df["predicted"].values + xgb_weight * x_df["predicted"].values
    ens_lo = prophet_weight * p_df["lower"].values + xgb_weight * x_df["lower"].values
    ens_hi = prophet_weight * p_df["upper"].values + xgb_weight * x_df["upper"].values

    stat_eval = evaluate_predictions(y_test, ens_pred, y_train, ens_lo, ens_hi, opening_balance=opening_bal)

    return {
        "statistical_ensemble": stat_eval,
        "opening_balance": opening_bal,
        "crm_info": crm_synthetic.CRM_INFO,
    }


# ─────────────────────────────────────────────────────────────
#  4. VISUALIZATION DASHBOARD GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_evaluation_plots(holdout_results, output_path=None):
    """
    Creates a publication-quality 6-panel evaluation figure
    illustrating error metrics, cumulative trajectory, and uncertainty coverage.
    """
    if output_path is None:
        output_path = os.path.join(main_app.BASE_DIR, "evaluation_dashboard.png")

    series = holdout_results["series"]
    models = holdout_results["models"]
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in series["dates"]]

    actual = np.array(series["actual"])
    prophet = np.array(series["prophet"])
    xgboost = np.array(series["xgboost"])
    ensemble = np.array(series["ensemble"])
    ens_lo = np.array(series["ensemble_lower"])
    ens_hi = np.array(series["ensemble_upper"])
    s_naive = np.array(series["seasonal_naive"])

    opening_bal = holdout_results["metadata"]["opening_balance"]
    act_balance = opening_bal + np.cumsum(actual)
    p_balance = opening_bal + np.cumsum(prophet)
    x_balance = opening_bal + np.cumsum(xgboost)
    ens_balance = opening_bal + np.cumsum(ensemble)
    sn_balance = opening_bal + np.cumsum(s_naive)

    # Setup 3x2 dark/modern grid
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    accent_blue = "#38bdf8"
    accent_purple = "#a855f7"
    accent_green = "#22c55e"
    accent_red = "#ef4444"
    accent_amber = "#f59e0b"
    grid_color = "#1e293b"

    for ax in axes.flat:
        ax.set_facecolor("#111827")
        ax.grid(True, linestyle="--", alpha=0.35, color=grid_color)
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#1f2937")

    # ──── Panel 1: Actual vs Predictions Daily Flow ────
    ax1 = axes[0, 0]
    ax1.plot(dates, actual / 1e3, label="Actual Flow", color="#ffffff", lw=2.2, marker="o", markersize=3.5, zorder=5)
    ax1.plot(dates, ensemble / 1e3, label="Ensemble Forecast", color=accent_blue, lw=2.0, zorder=4)
    ax1.plot(dates, xgboost / 1e3, label="XGBoost", color=accent_green, lw=1.2, linestyle="--", alpha=0.75)
    ax1.plot(dates, prophet / 1e3, label="Prophet", color=accent_purple, lw=1.2, linestyle=":", alpha=0.75)
    ax1.set_title("1. Daily Net Cash Flow: Actual vs Forecasts (₹ Thousands)", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax1.set_ylabel("Net Flow (₹K)", color="#94a3b8", fontsize=10)
    ax1.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    # ──── Panel 2: Cumulative Bank Balance Trajectory ────
    ax2 = axes[0, 1]
    ax2.plot(dates, act_balance / 1e5, label="Actual Bank Balance", color="#ffffff", lw=2.5, marker="s", markersize=3, zorder=5)
    ax2.plot(dates, ens_balance / 1e5, label="Ensemble Trajectory", color=accent_blue, lw=2.2, zorder=4)
    ax2.plot(dates, p_balance / 1e5, label="Prophet Trajectory", color=accent_purple, lw=1.4, linestyle=":", alpha=0.8)
    ax2.plot(dates, x_balance / 1e5, label="XGBoost Trajectory", color=accent_green, lw=1.4, linestyle="--", alpha=0.8)
    ax2.plot(dates, sn_balance / 1e5, label="Seasonal Naive", color=accent_amber, lw=1.2, linestyle="-.", alpha=0.6)

    # Highlight minimum dip
    min_idx = np.argmin(act_balance)
    ax2.scatter([dates[min_idx]], [act_balance[min_idx] / 1e5], color=accent_red, s=80, zorder=6, label=f"Actual Dip: ₹{act_balance[min_idx]/1e5:.1f}L")

    ax2.set_title("2. Cumulative Bank Balance Trajectory (₹ Lakhs)", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax2.set_ylabel("Bank Balance (₹ Lakhs)", color="#94a3b8", fontsize=10)
    ax2.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    # ──── Panel 3: Prediction Interval & Coverage ────
    ax3 = axes[1, 0]
    cov_pct = models["Ensemble (Prophet + XGBoost)"]["coverage_percent"]
    ax3.fill_between(dates, ens_lo / 1e3, ens_hi / 1e3, color=accent_blue, alpha=0.22, label=f"80% Confidence Band (Cov: {cov_pct}%)")
    ax3.plot(dates, ensemble / 1e3, color=accent_blue, lw=1.8, label="Ensemble Median")
    in_bounds = (actual >= ens_lo) & (actual <= ens_hi)
    ax3.scatter([d for d, ib in zip(dates, in_bounds) if ib], [a / 1e3 for a, ib in zip(actual, in_bounds) if ib], color=accent_green, s=28, label="Actuals Inside CI", zorder=5)
    ax3.scatter([d for d, ib in zip(dates, in_bounds) if not ib], [a / 1e3 for a, ib in zip(actual, in_bounds) if not ib], color=accent_red, s=35, marker="x", label="Actuals Outside CI", zorder=5)
    ax3.set_title("3. Ensemble 80% Confidence Interval & Coverage (₹K)", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax3.set_ylabel("Net Flow (₹K)", color="#94a3b8", fontsize=10)
    ax3.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    # ──── Panel 4: Residuals Error Distribution ────
    ax4 = axes[1, 1]
    res_prophet = (actual - prophet) / 1e3
    res_xgb = (actual - xgboost) / 1e3
    res_ens = (actual - ensemble) / 1e3

    bins = np.linspace(-60, 60, 20)
    ax4.hist(res_prophet, bins=bins, alpha=0.45, color=accent_purple, label="Prophet Residuals", density=True)
    ax4.hist(res_xgb, bins=bins, alpha=0.45, color=accent_green, label="XGBoost Residuals", density=True)
    ax4.hist(res_ens, bins=bins, alpha=0.6, color=accent_blue, label="Ensemble Residuals", density=True)
    ax4.axvline(0, color="#f8fafc", linestyle="--", lw=1.2, alpha=0.7)
    ax4.set_title("4. Daily Error Residual Distribution (Actual − Predicted, ₹K)", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax4.set_xlabel("Error (₹ Thousands)", color="#94a3b8", fontsize=10)
    ax4.set_ylabel("Density", color="#94a3b8", fontsize=10)
    ax4.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", fontsize=9)

    # ──── Panel 5: Key Error Metrics Benchmark (MAE, RMSE, WAPE) ────
    ax5 = axes[2, 0]
    bench_models = ["Naive", "14d Mean", "Seas. Naive", "Prophet", "XGBoost", "Ensemble"]
    mae_vals = [
        models["Naive (Persistence)"]["mae"] / 1e3,
        models["14d Rolling Mean"]["mae"] / 1e3,
        models["Seasonal Naive (7d)"]["mae"] / 1e3,
        models["Facebook Prophet"]["mae"] / 1e3,
        models["XGBoost Regressor"]["mae"] / 1e3,
        models["Ensemble (Prophet + XGBoost)"]["mae"] / 1e3,
    ]
    wape_vals = [
        models["Naive (Persistence)"]["wape"],
        models["14d Rolling Mean"]["wape"],
        models["Seasonal Naive (7d)"]["wape"],
        models["Facebook Prophet"]["wape"],
        models["XGBoost Regressor"]["wape"],
        models["Ensemble (Prophet + XGBoost)"]["wape"],
    ]

    x_idx = np.arange(len(bench_models))
    w = 0.38
    bars1 = ax5.bar(x_idx - w/2, mae_vals, width=w, color=accent_purple, alpha=0.85, label="MAE (₹K)")
    bars2 = ax5.bar(x_idx + w/2, wape_vals, width=w, color=accent_blue, alpha=0.85, label="WAPE (%)")
    ax5.set_xticks(x_idx)
    ax5.set_xticklabels(bench_models, fontsize=9, color="#94a3b8")
    ax5.set_title("5. Model Benchmark: Mean Absolute Error (₹K) & WAPE (%)", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax5.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", fontsize=9)
    for b in bars1:
        y = b.get_height()
        ax5.annotate(f"{y:.1f}k", (b.get_x() + b.get_width()/2, y), ha="center", va="bottom", color="#cbd5e1", fontsize=8)

    # ──── Panel 6: Directional Accuracy & Cumulative Balance MAE ────
    ax6 = axes[2, 1]
    mda_vals = [
        models["Naive (Persistence)"]["mda"],
        models["14d Rolling Mean"]["mda"],
        models["Seasonal Naive (7d)"]["mda"],
        models["Facebook Prophet"]["mda"],
        models["XGBoost Regressor"]["mda"],
        models["Ensemble (Prophet + XGBoost)"]["mda"],
    ]
    bal_mae_vals = [
        models["Naive (Persistence)"]["cum_balance_mae"] / 1e5,
        models["14d Rolling Mean"]["cum_balance_mae"] / 1e5,
        models["Seasonal Naive (7d)"]["cum_balance_mae"] / 1e5,
        models["Facebook Prophet"]["cum_balance_mae"] / 1e5,
        models["XGBoost Regressor"]["cum_balance_mae"] / 1e5,
        models["Ensemble (Prophet + XGBoost)"]["cum_balance_mae"] / 1e5,
    ]

    bars3 = ax6.bar(x_idx - w/2, mda_vals, width=w, color=accent_green, alpha=0.85, label="Directional Accuracy (MDA %)")
    bars4 = ax6.bar(x_idx + w/2, [b * 10 for b in bal_mae_vals], width=w, color=accent_amber, alpha=0.85, label="Cum. Balance MAE (x10k ₹)")
    ax6.set_xticks(x_idx)
    ax6.set_xticklabels(bench_models, fontsize=9, color="#94a3b8")
    ax6.set_title("6. Decision & Liquidity Reliability: MDA (%) vs Balance Error", color="#f8fafc", fontsize=12, fontweight="600", pad=10)
    ax6.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", fontsize=9)
    for b in bars3:
        y = b.get_height()
        ax6.annotate(f"{y:.0f}%", (b.get_x() + b.get_width()/2, y), ha="center", va="bottom", color="#cbd5e1", fontsize=8)

    fig.suptitle("BB+ Cash Flow AI — Rigorous Model Evaluation & Benchmarks", color="#f8fafc", fontsize=16, fontweight="700", y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[EVAL] Dashboard chart successfully saved to {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────
#  5. FORMATTED REPORT PRINTER & JSON EXPORTER
# ─────────────────────────────────────────────────────────────

def format_comparison_table(models_dict):
    """Generates a clean text / markdown comparison table of all models."""
    headers = ["Model", "MAE (₹)", "RMSE (₹)", "WAPE %", "sMAPE %", "MASE", "MDA %", "R²", "CumBal MAE", "Dip Timing"]
    rows = []
    for name, m in models_dict.items():
        rows.append([
            name,
            f"Rs.{m['mae']:,.0f}",
            f"Rs.{m['rmse']:,.0f}",
            f"{m['wape']:.1f}%",
            f"{m['smape']:.1f}%",
            f"{m['mase']:.3f}",
            f"{m['mda']:.1f}%",
            f"{m['r2']:.3f}",
            f"Rs.{m['cum_balance_mae']:,.0f}",
            f"{m['dip_timing_error_days']} days",
        ])

    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_strs = [" | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) for r in rows]

    return "\n".join([header_str, sep_str] + row_strs)


def run_full_evaluation_suite(save_artifacts=True):
    """Executes the complete evaluation suite, saves JSON and charts, prints report."""
    print("=" * 80)
    print(" >> BB+ AI CASH FLOW FORECAST -- COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)

    # 1. 30-Day Holdout
    print("\n[STEP 1/3] Running 30-Day Holdout Evaluation on Statement Data...")
    holdout_res = run_holdout_evaluation()
    table_str = format_comparison_table(holdout_res["models"])
    print("\n" + table_str + "\n")

    # 2. Walk-Forward CV
    print("[STEP 2/3] Running 3-Fold Walk-Forward Time-Series Cross Validation...")
    cv_res = run_walk_forward_cv(n_splits=3)
    print("Walk-forward CV complete across 3 expanding windows.")

    # 3. CRM Dataset Evaluation
    print("\n[STEP 3/3] Running Evaluation on Synthetic CRM Profile...")
    crm_res = run_crm_evaluation()
    print("CRM Evaluation complete.")

    # 4. Generate Visualizations
    chart_path = os.path.join(main_app.BASE_DIR, "evaluation_dashboard.png")
    generate_evaluation_plots(holdout_res, chart_path)

    full_results = {
        "timestamp": datetime.now().isoformat(),
        "holdout_30d": holdout_res,
        "walk_forward_cv": cv_res,
        "crm_evaluation": crm_res,
    }

    if save_artifacts:
        json_path = os.path.join(main_app.BASE_DIR, "evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"[EVAL] Results exported to {json_path}")

    return full_results


if __name__ == "__main__":
    run_full_evaluation_suite()
