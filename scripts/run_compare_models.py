# scripts/run_compare_models.py

"""
Compare volatility forecasting models.

This script:
1. Loads processed SPY data
2. Runs all forecasting models
3. Aligns forecasts into one table
4. Computes model evaluation metrics
5. Saves outputs
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

from src.random_forest_model import (
    train_random_forest,
)

from src.xgboost_model import (
    train_xgboost,
)

from src.statistical_baseline import (
    statistical_volatility_forecast,
)

from src.cev_forecast import (
    rolling_cev_forecast,
)

# ---------------------------------------------------
# Project paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "SPY_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------
# Load processed data
# ---------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True,
)

# ===================================================
# Statistical Baseline
# ===================================================

baseline_df = statistical_volatility_forecast(df)

baseline_df = baseline_df[
    [
        "baseline_forecast",
        "future_rv20",
    ]
].copy()

baseline_df = baseline_df.rename(
    columns={
        "future_rv20": "actual_future_vol",
    }
)

# ===================================================
# Random Forest
# ===================================================

(
    rf_model,
    rf_rmse,
    rf_mae,
    rf_pred_df,
) = train_random_forest(df)

# ===================================================
# XGBoost
# ===================================================

(
    xgb_model,
    xgb_rmse,
    xgb_mae,
    xgb_pred_df,
) = train_xgboost(df)

# ===================================================
# CEV Forecast
# ===================================================

cev_df = rolling_cev_forecast(df)

# Attach canonical target
cev_df["actual_future_vol"] = (
    baseline_df["actual_future_vol"]
)

# Keep only required columns
cev_df = cev_df[
    [
        "cev_vol_forecast",
        "actual_future_vol",
    ]
].copy()

# Rename for consistency
cev_df = cev_df.rename(
    columns={
        "cev_vol_forecast": "cev_forecast",
    }
)

# ===================================================
# Merge all forecasts
# ===================================================

comparison_df = pd.concat(
    [
        baseline_df[
            [
                "baseline_forecast",
            ]
        ],

        cev_df[
            [
                "cev_forecast",
            ]
        ],

        rf_pred_df[
            [
                "rf_forecast",
            ]
        ],

        xgb_pred_df[
            [
                "xgb_forecast",
            ]
        ],

        rf_pred_df[
            [
                "actual_future_vol",
            ]
        ],
    ],
    axis=1,
)

comparison_df = comparison_df.dropna()

# ===================================================
# Metrics
# ===================================================

actual = comparison_df["actual_future_vol"]

# ---------------------------------------------------
# Baseline
# ---------------------------------------------------

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual,
        comparison_df["baseline_forecast"],
    )
)

baseline_mae = mean_absolute_error(
    actual,
    comparison_df["baseline_forecast"],
)

# ---------------------------------------------------
# CEV
# ---------------------------------------------------

cev_rmse = np.sqrt(
    mean_squared_error(
        actual,
        comparison_df["cev_forecast"],
    )
)

cev_mae = mean_absolute_error(
    actual,
    comparison_df["cev_forecast"],
)

# ---------------------------------------------------
# Metrics table
# ---------------------------------------------------

metrics_df = pd.DataFrame(
    {
        "Model": [
            "Statistical Baseline",
            "CEV Forecast",
            "Random Forest",
            "XGBoost",
        ],
        "RMSE": [
            baseline_rmse,
            cev_rmse,
            rf_rmse,
            xgb_rmse,
        ],
        "MAE": [
            baseline_mae,
            cev_mae,
            rf_mae,
            xgb_mae,
        ],
    }
)

# ===================================================
# Save outputs
# ===================================================

comparison_path = (
    OUTPUT_DIR
    / "model_comparison.csv"
)

metrics_path = (
    OUTPUT_DIR
    / "model_metrics.csv"
)

comparison_df.to_csv(
    comparison_path
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)

# ===================================================
# Print summary
# ===================================================

print("\nModel Comparison Metrics")

print("-" * 60)

print(metrics_df)

print("\nLatest Forecast Comparison")

print("-" * 60)

print(
    comparison_df.tail()
)

print("\nSaved files")

print("-" * 60)

print(comparison_path)

print(metrics_path)