# scripts/run_random_forest.py

"""
Run Random Forest volatility forecasting pipeline.

This script:
1. Loads processed price data
2. Trains Random Forest model
3. Evaluates forecasting performance
4. Prints feature importance
"""

from pathlib import Path

import pandas as pd

from src.random_forest_model import (
    train_random_forest,
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

# ---------------------------------------------------
# Load processed data
# ---------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True,
)

# ---------------------------------------------------
# Train Random Forest model
# ---------------------------------------------------

(
    model,
    rmse,
    mae,
    pred_df
) = train_random_forest(df)

# ---------------------------------------------------
# Results
# ---------------------------------------------------

print("\nRandom Forest Volatility Forecast")

print("-" * 50)

print(f"RMSE : {rmse:.6f}")

print(f"MAE  : {mae:.6f}")

# ---------------------------------------------------
# Latest prediction
# ---------------------------------------------------

print("\nLatest Prediction")

print("-" * 50)

print(
    f"Predicted : "
    f"{pred_df['rf_forecast'].iloc[-1]:.6f}"
)

print(
    f"Actual    : "
    f"{pred_df['actual_future_vol'].iloc[-1]:.6f}"
)

# ---------------------------------------------------
# Feature importance
# ---------------------------------------------------

print("\nFeature Importance")

print("-" * 50)

features = [
    "rv20_lag1",
    "rv20_lag5",
    "rv20_lag10",
]

for name, importance in zip(
    features,
    model.feature_importances_,
):
    print(
        f"{name}: {importance:.4f}"
    )