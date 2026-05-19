# scripts/run_statistical_baseline.py

"""
Run statistical baseline volatility forecasting pipeline.

This script:
1. Loads processed price data
2. Computes rolling realized volatility
3. Generates naive volatility forecasts
4. Compares forecasts with future realized volatility
"""

from pathlib import Path

import pandas as pd

from src.statistical_baseline import (
    statistical_volatility_forecast,
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

df.index.name = "Date"

# ---------------------------------------------------
# Generate statistical baseline forecasts
# ---------------------------------------------------

baseline_df = statistical_volatility_forecast(
    df=df,
    price_col="Close",
    window=20,
)

# ---------------------------------------------------
# Display latest forecasts
# ---------------------------------------------------

print("\nStatistical Baseline Volatility Forecast")

print("-" * 50)

print("\nLatest Forecast Comparison")

print("-" * 50)

print(
    baseline_df.tail()
)
