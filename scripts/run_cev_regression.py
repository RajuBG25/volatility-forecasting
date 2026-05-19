# scripts/run_cev_forecast.py

"""
Run the rolling CEV volatility forecasting pipeline.

This script:
1. Loads processed price data
2. Computes rolling CEV forecasts
3. Computes future 20-day realized volatility
4. Saves the final feature dataset
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.cev_forecast import rolling_cev_forecast

TRADING_DAYS = 252

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

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PATH = (
    OUT_DIR
    / "SPY_cev_features.csv"
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
# Generate rolling CEV forecasts
# ---------------------------------------------------

cev_df = rolling_cev_forecast(
    df=df,
    price_col="Close",
    window=60,
)

# ---------------------------------------------------
# Compute future 20-day realized volatility
# ---------------------------------------------------

prices = df["Close"].astype(float)

log_returns = np.log(
    prices / prices.shift(1)
)

future_rv_20 = (
    log_returns
    .rolling(window=20)
    .std(ddof=1)
    * np.sqrt(TRADING_DAYS)
).shift(-20)

future_rv_20.name = "future_rv_20"

# Align target with forecast dates
cev_df["future_rv_20"] = (
    future_rv_20.loc[cev_df.index]
)

# Remove missing rows
cev_df = cev_df.dropna()

# ---------------------------------------------------
# Save output
# ---------------------------------------------------

cev_df.to_csv(OUT_PATH)

print(
    f"Saved CEV forecast features to:\n{OUT_PATH}"
)

print("\nPreview:")

print(cev_df.tail())
