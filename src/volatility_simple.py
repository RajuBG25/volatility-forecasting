# src/volatility_simple.py
"""
Tiny volatility helpers for the Black–Scholes baseline.

Provides:
- compute_log_returns(df) -> Series
- bs_sigma(df) -> float   # one constant sigma (annualized)
- rolling_sigma20(df) -> Series  # 20-day rolling annualized sigma (diagnostic)
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def compute_log_returns(df, price_col="Close"):
    s = df[price_col].astype(float).copy()
    return np.log(s / s.shift(1))

def bs_sigma(df, price_col="Close", annualize=True):
    """Estimate single BS sigma from entire sample (annualized by default)."""
    lr = compute_log_returns(df, price_col=price_col).dropna()
    sigma = lr.std(ddof=0)
    if annualize:
        sigma = sigma * np.sqrt(TRADING_DAYS)
    return float(sigma)

def rolling_sigma20(df, price_col="Close", annualize=True):
    """20-day rolling std of log returns (annualized), aligned with df index."""
    lr = compute_log_returns(df, price_col=price_col)
    roll = lr.rolling(window=20, min_periods=5).std(ddof=0)
    if annualize:
        roll = roll * np.sqrt(TRADING_DAYS)
    roll.name = "rv_20"
    return roll
