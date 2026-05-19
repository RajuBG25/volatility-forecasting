# src/statistical_baseline.py

"""
Statistical baseline models for volatility forecasting.

This module provides:
1. Log return computation
2. Rolling realized volatility
3. Naive future volatility forecast

The baseline assumption is:

    future volatility ≈ current realized volatility
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_log_returns(
    df: pd.DataFrame,
    price_col: str = "Close",
) -> pd.Series:
    """
    Compute log returns.
    """

    prices = df[price_col].astype(float)

    log_returns = np.log(
        prices / prices.shift(1)
    )

    return log_returns


def rolling_realized_volatility(
    df: pd.DataFrame,
    price_col: str = "Close",
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute rolling realized volatility.
    """

    log_returns = compute_log_returns(
        df,
        price_col=price_col,
    )

    rv = (
        log_returns
        .rolling(
            window=window,
            min_periods=window,
        )
        .std(ddof=1)
    )

    if annualize:
        rv = rv * np.sqrt(TRADING_DAYS)

    rv.name = "rv20"

    return rv


def statistical_volatility_forecast(
    df: pd.DataFrame,
    price_col: str = "Close",
    window: int = 20,
) -> pd.DataFrame:
    """
    Naive statistical volatility forecast.

    Forecast rule:
        future volatility ≈ current rolling volatility
    """

    out = df.copy()

    # ---------------------------------------------------
    # Current realized volatility
    # ---------------------------------------------------

    out["rv20"] = rolling_realized_volatility(
        out,
        price_col=price_col,
        window=window,
    )

    # ---------------------------------------------------
    # Forecast
    # ---------------------------------------------------

    out["baseline_forecast"] = out["rv20"]

    # ---------------------------------------------------
    # Future realized volatility target
    # ---------------------------------------------------

    out["future_rv20"] = (
        out["rv20"]
        .shift(-window)
    )

    # Keep only relevant columns
    out = out[
        [
            "rv20",
            "baseline_forecast",
            "future_rv20",
        ]
    ]

    return out.dropna()
