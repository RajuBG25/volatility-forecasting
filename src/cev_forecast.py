# src/cev_forecast.py

"""
Rolling CEV-based volatility forecasting.

This module:
1. Estimates rolling CEV parameters
2. Generates CEV-implied volatility forecasts

Returns:
- cev_sigma
- cev_beta
- cev_vol_forecast
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_cev_forecast(
    df: pd.DataFrame,
    price_col: str = "Close",
    window: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling CEV volatility forecasts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data
    price_col : str
        Price column name
    window : int
        Rolling estimation window

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - cev_sigma
        - cev_beta
        - cev_vol_forecast
    """

    # ---------------------------------------------------
    # Validate input
    # ---------------------------------------------------

    if price_col not in df.columns:
        raise ValueError(f"{price_col} column not found.")

    # Price series
    S = df[price_col].astype(float)

    # Storage
    sigma_list = []
    beta_list = []
    forecast_list = []
    dates = []

    dt = 1.0 / TRADING_DAYS

    # ---------------------------------------------------
    # Rolling estimation loop
    # ---------------------------------------------------

    for i in range(window, len(S)):

        # Rolling window
        S_window = S.iloc[i - window:i]

        # Price increments
        dS = S_window.diff().dropna()

        # Lagged prices
        S_lag = S_window.shift(1).dropna()

        # Align indices
        dS = dS.loc[S_lag.index]

        # Remove zero increments
        mask = dS != 0

        dS = dS[mask]
        S_lag = S_lag[mask]

        # Skip tiny samples
        if len(dS) < 10:
            continue

        # ---------------------------------------------------
        # Log-log regression
        #
        # log(dS^2) = a + 2β log(S) + error
        # ---------------------------------------------------

        y = np.log(dS.values ** 2)

        x = np.log(S_lag.values)

        # Design matrix
        X = np.column_stack([
            np.ones(len(x)),
            x,
        ])

        # OLS estimation
        coeffs = np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )[0]

        a_hat, b_hat = coeffs

        # Recover CEV parameters
        beta_hat = b_hat / 2.0

        sigma_hat = np.sqrt(
            np.exp(a_hat) / dt
        )

        # ---------------------------------------------------
        # CEV-implied volatility forecast
        # ---------------------------------------------------

        S_t = S.iloc[i]

        cev_vol_forecast = (
            sigma_hat
            * (S_t ** (beta_hat - 1))
        )

        # Store results
        sigma_list.append(float(sigma_hat))

        beta_list.append(float(beta_hat))

        forecast_list.append(float(cev_vol_forecast))

        dates.append(S.index[i])

    # ---------------------------------------------------
    # Final output DataFrame
    # ---------------------------------------------------

    forecasts = pd.DataFrame(
        {
            "cev_sigma": sigma_list,
            "cev_beta": beta_list,
            "cev_vol_forecast": forecast_list,
        },
        index=dates,
    )

    return forecasts