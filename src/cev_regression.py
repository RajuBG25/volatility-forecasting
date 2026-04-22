import numpy as np
import pandas as pd
from typing import Tuple

TRADING_DAYS = 252

def estimate_cev_regression(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Estimate CEV parameters (sigma, beta) via log–log regression:
        log(ΔS^2) = a + 2β log(S) + error
    Returns (sigma_hat, beta_hat).
    """
    # Ensure Close exists and is numeric
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")

    S = df["Close"].astype(float)

    # Price increments
    dS = S.diff().dropna()
    S_lag = S.shift(1).dropna()

    # Align
    dS = dS.loc[S_lag.index]
    # Remove zero price increments (log undefined)
    mask = dS != 0
    dS = dS[mask]
    S_lag = S_lag[mask]


    # Regression variables
    y = np.log(dS.values ** 2)
    x = np.log(S_lag.values)

    # Add intercept
    X = np.column_stack([np.ones(len(x)), x])

    # OLS: (X'X)^{-1} X'y
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    a_hat, b_hat = beta_ols

    beta_hat = b_hat / 2.0
    dt = 1.0 / TRADING_DAYS
    sigma_hat = np.sqrt(np.exp(a_hat) / dt)

    return float(sigma_hat), float(beta_hat)
