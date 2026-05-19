# src/xgboost_model.py

"""
XGBoost model for future volatility forecasting.

This module:
1. Builds volatility forecasting features
2. Uses lagged realized volatility
3. Predicts future 20-day realized volatility
"""

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

TRADING_DAYS = 252


def prepare_features(
    df: pd.DataFrame,
    future_window: int = 20,
) -> pd.DataFrame:
    """
    Prepare volatility forecasting dataset.
    """

    out = df.copy()

    # ---------------------------------------------------
    # Log returns
    # ---------------------------------------------------

    out["log_ret"] = np.log(
        out["Close"] / out["Close"].shift(1)
    )

    # ---------------------------------------------------
    # Rolling realized volatility
    # ---------------------------------------------------

    out["rv20"] = (
        out["log_ret"]
        .rolling(window=20)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    # ---------------------------------------------------
    # Lagged volatility features
    # ---------------------------------------------------

    out["rv20_lag1"] = out["rv20"].shift(1)

    out["rv20_lag5"] = out["rv20"].shift(5)

    out["rv20_lag10"] = out["rv20"].shift(10)

    # ---------------------------------------------------
    # Future realized volatility target
    # ---------------------------------------------------

    out["target_vol"] = (
        out["rv20"]
        .shift(-future_window)
    )

    return out.dropna()


def train_xgboost(
    df: pd.DataFrame,
):
    """
    Train XGBoost volatility forecasting model.
    """

    # ---------------------------------------------------
    # Prepare dataset
    # ---------------------------------------------------

    data = prepare_features(df)

    X = data[
        [
            "rv20_lag1",
            "rv20_lag5",
            "rv20_lag10",
        ]
    ]

    y = data["target_vol"]

    # ---------------------------------------------------
    # Time-series split
    # ---------------------------------------------------

    split_date = "2023-01-01"

    X_train = X.loc[:split_date]
    X_test = X.loc[split_date:]

    y_train = y.loc[:split_date]
    y_test = y.loc[split_date:]

    # ---------------------------------------------------
    # XGBoost model
    # ---------------------------------------------------

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    # Train model
    model.fit(
        X_train,
        y_train,
    )

    # Predict
    preds = model.predict(X_test)

    # ---------------------------------------------------
    # Evaluation metrics
    # ---------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds,
        )
    )

    mae = mean_absolute_error(
        y_test,
        preds,
    )
    pred_df = pd.DataFrame(
    {
        "actual_future_vol": y_test,
        "xgb_forecast": preds,
    },
    index=y_test.index,
    )

    return (
        model,
        rmse,
        mae,
        pred_df
    )