# src/random_forest_model.py

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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
    # Current realized volatility
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
    # Future volatility target
    # ---------------------------------------------------

    out["target_vol"] = (
        out["rv20"]
        .shift(-future_window)
    )

    return out.dropna()


def train_random_forest(
    df: pd.DataFrame,
):
    """
    Train Random Forest volatility forecasting model.
    """

    data = prepare_features(df)

    # Features
    X = data[
        [
            "rv20_lag1",
            "rv20_lag5",
            "rv20_lag10",
        ]
    ]

    # Target
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
    # Random Forest model
    # ---------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    mae = mean_absolute_error(
        y_test,
        preds,
    )
    pred_df = pd.DataFrame(
    {
        "actual_future_vol": y_test,
        "rf_forecast": preds,
    },
    index=y_test.index,
    )      
    return (
        model,
        rmse,
        mae,
        pred_df
    )