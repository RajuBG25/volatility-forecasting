"""
Data preprocessing for market price data.

- Loads raw CSV from data/raw/
- Fixes Yahoo Finance multi-index columns
- Cleans missing values
- Keeps Close prices only
- Saves processed data to data/processed/
"""
import os
import pandas as pd


def preprocess_price_data(
    symbol: str,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
):
    raw_path = os.path.join(raw_dir, f"{symbol}.csv")

    # Read CSV (Yahoo saves Date as index)
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    # If columns are MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep Close column only
    if "Close" not in df.columns:
        raise ValueError(f"Close column not found. Columns: {df.columns}")

    df = df[["Close"]]

    # Force numeric, drop non-numeric rows (THIS FIXES 'SPY')
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()

    # Sort by date
    df = df.sort_index()

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, f"{symbol}_clean.csv")
    df.to_csv(out_path)

    print(f"Saved processed data to: {out_path}")


if __name__ == "__main__":
    preprocess_price_data("SPY")
