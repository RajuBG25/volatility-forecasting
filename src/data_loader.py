"""
Data loader for market price data.

Downloads historical OHLCV data from Yahoo Finance
and saves it to data/raw/ as a CSV file.
"""

import yfinance as yf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw"


def download_price_data(
    symbol: str,
    start: str = "2015-01-01",
    end: str | None = None,
    out_dir: Path = OUT_DIR,
):
    """
    Download historical price data and save as CSV.
    """

    # ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # download data
    df = yf.download(symbol, start=start, end=end)

    if df.empty:
        raise ValueError("No data downloaded. Check symbol or dates.")

    # save CSV
    out_path = out_dir / f"{symbol}.csv"
    df.to_csv(out_path)

    print(f"Saved raw data to: {out_path}")


if __name__ == "__main__":
    download_price_data("SPY")
    download_price_data("AAPL")