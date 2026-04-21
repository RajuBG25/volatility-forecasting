"""
Data loader for market price data.

Downloads historical OHLCV data from Yahoo Finance
and saves it to data/raw/ as a CSV file.
"""

import os
import yfinance as yf


def download_price_data(
    symbol: str,
    start: str = "2015-01-01",
    end: str | None = None,
    out_dir: str = "data/raw",
):
    """
    Download historical price data and save as CSV.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. 'SPY')
    start : str
        Start date in YYYY-MM-DD format
    end : str | None
        End date in YYYY-MM-DD format (None = today)
    out_dir : str
        Output directory for raw data
    """
    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # download data
    df = yf.download(symbol, start=start, end=end)

    if df.empty:
        raise ValueError("No data downloaded. Check symbol or dates.")

    # save CSV
    out_path = os.path.join(out_dir, f"{symbol}.csv")
    df.to_csv(out_path)

    print(f"Saved raw data to: {out_path}")


if __name__ == "__main__":
    download_price_data("SPY")
