from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def preprocess_price_data(
    symbol: str,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
):
    raw_path = raw_dir / f"{symbol}.csv"

    # Read CSV
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep Close only
    if "Close" not in df.columns:
        raise ValueError(f"Close column not found. Columns: {df.columns}")

    df = df[["Close"]]

    # Convert numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()

    # Sort by date
    df = df.sort_index()

    # Ensure folder exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / f"{symbol}_clean.csv"
    df.to_csv(out_path)

    print(f"Saved processed data to: {out_path}")


if __name__ == "__main__":
    preprocess_price_data("SPY")