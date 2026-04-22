import os
import pandas as pd

from .volatility_simple import bs_sigma, rolling_sigma20

# move to project root (robust to how Python is launched)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# load processed data
df = pd.read_csv(
    "data/processed/SPY_clean.csv",
    index_col=0,
    parse_dates=True
)
df.index.name = "Date"

sigma_bs = bs_sigma(df)
sigma_20 = rolling_sigma20(df).dropna().iloc[-1]

print(f"Black–Scholes constant sigma (annualized): {sigma_bs:.4f}")
print(f"Latest 20-day rolling vol (annualized): {sigma_20:.4f}")
