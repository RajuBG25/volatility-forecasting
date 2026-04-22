import os
import pandas as pd

from cev_regression import estimate_cev_regression

# Move to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Load processed data
df = pd.read_csv(
    "data/processed/SPY_clean.csv",
    index_col=0,
    parse_dates=True
)

sigma_hat, beta_hat = estimate_cev_regression(df)

print(f"CEV (Regression) sigma: {sigma_hat:.4f}")
print(f"CEV (Regression) beta : {beta_hat:.4f}")
