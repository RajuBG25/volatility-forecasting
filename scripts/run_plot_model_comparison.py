# scripts/plot_model_comparison.py

"""
Visualize volatility forecasting model performance.

This script:
1. Loads model comparison outputs
2. Creates forecast comparison plots
3. Creates RMSE / MAE comparison charts
4. Saves figures
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Project paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------
# Load results
# ---------------------------------------------------

comparison_df = pd.read_csv(
    RESULTS_DIR / "model_comparison.csv",
    index_col=0,
    parse_dates=True,
)

metrics_df = pd.read_csv(
    RESULTS_DIR / "model_metrics.csv",
)

# ===================================================
# Plot 1
# Actual vs Forecast Comparison
# ===================================================

plt.figure(figsize=(14, 7))

plt.plot(
    comparison_df.index,
    comparison_df["actual_future_vol"],
    label="Actual Future Volatility",
    linewidth=2,
)

plt.plot(
    comparison_df.index,
    comparison_df["baseline_forecast"],
    label="Statistical Baseline",
)

plt.plot(
    comparison_df.index,
    comparison_df["cev_forecast"],
    label="CEV Forecast",
)

plt.plot(
    comparison_df.index,
    comparison_df["rf_forecast"],
    label="Random Forest",
)

plt.plot(
    comparison_df.index,
    comparison_df["xgb_forecast"],
    label="XGBoost",
)

plt.title(
    "Volatility Forecast Comparison"
)

plt.xlabel("Date")

plt.ylabel("Annualized Volatility")

plt.legend()

plt.grid(True)

plt.tight_layout()

forecast_plot_path = (
    FIGURES_DIR
    / "forecast_comparison.png"
)

plt.savefig(
    forecast_plot_path,
    dpi=300,
)

plt.close()

# ===================================================
# Plot 2
# RMSE Comparison
# ===================================================

plt.figure(figsize=(10, 6))

plt.bar(
    metrics_df["Model"],
    metrics_df["RMSE"],
)

plt.title(
    "RMSE Comparison"
)

plt.ylabel("RMSE")

plt.xticks(rotation=10)

plt.grid(
    axis="y",
)

plt.tight_layout()

rmse_plot_path = (
    FIGURES_DIR
    / "rmse_comparison.png"
)

plt.savefig(
    rmse_plot_path,
    dpi=300,
)

plt.close()

# ===================================================
# Plot 3
# MAE Comparison
# ===================================================

plt.figure(figsize=(10, 6))

plt.bar(
    metrics_df["Model"],
    metrics_df["MAE"],
)

plt.title(
    "MAE Comparison"
)

plt.ylabel("MAE")

plt.xticks(rotation=10)

plt.grid(
    axis="y",
)

plt.tight_layout()

mae_plot_path = (
    FIGURES_DIR
    / "mae_comparison.png"
)

plt.savefig(
    mae_plot_path,
    dpi=300,
)

plt.close()

# ===================================================
# Print summary
# ===================================================

print("\nSaved figures")

print("-" * 50)

print(forecast_plot_path)

print(rmse_plot_path)

print(mae_plot_path)