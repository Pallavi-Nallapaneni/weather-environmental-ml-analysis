"""
train_model.py

Trains a Random Forest regressor to predict tomorrow's max temperature
and compares it against a persistence baseline (tomorrow's tmax =
today's tmax). Weather is highly autocorrelated, so the real question
is not "is R^2 high" but "does the model meaningfully beat the naive
baseline."

Uses a CHRONOLOGICAL train/test split (no shuffling) since this is
time-series data -- shuffling would let the model implicitly see
information from around the same time period as the held-out days.

Usage:
    python src/train_model.py --in data/processed/weather_features.csv --outdir images
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


FEATURE_COLS = [
    "tmax_today", "tmin_today", "tmax_lag1", "tmax_lag2",
    "tmax_roll7_past", "precip_today", "wind_today",
    "day_of_year", "month", "season_sin", "season_cos",
]


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Train next-day temperature model vs persistence baseline.")
    parser.add_argument("--in", dest="in_path", default="data/processed/weather_features.csv")
    parser.add_argument("--outdir", default="images")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    df["date"] = pd.to_datetime(df["date"])

    train, test = chronological_split(df, args.test_frac)

    X_train, y_train = train[FEATURE_COLS], train["target_tmax_next"]
    X_test, y_test = test[FEATURE_COLS], test["target_tmax_next"]

    # --- Persistence baseline: tomorrow = today ---
    baseline_pred = test["baseline_pred_tmax_next"]
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)

    # --- Random Forest model ---
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    rf_pred = model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    print("\nModel comparison (held-out chronological test set)")
    print(f"{'Model':<22}{'MAE (C)':>10}{'R2':>10}")
    print(f"{'Persistence baseline':<22}{baseline_mae:>10.3f}{baseline_r2:>10.3f}")
    print(f"{'Random Forest':<22}{rf_mae:>10.3f}{rf_r2:>10.3f}")

    improvement = (baseline_mae - rf_mae) / baseline_mae * 100
    print(f"\nRandom Forest reduces MAE vs. persistence baseline by {improvement:.1f}%")

    # --- Feature importance plot ---
    os.makedirs(args.outdir, exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    plt.figure(figsize=(8, 6))
    importances.plot.barh(color="darkgreen")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "feature_importance.png"), dpi=150)
    plt.close()

    # --- Predicted vs actual plot ---
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, rf_pred, alpha=0.3, s=10, label="Random Forest")
    plt.scatter(y_test, baseline_pred, alpha=0.15, s=10, color="gray", label="Persistence baseline")
    lims = [min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("Actual next-day TMAX (\u00b0C)")
    plt.ylabel("Predicted next-day TMAX (\u00b0C)")
    plt.title("Predicted vs. Actual Next-Day Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "predictions_vs_actual.png"), dpi=150)
    plt.close()

    print(f"\nSaved feature_importance.png and predictions_vs_actual.png to {args.outdir}/")


if __name__ == "__main__":
    main()
