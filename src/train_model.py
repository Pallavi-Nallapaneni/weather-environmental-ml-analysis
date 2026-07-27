"""
train_model.py

Trains a Random Forest regressor to predict tomorrow's temperature
ANOMALY -- the deviation from the seasonal climatological norm -- rather
than the raw temperature value.

Why anomaly, not raw temperature: predicting raw next-day tmax is
dominated almost entirely by "today's temperature" (weather is highly
autocorrelated), which makes for a fairly uninteresting model -- the
feature-importance chart just shows one feature at ~99% and nothing
else. Predicting the anomaly instead asks a more interesting question:
given how far today is from what's normal for this time of year, how
far from normal will tomorrow be? That requires the model to actually
reason about persistence of weather regimes rather than exploit the
seasonal cycle directly.

Two baselines are compared against the Random Forest:
    1. Climatology baseline  -- predicts anomaly = 0 (i.e. "tomorrow
       will just be the seasonal normal"). This is the naive baseline
       when you ignore today's conditions entirely.
    2. Persistence-of-anomaly baseline -- predicts tomorrow's anomaly
       equals today's anomaly (i.e. "today's unusual warmth/coolness
       carries over unchanged").

The seasonal climatology curve (a harmonic regression on season_sin /
season_cos) is fit using ONLY the training set, then applied to both
train and test, so the test set's climatology is never informed by
test-period data -- this avoids a second, subtler form of leakage.

Uses a CHRONOLOGICAL train/test split (no shuffling), consistent with
the rest of this pipeline.

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ANOMALY_FEATURE_COLS = [
    "anomaly_today", "anomaly_lag1", "anomaly_lag2", "anomaly_roll7",
    "precip_today", "wind_today",
]


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def seasonal_sin_cos(dates: pd.Series, shift_days: int = 0):
    shifted = dates + pd.to_timedelta(shift_days, unit="D")
    doy = shifted.dt.dayofyear
    return np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)


def add_anomaly_columns(df: pd.DataFrame, harmonic_model: LinearRegression):
    def seasonal_mean(sin_vals, cos_vals):
        X = np.column_stack([sin_vals, cos_vals])
        return harmonic_model.predict(X)

    df["seasonal_mean_today"] = seasonal_mean(df["season_sin"], df["season_cos"])

    sin_l1, cos_l1 = seasonal_sin_cos(df["date"], shift_days=-1)
    df["seasonal_mean_lag1"] = seasonal_mean(sin_l1, cos_l1)

    sin_l2, cos_l2 = seasonal_sin_cos(df["date"], shift_days=-2)
    df["seasonal_mean_lag2"] = seasonal_mean(sin_l2, cos_l2)

    sin_next, cos_next = seasonal_sin_cos(df["date"], shift_days=1)
    df["seasonal_mean_next"] = seasonal_mean(sin_next, cos_next)

    df["anomaly_today"] = df["tmax_today"] - df["seasonal_mean_today"]
    df["anomaly_lag1"] = df["tmax_lag1"] - df["seasonal_mean_lag1"]
    df["anomaly_lag2"] = df["tmax_lag2"] - df["seasonal_mean_lag2"]
    # 7-day rolling average compared against today's seasonal norm (a
    # reasonable approximation since the seasonal curve barely moves
    # over a 7-day window).
    df["anomaly_roll7"] = df["tmax_roll7_past"] - df["seasonal_mean_today"]

    df["target_anomaly_next"] = df["target_tmax_next"] - df["seasonal_mean_next"]
    return df


def main():
    parser = argparse.ArgumentParser(description="Train next-day temperature ANOMALY model vs. two baselines.")
    parser.add_argument("--in", dest="in_path", default="data/processed/weather_features.csv")
    parser.add_argument("--outdir", default="images")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    df["date"] = pd.to_datetime(df["date"])

    train, test = chronological_split(df, args.test_frac)

    # --- Fit seasonal climatology curve on TRAIN ONLY ---
    harmonic = LinearRegression()
    harmonic.fit(train[["season_sin", "season_cos"]], train["tmax_today"])

    train = add_anomaly_columns(train, harmonic)
    test = add_anomaly_columns(test, harmonic)

    X_train, y_train = train[ANOMALY_FEATURE_COLS], train["target_anomaly_next"]
    X_test, y_test = test[ANOMALY_FEATURE_COLS], test["target_anomaly_next"]

    # --- Baseline 1: climatology (predict anomaly = 0) ---
    climatology_pred = np.zeros(len(test))
    climatology_mae = mean_absolute_error(y_test, climatology_pred)
    climatology_r2 = r2_score(y_test, climatology_pred)

    # --- Baseline 2: persistence of anomaly (tomorrow's anomaly = today's anomaly) ---
    persistence_pred = test["anomaly_today"].values
    persistence_mae = mean_absolute_error(y_test, persistence_pred)
    persistence_r2 = r2_score(y_test, persistence_pred)

    # --- Random Forest on anomaly features ---
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    rf_pred = model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    print("\nModel comparison (held-out chronological test set, predicting TEMPERATURE ANOMALY)")
    print(f"{'Model':<28}{'MAE (C)':>10}{'R2':>10}")
    print(f"{'Climatology (anomaly=0)':<28}{climatology_mae:>10.3f}{climatology_r2:>10.3f}")
    print(f"{'Persistence of anomaly':<28}{persistence_mae:>10.3f}{persistence_r2:>10.3f}")
    print(f"{'Random Forest':<28}{rf_mae:>10.3f}{rf_r2:>10.3f}")

    best_baseline_mae = min(climatology_mae, persistence_mae)
    improvement = (best_baseline_mae - rf_mae) / best_baseline_mae * 100
    print(f"\nRandom Forest reduces MAE vs. the better baseline by {improvement:.1f}%")

    # --- Feature importance plot ---
    os.makedirs(args.outdir, exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=ANOMALY_FEATURE_COLS).sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot.barh(color="darkgreen")
    plt.title("Random Forest Feature Importance (Anomaly Prediction)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "feature_importance.png"), dpi=150)
    plt.close()

    # --- Predicted vs actual anomaly plot ---
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, rf_pred, alpha=0.3, s=10, label="Random Forest")
    plt.scatter(y_test, persistence_pred, alpha=0.15, s=10, color="gray", label="Persistence of anomaly")
    lims = [min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.axhline(0, color="steelblue", linewidth=0.8, linestyle=":")
    plt.axvline(0, color="steelblue", linewidth=0.8, linestyle=":")
    plt.xlabel("Actual next-day temperature anomaly (\u00b0C)")
    plt.ylabel("Predicted next-day temperature anomaly (\u00b0C)")
    plt.title("Predicted vs. Actual Next-Day Temperature Anomaly")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "predictions_vs_actual.png"), dpi=150)
    plt.close()

    print(f"\nSaved feature_importance.png and predictions_vs_actual.png to {args.outdir}/")


if __name__ == "__main__":
    main()