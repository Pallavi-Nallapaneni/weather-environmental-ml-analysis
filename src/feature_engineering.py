"""
feature_engineering.py

Builds features for next-day temperature prediction.

IMPORTANT (data leakage): every feature here is constructed using only
information available up to and including "today" (the row's own date).
The prediction target, tomorrow's tmax_c, is created by shifting the
target column BACKWARD relative to the feature row, not by including
any future value inside a rolling window. Rolling/lag windows below use
only past and present days -- never future ones.

Usage:
    python src/feature_engineering.py --in data/processed/weather_clean.csv \
        --out data/processed/weather_features.csv
"""

import argparse

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Target: TOMORROW's max temperature ---
    df["target_tmax_next"] = df["tmax_c"].shift(-1)

    # --- Features using only data up to and including today ---
    df["tmax_today"] = df["tmax_c"]
    df["tmin_today"] = df["tmin_c"]
    df["tmax_lag1"] = df["tmax_c"].shift(1)          # yesterday
    df["tmax_lag2"] = df["tmax_c"].shift(2)           # 2 days ago
    # Rolling average computed over the PAST 7 days INCLUDING today, using
    # only values already observed by end of "today" -- no lookahead.
    df["tmax_roll7_past"] = df["tmax_c"].rolling(window=7, min_periods=3).mean()
    df["precip_today"] = df["precip_mm"]
    df["wind_today"] = df["wind_mps"]
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["season_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Simple persistence baseline prediction, computed alongside the
    # features so train_model.py can evaluate it on the exact same rows.
    df["baseline_pred_tmax_next"] = df["tmax_today"]

    feature_cols = [
        "date", "tmax_today", "tmin_today", "tmax_lag1", "tmax_lag2",
        "tmax_roll7_past", "precip_today", "wind_today", "day_of_year",
        "month", "season_sin", "season_cos",
        "baseline_pred_tmax_next", "target_tmax_next",
    ]
    out = df[feature_cols].dropna().reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build leakage-safe features for next-day temp prediction.")
    parser.add_argument("--in", dest="in_path", default="data/processed/weather_clean.csv")
    parser.add_argument("--out", dest="out_path", default="data/processed/weather_features.csv")
    args = parser.parse_args()

    clean = pd.read_csv(args.in_path)
    features = build_features(clean)

    import os
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    features.to_csv(args.out_path, index=False)
    print(f"Built {len(features)} feature rows -> {args.out_path}")


if __name__ == "__main__":
    main()
