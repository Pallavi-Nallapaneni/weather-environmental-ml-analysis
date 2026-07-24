"""
generate_data.py

Generates a synthetic daily weather dataset with the SAME schema as
fetch_noaa_data.py's output (date, tmax_c, tmin_c, precip_mm, wind_mps,
station). This exists purely as an offline fallback for development,
testing, and reviewers who want to run the pipeline without a live
internet connection to NOAA's servers.

The synthetic series is built from a seasonal sinusoid plus
autocorrelated noise, so it behaves realistically enough to sanity-check
the feature engineering and modeling code -- but it is NOT real
observational data and should never be presented as such.

Usage:
    python src/generate_data.py --start 2006-01-01 --end 2020-12-31 \
        --out data/raw/noaa_weather.csv
"""

import argparse

import numpy as np
import pandas as pd


def generate(start: str, end: str, station: str = "SYNTHETIC0001", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    day_of_year = dates.dayofyear.to_numpy()

    # Seasonal temperature signal (roughly mid-latitude Northern Hemisphere)
    seasonal = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)

    # Slow multi-year drift + autocorrelated daily noise (AR(1))
    drift = np.linspace(0, 0.6, n)  # mild long-run warming signal, synthetic
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 1.4)

    tmax = seasonal + drift + noise + rng.normal(0, 0.5, n)
    tmin = tmax - rng.uniform(5, 10, n)

    precip = rng.gamma(shape=0.4, scale=6.0, size=n)
    precip[rng.random(n) > 0.35] = 0.0  # ~35% of days have measurable precip

    wind = np.clip(rng.normal(4.0, 1.5, n), 0.2, None)

    df = pd.DataFrame({
        "date": dates,
        "tmax_c": tmax.round(1),
        "tmin_c": tmin.round(1),
        "precip_mm": precip.round(1),
        "wind_mps": wind.round(1),
        "station": station,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic offline weather data (fallback only).")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--out", default="data/raw/noaa_weather.csv")
    parser.add_argument("--station", default="SYNTHETIC0001")
    args = parser.parse_args()

    df = generate(args.start, args.end, station=args.station)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} synthetic daily records to {args.out}")
    print("NOTE: this is synthetic fallback data, not real NOAA observations.")


if __name__ == "__main__":
    main()
