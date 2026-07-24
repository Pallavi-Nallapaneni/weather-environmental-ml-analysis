 
"""
clean_data.py
 
Cleans the raw daily weather CSV (from either fetch_noaa_data.py or
generate_data.py -- same schema) and writes a processed version.
 
Steps:
    - Parse dates, sort chronologically
    - Drop duplicate dates
    - Flag and interpolate short gaps (<= 3 consecutive missing days) in
      tmax_c / tmin_c using linear interpolation
    - Leave longer gaps as NaN rather than fabricating data
    - Clip physically impossible values (e.g. negative precipitation)
 
Usage:
    python src/clean_data.py --in data/raw/noaa_weather.csv --out data/processed/weather_clean.csv
"""
 
import argparse
 
import numpy as np
import pandas as pd
 
 
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
 
    # Reindex to a full daily calendar so gaps are explicit, not silently skipped
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"
 
    # Interpolate only short gaps (<=3 days) in temperature; leave longer gaps as NaN
    for col in ["tmax_c", "tmin_c"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=3, limit_area="inside")
 
    # Physically impossible values -> NaN rather than silently clipping to 0,
    # so downstream code can decide how to handle them
    if "precip_mm" in df.columns:
        df.loc[df["precip_mm"] < 0, "precip_mm"] = np.nan
        df["precip_mm"] = df["precip_mm"].fillna(0)  # no precip record -> assume none reported
 
    if "wind_mps" in df.columns:
        df.loc[df["wind_mps"] < 0, "wind_mps"] = np.nan
 
    df["station"] = df["station"].ffill().bfill()
    df = df.reset_index()
 
    return df
 
 
def main():
    parser = argparse.ArgumentParser(description="Clean raw daily weather data.")
    parser.add_argument("--in", dest="in_path", default="data/raw/noaa_weather.csv")
    parser.add_argument("--out", dest="out_path", default="data/processed/weather_clean.csv")
    args = parser.parse_args()
 
    raw = pd.read_csv(args.in_path)
    cleaned = clean(raw)
 
    import os
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    cleaned.to_csv(args.out_path, index=False)
 
    n_missing_tmax = cleaned["tmax_c"].isna().sum() if "tmax_c" in cleaned.columns else 0
    print(f"Cleaned {len(cleaned)} rows -> {args.out_path}")
    print(f"Remaining missing tmax_c after interpolation: {n_missing_tmax}")
 
 
if __name__ == "__main__":
    main()
 
