"""
fetch_noaa_data.py

Downloads real daily weather observations from NOAA's Global Historical
Climatology Network - Daily (GHCN-Daily) archive.

Data source: https://www.ncei.noaa.gov/pub/data/ghcn/daily/
Station list: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt

Usage:
    python src/fetch_noaa_data.py --station USW00094728 --start 2006 --end 2020

If no internet connection is available, use generate_data.py instead to
create a synthetic dataset with the same schema for offline development
and testing.
"""

import argparse
import io
import os
import sys
import urllib.request

import pandas as pd

GHCN_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station}.dly"

# Column widths for the fixed-width .dly GHCN-Daily format
DLY_COLSPECS = [
    (0, 11), (11, 15), (15, 17), (17, 21),
]
ELEMENT_VALUE_WIDTH = 8  # each day-value block is 8 chars (5 value + 3 flags)


def parse_dly(path_or_buffer):
    """Parse a raw NOAA .dly fixed-width file into a tidy DataFrame with
    columns: station, date, element, value.
    """
    rows = []
    with open(path_or_buffer, "r") if isinstance(path_or_buffer, str) else path_or_buffer as f:
        for line in f:
            station = line[0:11]
            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21].strip()
            values_block = line[21:]
            for day in range(31):
                start = day * ELEMENT_VALUE_WIDTH
                chunk = values_block[start:start + ELEMENT_VALUE_WIDTH]
                if len(chunk) < 5:
                    continue
                raw_val = chunk[0:5]
                try:
                    val = int(raw_val)
                except ValueError:
                    continue
                if val == -9999:
                    continue  # missing value sentinel
                try:
                    date = pd.Timestamp(year=year, month=month, day=day + 1)
                except ValueError:
                    continue  # invalid day for this month
                rows.append((station, date, element, val))
    return pd.DataFrame(rows, columns=["station", "date", "element", "value"])


def download_station(station_id: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    url = GHCN_BASE_URL.format(station=station_id)
    dest_path = os.path.join(dest_dir, f"{station_id}.dly")
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy long-format NOAA data to one row per date with
    TMAX / TMIN / PRCP / AWND (tenths of units per NOAA spec, converted below).
    """
    wide = df.pivot_table(index="date", columns="element", values="value", aggfunc="first")
    wide = wide.reset_index()

    # NOAA stores TMAX/TMIN in tenths of degrees C, PRCP in tenths of mm,
    # AWND in tenths of m/s. Convert to natural units.
    for col, factor in [("TMAX", 0.1), ("TMIN", 0.1), ("PRCP", 0.1), ("AWND", 0.1)]:
        if col in wide.columns:
            wide[col] = wide[col] * factor

    rename = {"TMAX": "tmax_c", "TMIN": "tmin_c", "PRCP": "precip_mm", "AWND": "wind_mps"}
    wide = wide.rename(columns={k: v for k, v in rename.items() if k in wide.columns})
    return wide


def main():
    parser = argparse.ArgumentParser(description="Fetch NOAA GHCN-Daily station data.")
    parser.add_argument("--station", default="USW00094728",
                         help="GHCN station ID (default: USW00094728, LaGuardia AP, NY)")
    parser.add_argument("--start", type=int, default=2006)
    parser.add_argument("--end", type=int, default=2020)
    parser.add_argument("--out", default="data/raw/noaa_weather.csv")
    args = parser.parse_args()

    try:
        raw_path = download_station(args.station, "data/raw")
    except Exception as e:
        print(f"ERROR: could not download NOAA data ({e}).", file=sys.stderr)
        print("If you don't have an internet connection right now, run "
              "src/generate_data.py to create a synthetic dataset with the "
              "same schema instead.", file=sys.stderr)
        sys.exit(1)

    long_df = parse_dly(raw_path)
    wide_df = to_wide(long_df)
    wide_df = wide_df[(wide_df["date"].dt.year >= args.start) & (wide_df["date"].dt.year <= args.end)]
    wide_df = wide_df.sort_values("date").reset_index(drop=True)
    wide_df["station"] = args.station

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wide_df.to_csv(args.out, index=False)
    print(f"Saved {len(wide_df)} daily records to {args.out}")


if __name__ == "__main__":
    main()
