"""
analyze_noaa.py

Analyzes long-term local weather patterns from the cleaned daily dataset
and produces summary plots.

Deliberately conservative framing: this analyzes observed patterns at a
single station over the available period. It does NOT claim to prove or
disprove global climate change from one station's short record.

Usage:
    python src/analyze_noaa.py --in data/processed/weather_clean.csv --outdir images
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_yearly_trend(df: pd.DataFrame, outdir: str):
    yearly = df.groupby(df["date"].dt.year)["tmax_c"].mean().dropna()
    plt.figure(figsize=(9, 5))
    plt.plot(yearly.index, yearly.values, marker="o")
    plt.title("Mean Annual Max Temperature (Observed, Single Station)")
    plt.xlabel("Year")
    plt.ylabel("Mean TMAX (\u00b0C)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "yearly_temperature_trends.png"), dpi=150)
    plt.close()


def plot_precip_trend(df: pd.DataFrame, outdir: str):
    yearly = df.groupby(df["date"].dt.year)["precip_mm"].sum().dropna()
    plt.figure(figsize=(9, 5))
    plt.bar(yearly.index, yearly.values, color="steelblue")
    plt.title("Total Annual Precipitation (Observed, Single Station)")
    plt.xlabel("Year")
    plt.ylabel("Total Precipitation (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "precipitation_trends.png"), dpi=150)
    plt.close()


def plot_seasonal_pattern(df: pd.DataFrame, outdir: str):
    monthly = df.groupby(df["date"].dt.month)["tmax_c"].agg(["mean", "std"]).dropna()
    plt.figure(figsize=(9, 5))
    plt.errorbar(monthly.index, monthly["mean"], yerr=monthly["std"], fmt="-o", capsize=4)
    plt.title("Seasonal Temperature Pattern (Monthly Mean \u00b1 Std Dev)")
    plt.xlabel("Month")
    plt.ylabel("TMAX (\u00b0C)")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "seasonal_temperature_pattern.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize local weather trends.")
    parser.add_argument("--in", dest="in_path", default="data/processed/weather_clean.csv")
    parser.add_argument("--outdir", default="images")
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    df["date"] = pd.to_datetime(df["date"])
    os.makedirs(args.outdir, exist_ok=True)

    plot_yearly_trend(df, args.outdir)
    plot_precip_trend(df, args.outdir)
    plot_seasonal_pattern(df, args.outdir)

    print(f"Saved trend plots to {args.outdir}/")
    print("Note: this reflects patterns observed at a single station over "
          "the available record. It is not a claim about global climate trends.")


if __name__ == "__main__":
    main()
