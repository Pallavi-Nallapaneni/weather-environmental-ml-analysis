# Weather & Environmental ML Analysis

An end-to-end Python pipeline for analyzing daily weather observations, exploring long-term temperature and precipitation patterns, and predicting next-day maximum temperature using time-series-aware machine learning.

The primary data path uses real observations from NOAA's Global Historical Climatology Network-Daily (GHCN-Daily) archive. An offline synthetic-data fallback is also included so the pipeline can be run without internet access.

## Project Overview

This project demonstrates:
- Real-world weather data acquisition
- Data cleaning and missing-value handling
- Time-series feature engineering
- Leakage-safe machine learning design
- Long-term weather trend analysis
- Chronological train/test splitting
- Baseline model comparison
- Random Forest regression
- Data visualization

## Pipeline

```text
NOAA GHCN-Daily data
        ↓
Data download and parsing
        ↓
Data cleaning
        ↓
Feature engineering
        ↓
Trend analysis
        ↓
Next-day temperature prediction
        ↓
Model evaluation
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Pallavi-Nallapaneni/weather-environmental-ml-analysis.git
cd weather-environmental-ml-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download real NOAA weather data

The primary workflow uses real observations from NOAA's GHCN-Daily dataset:

```bash
python src/fetch_noaa_data.py --station USW00094728 --start 2006 --end 2020 --out data/raw/noaa_weather.csv
```

### 4. Clean the data

```bash
python src/clean_data.py --in data/raw/noaa_weather.csv --out data/processed/weather_clean.csv
```

### 5. Build machine-learning features

```bash
python src/feature_engineering.py --in data/processed/weather_clean.csv --out data/processed/weather_features.csv
```

### 6. Run weather trend analysis

```bash
python src/analyze_noaa.py --in data/processed/weather_clean.csv --outdir images
```

### 7. Train and evaluate the model

```bash
python src/train_model.py --in data/processed/weather_features.csv --outdir images
```

## Offline Synthetic Data Fallback

If real NOAA data can't be downloaded (e.g. no internet connection), generate a schema-matching synthetic dataset instead:

```bash
python src/generate_data.py --start 2006-01-01 --end 2020-12-31 --out data/raw/noaa_weather.csv
```
