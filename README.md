# Weather & Environmental ML Analysis

An end-to-end Python pipeline for analyzing daily weather observations, exploring long-term temperature and precipitation patterns, and predicting next-day temperature anomalies using time-series-aware machine learning.

The primary data path uses real observations from NOAA's Global Historical Climatology Network-Daily (GHCN-Daily) archive. An offline synthetic-data fallback is also included so the pipeline can be run without internet access.

The machine-learning model predicts how much warmer or cooler tomorrow's maximum temperature is expected to be compared with the seasonal climatological norm. This anomaly-based approach reduces the trivial effect of seasonal temperature patterns and provides a more meaningful test of weather-persistence signals.

## Project Overview

This project demonstrates:

- Real-world weather data acquisition
- Data cleaning and missing-value handling
- Time-series feature engineering
- Leakage-safe machine learning design
- Long-term weather trend analysis
- Chronological train/test splitting
- Baseline model comparison
- Random Forest regression for temperature anomaly prediction
- Seasonal climatology modeling
- Comparison against climatology and persistence baselines
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
Seasonal climatology estimation
        ↓
Temperature anomaly calculation
        ↓
Next-day anomaly prediction
        ↓
Baseline comparison and model evaluation
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

### 7. Train and evaluate the anomaly prediction model

```bash
python src/train_model.py --in data/processed/weather_features.csv --outdir images
```

## Offline Synthetic Data Fallback

If real NOAA data cannot be downloaded because of an internet connection problem, generate a schema-matching synthetic dataset instead:

```bash
python src/generate_data.py --start 2006-01-01 --end 2020-12-31 --out data/raw/noaa_weather.csv
```
## Machine Learning Approach

The model predicts the next day's maximum temperature anomaly rather than the raw temperature.

A temperature anomaly represents the difference between the observed temperature and the expected seasonal temperature:

```text
Temperature Anomaly =
Observed Temperature − Seasonal Climatological Temperature '''
