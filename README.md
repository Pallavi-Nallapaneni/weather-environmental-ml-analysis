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
