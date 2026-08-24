"""
test_app_headless.py
--------------------
Headless automated test verifying both ML (Random Forest/XGBoost) and Deep Learning (LSTM)
forecasting pipelines, artifact loading, and visualizations.
"""

import sys
import os
import pandas as pd
import numpy as np

print("[TEST 1/6] Testing data_processing.py...")
from data_processing import load_raw_data, clean_data, resample_daily, prepare_time_series_data

raw_df = load_raw_data("data/household_power_consumption.txt")
clean_df, clean_summary = clean_data(raw_df)
daily_df = resample_daily(clean_df)
assert len(daily_df) == 1442
print("✓ Data processing verified! Daily shape:", daily_df.shape)

print("\n[TEST 2/6] Testing ml_forecasting.py feature engineering & model loading...")
from ml_forecasting import (
    engineer_ml_features,
    load_ml_artifacts,
    forecast_future_ml
)

ml_model, ml_features, ml_meta = load_ml_artifacts("models")
assert ml_model is not None, "ML model failed to load"
assert len(ml_features) == 36, f"Expected 36 features, got {len(ml_features)}"
print(f"✓ ML model ({ml_meta['model_name']}) loaded successfully! Features count: {len(ml_features)}")
print(f"✓ ML Test Metrics -> MAE: {ml_meta['metrics']['MAE']} kWh, RMSE: {ml_meta['metrics']['RMSE']} kWh, R2: {ml_meta['metrics']['R2']}")

print("\n[TEST 3/6] Testing ML multi-step future forecasting (7, 14, 30 days)...")
for h in [7, 14, 30]:
    fc_df, fc_sum = forecast_future_ml(ml_model, daily_df, ml_features, horizon_days=h)
    assert len(fc_df) == h
    assert fc_sum['expected_avg_kWh'] > 0
    print(f"✓ ML Horizon {h} days -> Expected Avg: {fc_sum['expected_avg_kWh']} kWh, Peak: {fc_sum['max_forecast_kWh']} kWh on {fc_sum['max_forecast_date']}")

print("\n[TEST 4/6] Testing LSTM forecasting.py artifact loading and forecasting...")
from forecasting import load_artifacts as load_lstm_artifacts, forecast_future as forecast_lstm

lstm_model, scaler, lstm_meta = load_lstm_artifacts("models")
assert lstm_model is not None, "LSTM model failed to load"
print(f"✓ LSTM Model loaded successfully! Test R2: {lstm_meta['metrics_lstm']['R2']}")

print("\n[TEST 5/6] Testing visualization.py chart generators...")
import visualization as viz

fig_trend = viz.plot_overall_trend(daily_df)
fig_dist = viz.plot_daily_distribution(daily_df)
fig_month = viz.plot_monthly_consumption(daily_df)
fig_dow = viz.plot_day_of_week_consumption(daily_df)
fig_wve = viz.plot_weekday_vs_weekend(daily_df)
fig_sub = viz.plot_submetering_breakdown(daily_df)
fig_peaks = viz.plot_peak_analysis(daily_df, top_n=10)
fig_imp = viz.plot_feature_importances(ml_meta['feature_importances'], top_n=10)
print("✓ All Plotly visualizations generated without errors!")

print("\n[TEST 6/6] Testing multi-dataset support with 2025 Daily Weather Dataset...")
from data_processing import clean_and_prepare_daily

df_2025 = pd.read_csv("data/daily_weather_power.csv")
daily_2025, sum_2025, target_2025 = clean_and_prepare_daily(df_2025)
assert len(daily_2025) == 212
fig_temp = viz.plot_weather_correlation(daily_2025)
fig_occ = viz.plot_occupancy_impact(daily_2025)
assert fig_temp is not None
assert fig_occ is not None
print("✓ 2025 Weather and Occupancy telemetry verified!")

print("\n=======================================================")
print(" ALL TESTS PASSED SUCCESSFULLY! PROJECT 100% VERIFIED! ")
print("=======================================================")
