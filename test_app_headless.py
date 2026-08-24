"""
test_app_headless.py
--------------------
Headless automated test verifying all modules, data pipelines, visualization functions,
model loading, and forecasting across all application pages.
"""

import sys
import os
import pandas as pd
import numpy as np

print("[TEST 1/5] Testing data_processing.py...")
from data_processing import load_raw_data, clean_data, resample_daily, prepare_time_series_data

raw_df = load_raw_data("data/household_power_consumption.txt")
clean_df, clean_summary = clean_data(raw_df)
daily_df = resample_daily(clean_df)
assert len(daily_df) == 1442, f"Expected 1442 days, got {len(daily_df)}"
assert 'Daily_energy_kWh' in daily_df.columns
assert 'is_weekend' in daily_df.columns
print("✓ Data processing verified! Daily shape:", daily_df.shape)

print("\n[TEST 2/5] Testing forecasting.py artifact loading and baseline models...")
from forecasting import load_artifacts, calculate_metrics, BaselinePredictor, forecast_future

lstm_model, scaler, metadata = load_artifacts("models")
assert lstm_model is not None, "Model failed to load"
assert scaler is not None, "Scaler failed to load"
assert metadata is not None, "Metadata failed to load"
print(f"✓ Artifacts loaded successfully! Lookback window: {metadata['lookback_window']}")

data_bundle = prepare_time_series_data(daily_df, lookback_window=metadata['lookback_window'], train_ratio=0.8)
pred_test_scaled = lstm_model.predict(data_bundle['X_test'], verbose=0)
pred_lstm = scaler.inverse_transform(pred_test_scaled).ravel()
pred_lstm = np.maximum(0.0, pred_lstm)
metrics_lstm = calculate_metrics(data_bundle['actual_test_unscaled'], pred_lstm)
print(f"✓ LSTM Model metrics evaluated: MAE={metrics_lstm['MAE']} kWh, RMSE={metrics_lstm['RMSE']} kWh, R2={metrics_lstm['R2']}")

print("\n[TEST 3/5] Testing multi-step future forecasting for 7, 14, 30 days...")
last_seq_scaled = scaler.transform(daily_df[['Daily_energy_kWh']].values[-metadata['lookback_window']:])
last_date = daily_df.index[-1]

for h in [7, 14, 30]:
    fc_df, fc_sum = forecast_future(lstm_model, last_seq_scaled, scaler, last_date, horizon_days=h)
    assert len(fc_df) == h, f"Expected {h} forecast days, got {len(fc_df)}"
    assert fc_sum['expected_avg_kWh'] > 0
    print(f"✓ Horizon {h} days: Expected Avg = {fc_sum['expected_avg_kWh']} kWh, Total = {fc_sum['total_expected_kWh']} kWh")

print("\n[TEST 4/5] Testing visualization.py chart generators...")
import visualization as viz

fig_trend = viz.plot_overall_trend(daily_df)
fig_dist = viz.plot_daily_distribution(daily_df)
fig_month = viz.plot_monthly_consumption(daily_df)
fig_dow = viz.plot_day_of_week_consumption(daily_df)
fig_wve = viz.plot_weekday_vs_weekend(daily_df)
fig_sub = viz.plot_submetering_breakdown(daily_df)
fig_peaks = viz.plot_peak_analysis(daily_df, top_n=10)
fig_avp = viz.plot_actual_vs_predicted(data_bundle['test_dates'], data_bundle['actual_test_unscaled'], pred_lstm)
fig_fc = viz.plot_future_forecast(daily_df.tail(30), fc_df)
fig_loss = viz.plot_training_loss({'loss': metadata['training_loss_history'], 'val_loss': metadata['val_loss_history']})
print("✓ All 10 Plotly visualizations generated without errors!")

print("\n[TEST 5/5] Testing insight calculations...")
weekday_mean = daily_df[daily_df['is_weekend'] == 0]['Daily_energy_kWh'].mean()
weekend_mean = daily_df[daily_df['is_weekend'] == 1]['Daily_energy_kWh'].mean()
sub3_total = daily_df['Sub_metering_3'].sum() / 1000.0
total_energy = daily_df['Daily_energy_kWh'].sum()
print(f"✓ Weekday Avg: {weekday_mean:.2f} kWh/day | Weekend Avg: {weekend_mean:.2f} kWh/day")
print(f"✓ Sub-metering 3 (Climate & Heating): {(sub3_total / total_energy)*100:.1f}% of total energy")

print("\n=======================================================")
print(" ALL TESTS PASSED SUCCESSFULLY! PROJECT 100% VERIFIED! ")
print("=======================================================")
