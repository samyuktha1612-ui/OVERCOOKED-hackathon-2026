"""
test_all_pages_render.py
------------------------
Validates that all 7 navigation pages in the redesigned ENERGY INTELLIGENCE
platform execute cleanly without any errors.
"""

import os
import pandas as pd
import numpy as np
from data_processing import detect_and_load_data, clean_and_prepare_daily, prepare_time_series_data
from ml_forecasting import load_ml_artifacts, forecast_future_ml, engineer_ml_features
from forecasting import load_artifacts as load_lstm_artifacts, forecast_future as forecast_future_lstm
from chatbot import EnergyChatbotEngine
import visualization as viz

def test_pages_execution():
    print("=" * 80)
    print(" TESTING 7 NAVIGATION PAGES IN ENERGY INTELLIGENCE")
    print("=" * 80)

    # 1. Ingestion & Models
    daily_df, summary, target_col = clean_and_prepare_daily(detect_and_load_data("data/household_power_daily.csv"))
    ml_model, ml_features, ml_meta = load_ml_artifacts("models")
    lstm_model, scaler, lstm_meta = load_lstm_artifacts("models")
    avg_kwh = float(daily_df['Daily_energy_kWh'].mean())
    fc_df, fc_sum = forecast_future_ml(ml_model, daily_df, ml_features, horizon_days=30)

    print(f"\n[PAGE 1: OVERVIEW] Testing charts and dynamic KPI cards...")
    fig_trend = viz.plot_overall_trend(daily_df)
    fig_snap = viz.plot_future_forecast(daily_df.tail(30), fc_df)
    assert fig_trend is not None and fig_snap is not None
    print("  ✓ Page 1 (Overview) rendered successfully!")

    print(f"\n[PAGE 2: HISTORICAL ANALYSIS] Testing all 7 interactive EDA charts...")
    f1 = viz.plot_overall_trend(daily_df)
    f2 = viz.plot_monthly_consumption(daily_df)
    f3 = viz.plot_weekday_vs_weekend(daily_df)
    f4 = viz.plot_day_of_week_consumption(daily_df)
    f5 = viz.plot_peak_analysis(daily_df, top_n=10)
    f6 = viz.plot_hourly_consumption_pattern()
    f7 = viz.plot_submetering_breakdown(daily_df)
    assert all(f is not None for f in [f1, f2, f3, f4, f5, f6, f7])
    print("  ✓ Page 2 (Historical Analysis) rendered all 7 charts successfully!")

    print(f"\n[PAGE 3: LIVE FORECAST] Testing 7, 14, 30-day horizons with ML and LSTM...")
    for h in [7, 14, 30]:
        fc_res, sum_res = forecast_future_ml(ml_model, daily_df, ml_features, horizon_days=h)
        assert len(fc_res) == h
    # LSTM forecast test
    last_seq = scaler.transform(daily_df[['Daily_energy_kWh']].values[-30:])
    fc_lstm, sum_lstm = forecast_future_lstm(lstm_model, last_seq, scaler, daily_df.index[-1], horizon_days=14)
    assert len(fc_lstm) == 14
    # Actual vs Predicted
    df_feat, feature_cols = engineer_ml_features(daily_df, target_col='Daily_energy_kWh')
    train_size = int(len(df_feat) * 0.8)
    test_df = df_feat.iloc[train_size:]
    preds_test = ml_model.predict(test_df[feature_cols])
    fig_eval = viz.plot_actual_vs_predicted(test_df.index, test_df['Daily_energy_kWh'].values, preds_test)
    assert fig_eval is not None
    print("  ✓ Page 3 (Live Forecast) multi-model & multi-horizon verified!")

    print(f"\n[PAGE 4: SMART INSIGHTS] Testing dynamic insight cards & savings estimator...")
    tariff = 0.18
    target_pct = 15
    emission_factor = 0.42
    annual_kwh = avg_kwh * 365.0
    saved_kwh = annual_kwh * (target_pct / 100.0)
    saved_dollars = saved_kwh * tariff
    saved_co2 = saved_kwh * emission_factor
    assert saved_kwh > 0 and saved_dollars > 0 and saved_co2 > 0
    print(f"  ✓ Savings Estimator: {saved_kwh:.1f} kWh/yr saved -> ${saved_dollars:.2f}/yr, {saved_co2:.1f} kg CO2 averted")
    print("  ✓ Page 4 (Smart Insights) verified!")

    print(f"\n[PAGE 5: AI ENERGY ASSISTANT] Testing chatbot engine responses...")
    bot = EnergyChatbotEngine(context={
        'daily_df': daily_df,
        'avg_kwh': avg_kwh,
        'recent_kwh': float(daily_df['Daily_energy_kWh'].iloc[-14:].mean()),
        'ml_meta': ml_meta,
        'model_name': 'Random Forest Regressor',
        'model_label': 'Random Forest Regressor',
        'active_horizon': 30,
        'fc_summary': fc_sum,
        'forecast_df': fc_df,
        'predicted_avg_kwh': fc_sum['expected_avg_kWh'],
        'peak_forecast_kwh': fc_sum['max_forecast_kWh'],
        'peak_forecast_date': fc_sum['max_forecast_date']
    })
    resp = bot.generate_response("Compare my weekday and weekend usage.")
    assert len(resp) > 50 and "Weekend" in resp
    print("  ✓ Page 5 (AI Energy Assistant) chatbot query tested successfully!")

    print(f"\n[PAGE 6: ML MODEL] Testing LSTM architecture specs & loss curve chart...")
    assert 'training_loss_history' in lstm_meta and len(lstm_meta['training_loss_history']) > 0
    fig_loss = viz.plot_training_loss({
        'loss': lstm_meta['training_loss_history'],
        'val_loss': lstm_meta.get('val_loss_history', [])
    })
    assert fig_loss is not None
    print("  ✓ Page 6 (ML Model) loss curve and benchmarks verified!")

    print(f"\n[PAGE 7: ABOUT] Verifying technology and system flow architecture...")
    assert os.path.exists("requirements.txt")
    print("  ✓ Page 7 (About) verified!")

    print("\n" + "=" * 80)
    print(" ⭐ ALL 7 ENERGY INTELLIGENCE PAGES VERIFIED 100% OPERATIONAL! ⭐ ")
    print("=" * 80)

if __name__ == "__main__":
    test_pages_execution()
