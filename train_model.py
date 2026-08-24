"""
train_model.py
--------------
End-to-end model training script for Household Electricity Consumption Forecasting.
Executes the full pipeline: data cleaning, daily aggregation, baseline evaluation,
LSTM deep learning model training, model comparison, and artifact serialization.
"""

import os
import sys
import numpy as np
import pandas as pd

from data_processing import (
    load_raw_data,
    clean_data,
    resample_daily,
    prepare_time_series_data
)
from forecasting import (
    BaselinePredictor,
    build_lstm_model,
    train_lstm_model,
    calculate_metrics,
    save_artifacts,
    forecast_future
)


def run_training_pipeline(
    data_path: str = "data/household_power_consumption.txt",
    lookback_window: int = 30,
    epochs: int = 60,
    batch_size: int = 32,
    save_dir: str = "models"
) -> dict:
    """Executes the full training and evaluation workflow."""
    print("=" * 80)
    print(" HOUSEHOLD ELECTRICITY CONSUMPTION FORECASTING: MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    raw_df = load_raw_data(data_path)
    
    # 2. Clean Data
    clean_df, clean_summary = clean_data(raw_df)
    
    # 3. Resample Daily
    daily_df = resample_daily(clean_df)
    
    # 4. Prepare Time Series Sequences
    data_bundle = prepare_time_series_data(
        daily_df,
        target_col='Daily_energy_kWh',
        lookback_window=lookback_window,
        train_ratio=0.8
    )
    
    train_df = data_bundle['train_df']
    test_df = data_bundle['test_df']
    scaler = data_bundle['scaler']
    X_train, y_train = data_bundle['X_train'], data_bundle['y_train']
    X_val, y_val = data_bundle['X_val'], data_bundle['y_val']
    X_test, y_test = data_bundle['X_test'], data_bundle['y_test']
    actual_test_unscaled = data_bundle['actual_test_unscaled']
    test_dates = data_bundle['test_dates']
    
    print("\n" + "-" * 80)
    print(" STEP 5: EVALUATING BASELINE TIME-SERIES MODELS")
    print("-" * 80)
    
    # Full unscaled series across train and test for continuous baseline calculation
    full_unscaled_series = np.concatenate([
        train_df['Daily_energy_kWh'].values,
        test_df['Daily_energy_kWh'].values
    ])
    # The test portion starts at train_size
    train_size = len(train_df)
    test_context_unscaled = full_unscaled_series[train_size - lookback_window :]
    
    # Baseline 1: Persistence (y_t = y_{t-1})
    pred_persistence = BaselinePredictor.persistence_predict(
        test_context_unscaled, lookback_window=lookback_window
    )
    metrics_persistence = calculate_metrics(actual_test_unscaled, pred_persistence)
    print(f" Baseline 1 (Persistence T-1) -> MAE: {metrics_persistence['MAE']:.2f} kWh | RMSE: {metrics_persistence['RMSE']:.2f} kWh | R2: {metrics_persistence['R2']:.4f} | MAPE: {metrics_persistence['MAPE']:.2f}%")
    
    # Baseline 2: 7-Day Moving Average
    pred_ma7 = BaselinePredictor.moving_average_predict(
        test_context_unscaled, lookback_window=lookback_window, ma_window=7
    )
    metrics_ma7 = calculate_metrics(actual_test_unscaled, pred_ma7)
    print(f" Baseline 2 (7-Day Moving Avg) -> MAE: {metrics_ma7['MAE']:.2f} kWh | RMSE: {metrics_ma7['RMSE']:.2f} kWh | R2: {metrics_ma7['R2']:.4f} | MAPE: {metrics_ma7['MAPE']:.2f}%")
    
    print("\n" + "-" * 80)
    print(" STEP 6: BUILDING & TRAINING LSTM DEEP LEARNING MODEL")
    print("-" * 80)
    
    # Build LSTM Model
    lstm_model = build_lstm_model(
        lookback_window=lookback_window,
        num_features=1,
        units_1=64,
        units_2=32,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    lstm_model.summary()
    
    # Train LSTM
    lstm_model, history = train_lstm_model(
        lstm_model,
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=12
    )
    
    # Evaluate LSTM on Unseen Test Data
    pred_test_scaled = lstm_model.predict(X_test, verbose=0)
    pred_lstm_unscaled = scaler.inverse_transform(pred_test_scaled).ravel()
    # Clip negative values
    pred_lstm_unscaled = np.maximum(0.0, pred_lstm_unscaled)
    
    metrics_lstm = calculate_metrics(actual_test_unscaled, pred_lstm_unscaled)
    print(f"\n LSTM Model (Out-of-Sample Test) -> MAE: {metrics_lstm['MAE']:.2f} kWh | RMSE: {metrics_lstm['RMSE']:.2f} kWh | R2: {metrics_lstm['R2']:.4f} | MAPE: {metrics_lstm['MAPE']:.2f}%")
    
    print("\n" + "=" * 80)
    print(" STEP 7: MODEL COMPARISON TABLE")
    print("=" * 80)
    
    comparison_df = pd.DataFrame([
        {
            'Model': 'Baseline: Persistence (Lag-1)',
            'MAE (kWh)': metrics_persistence['MAE'],
            'RMSE (kWh)': metrics_persistence['RMSE'],
            'R² Score': metrics_persistence['R2'],
            'MAPE (%)': metrics_persistence['MAPE'],
            'Type': 'Heuristic Baseline'
        },
        {
            'Model': 'Baseline: 7-Day Moving Average',
            'MAE (kWh)': metrics_ma7['MAE'],
            'RMSE (kWh)': metrics_ma7['RMSE'],
            'R² Score': metrics_ma7['R2'],
            'MAPE (%)': metrics_ma7['MAPE'],
            'Type': 'Heuristic Baseline'
        },
        {
            'Model': 'Deep Learning: Stacked LSTM',
            'MAE (kWh)': metrics_lstm['MAE'],
            'RMSE (kWh)': metrics_lstm['RMSE'],
            'R² Score': metrics_lstm['R2'],
            'MAPE (%)': metrics_lstm['MAPE'],
            'Type': 'Deep Neural Network'
        }
    ])
    print(comparison_df.to_string(index=False))
    
    # Step 8: Multi-step future forecast demo
    print("\n" + "-" * 80)
    print(" STEP 8: TESTING MULTI-STEP FUTURE FORECASTING (30 DAYS)")
    print("-" * 80)
    last_seq_scaled = scaler.transform(daily_df[['Daily_energy_kWh']].values[-lookback_window:])
    last_hist_date = daily_df.index[-1]
    fc_df, fc_summary = forecast_future(
        lstm_model, last_seq_scaled, scaler, last_hist_date, horizon_days=30
    )
    print("Forecast summary (Next 30 Days):", fc_summary)
    print("First 5 forecasted days:")
    print(fc_df.head())
    
    # Save artifacts
    metadata = {
        'target_col': 'Daily_energy_kWh',
        'lookback_window': lookback_window,
        'train_records': len(train_df),
        'test_records': len(test_df),
        'total_daily_records': len(daily_df),
        'start_date': str(daily_df.index.min().date()),
        'end_date': str(daily_df.index.max().date()),
        'avg_daily_consumption_kwh': round(float(daily_df['Daily_energy_kWh'].mean()), 2),
        'max_daily_consumption_kwh': round(float(daily_df['Daily_energy_kWh'].max()), 2),
        'min_daily_consumption_kwh': round(float(daily_df['Daily_energy_kWh'].min()), 2),
        'clean_summary': clean_summary,
        'metrics_persistence': metrics_persistence,
        'metrics_ma7': metrics_ma7,
        'metrics_lstm': metrics_lstm,
        'training_loss_history': [float(x) for x in history.get('loss', [])],
        'val_loss_history': [float(x) for x in history.get('val_loss', [])]
    }
    
    save_artifacts(lstm_model, scaler, metadata, model_dir=save_dir)
    print("\n[SUCCESS] Training pipeline completed and all models/scalers saved successfully!")
    
    return metadata


if __name__ == "__main__":
    run_training_pipeline()
