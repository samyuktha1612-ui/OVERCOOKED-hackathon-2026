"""
train_ml_model.py
-----------------
Standalone CLI script to train, evaluate, and save machine learning time-series
forecasting models (Random Forest, XGBoost, Gradient Boosting) for Household Electricity Consumption.
"""

import os
import sys
import pandas as pd
import numpy as np

from data_processing import load_raw_data, clean_data, resample_daily
from ml_forecasting import (
    train_evaluate_ml_models,
    save_ml_artifacts,
    forecast_future_ml
)


def run_ml_training_pipeline(
    data_path: str = "data/household_power_consumption.txt",
    save_dir: str = "models"
) -> dict:
    """Executes the machine learning training and evaluation pipeline."""
    print("=" * 80)
    print(" HOUSEHOLD ELECTRICITY CONSUMPTION: MACHINE LEARNING FORECASTING PIPELINE")
    print("=" * 80)
    
    # 1. Load and Clean Data
    raw_df = load_raw_data(data_path)
    clean_df, clean_summary = clean_data(raw_df)
    daily_df = resample_daily(clean_df)
    
    # 2. Train and Benchmark ML Models
    eval_bundle = train_evaluate_ml_models(
        daily_df,
        target_col='Daily_energy_kWh',
        train_ratio=0.8
    )
    
    best_name = eval_bundle['best_model_name']
    best_model = eval_bundle['best_model']
    best_metrics = eval_bundle['best_metrics']
    feature_cols = eval_bundle['feature_cols']
    importances = eval_bundle['best_feature_importances']
    
    print("\n" + "=" * 80)
    print(f" TOP PERFORMING ML MODEL: {best_name}")
    print("=" * 80)
    print(f" Test MAE  : {best_metrics['MAE']:.4f} kWh")
    print(f" Test RMSE : {best_metrics['RMSE']:.4f} kWh")
    print(f" Test R²   : {best_metrics['R2']:.4f}")
    print(f" Test MAPE : {best_metrics['MAPE']:.2f}%")
    
    print("\n Top 10 Most Important Features:")
    for rank, (feat, score) in enumerate(list(importances.items())[:10], 1):
        print(f"  {rank:2d}. {feat:<24} : {score:.4f}")
        
    # 3. Test Multi-Step Future Forecasting (30 Days)
    print("\n" + "-" * 80)
    print(" TESTING MULTI-STEP ML FUTURE FORECAST (30 DAYS)")
    print("-" * 80)
    fc_df, fc_summary = forecast_future_ml(
        best_model, daily_df, feature_cols, horizon_days=30
    )
    print(" Forecast Summary:", fc_summary)
    print("\n First 5 Forecasted Days:")
    print(fc_df.head())
    
    # 4. Save Artifacts
    meta = {
        'model_name': best_name,
        'metrics': best_metrics,
        'all_evaluations': {
            k: v['metrics'] for k, v in eval_bundle['all_evaluations'].items()
        },
        'feature_cols': feature_cols,
        'feature_importances': importances,
        'train_samples': len(eval_bundle['train_df']),
        'test_samples': len(eval_bundle['test_df']),
        'target_col': 'Daily_energy_kWh'
    }
    
    save_ml_artifacts(best_model, feature_cols, meta, model_dir=save_dir)
    print("\n[SUCCESS] Machine learning model trained and saved to models/ml_forecast_model.joblib successfully!")
    
    return eval_bundle


if __name__ == "__main__":
    run_ml_training_pipeline()
