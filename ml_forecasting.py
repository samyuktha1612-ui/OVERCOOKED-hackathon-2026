"""
ml_forecasting.py
-----------------
Machine Learning Time-Series Forecasting Module for Household Electricity Consumption.
Includes comprehensive feature engineering (calendar, cyclical, multi-horizon lags,
rolling statistics, momentum ratios), model training (Random Forest, Gradient Boosting, XGBoost),
leakage-free chronological evaluation, model persistence via joblib, and recursive multi-step forecasting.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def engineer_ml_features(
    daily_df: pd.DataFrame,
    target_col: str = 'Daily_energy_kWh'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs a rich tabular feature matrix from daily time-series data:
      - Calendar: day, day_of_week, month, year, quarter, day_of_year, is_weekend
      - Cyclical Encodings: sin/cos of day_of_week and month
      - Multi-Period Lags: 1, 2, 3, 4, 5, 6, 7, 14, 21, 30 days
      - Rolling Window Statistics: mean, std, min, max over 7, 14, 30 day windows (strictly shifted)
      - Trend & Ratios: lag differences and relative ratio to moving averages
      
    Parameters:
        daily_df (pd.DataFrame): Daily aggregated time series with DatetimeIndex.
        target_col (str): Target column name.
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Feature-engineered DataFrame (NaN-free) and list of feature column names.
    """
    df = daily_df.copy().sort_index()
    
    # 1. Calendar Features
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. Cyclical Features (captures continuous seasonality)
    df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 3. Multi-Horizon Lags (Autoregressive features)
    lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
    for lag in lag_days:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
        
    # 4. Rolling Window Statistics (strictly shift(1) to prevent leakage)
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
        df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
        df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
        
    # 5. Trend & Momentum Ratios
    df['diff_lag1_lag2'] = df['lag_1'] - df['lag_2']
    df['diff_lag1_lag7'] = df['lag_1'] - df['lag_7']
    df['ratio_lag1_rolling7'] = df['lag_1'] / (df['rolling_mean_7'] + 1e-5)
    
    # Drop rows with NaNs resulting from the 30-day lookback lag
    df_clean = df.dropna().copy()
    
    # Identify feature columns
    excluded_cols = [
        target_col, 'date', 'Datetime', 'day_name', 'month_name',
        'Global_active_power_sum', 'Global_active_power_mean',
        'Global_active_power_max', 'Global_active_power_min', 'Global_active_power_std',
        'Global_reactive_power_mean', 'Voltage_mean', 'Global_intensity_mean',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_remainder',
        'Lag_1_kWh', 'Lag_7_kWh', 'Rolling_Mean_7'
    ]
    feature_cols = [c for c in df_clean.columns if c not in excluded_cols and pd.api.types.is_numeric_dtype(df_clean[c])]
    
    return df_clean, feature_cols


def train_evaluate_ml_models(
    daily_df: pd.DataFrame,
    target_col: str = 'Daily_energy_kWh',
    train_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    Trains and benchmarks Random Forest, Gradient Boosting, and XGBoost models
    on chronological train/test splits.
    
    Returns:
        Dict[str, Any]: Dictionary containing best model, all model metrics,
                        predictions, feature importances, and train/test splits.
    """
    df_feat, feature_cols = engineer_ml_features(daily_df, target_col=target_col)
    
    total_len = len(df_feat)
    train_size = int(total_len * train_ratio)
    
    train_df = df_feat.iloc[:train_size]
    test_df = df_feat.iloc[train_size:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"[INFO] Chronological ML Split: Train={len(X_train)} samples ({train_df.index.min().date()} to {train_df.index.max().date()}), "
          f"Test={len(X_test)} samples ({test_df.index.min().date()} to {test_df.index.max().date()})")
    print(f"[INFO] Total Engineered Features: {len(feature_cols)}")
    
    # Define candidate models
    candidate_models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            random_state=42
        )
    }
    
    if HAS_XGBOOST:
        candidate_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
    model_evaluations = {}
    best_model_name = None
    best_rmse = float('inf')
    
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.maximum(0.0, preds)
        
        mae = float(mean_absolute_error(y_test, preds))
        mse = float(mean_squared_error(y_test, preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))
        
        denom = np.where(np.abs(y_test) < 1e-5, 1e-5, np.abs(y_test))
        mape = float(np.mean(np.abs((y_test - preds) / denom)) * 100.0)
        
        metrics = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'MAPE': round(mape, 2)
        }
        
        # Feature importances
        if hasattr(model, 'feature_importances_'):
            importances = dict(zip(feature_cols, [round(float(x), 4) for x in model.feature_importances_]))
            sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        else:
            sorted_importances = {}
            
        model_evaluations[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': preds,
            'feature_importances': sorted_importances
        }
        
        print(f"[EVALUATION] {name} -> MAE: {mae:.2f} kWh | RMSE: {rmse:.2f} kWh | R²: {r2:.4f} | MAPE: {mape:.2f}%")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            
    best_entry = model_evaluations[best_model_name]
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_entry['model'],
        'best_metrics': best_entry['metrics'],
        'best_predictions': best_entry['predictions'],
        'best_feature_importances': best_entry['feature_importances'],
        'all_evaluations': model_evaluations,
        'feature_cols': feature_cols,
        'test_dates': test_df.index,
        'y_test': y_test.values,
        'train_df': train_df,
        'test_df': test_df
    }


def save_ml_artifacts(
    model: Any,
    feature_cols: List[str],
    metadata: Dict[str, Any],
    model_dir: str = "models",
    all_models: Optional[Dict[str, Any]] = None
) -> None:
    """Saves the trained machine learning model(s) and configuration using joblib."""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "ml_forecast_model.joblib")
    all_models_path = os.path.join(model_dir, "all_ml_models.joblib")
    config_path = os.path.join(model_dir, "ml_feature_config.json")
    meta_path = os.path.join(model_dir, "ml_metadata.json")
    
    print(f"[INFO] Saving primary ML model to {model_path} via joblib...")
    joblib.dump(model, model_path)
    
    if all_models is not None:
        print(f"[INFO] Saving all candidate ML models to {all_models_path}...")
        joblib.dump(all_models, all_models_path)
    
    print(f"[INFO] Saving feature configuration to {config_path}...")
    with open(config_path, "w") as f:
        json.dump({'feature_cols': feature_cols}, f, indent=4)
        
    serializable_meta = {k: v for k, v in metadata.items() if k != 'all_models'}
    print(f"[INFO] Saving ML metadata to {meta_path}...")
    with open(meta_path, "w") as f:
        json.dump(serializable_meta, f, indent=4)
        
    print("[INFO] Machine learning artifacts persisted successfully.")


def load_ml_artifacts(model_dir: str = "models") -> Tuple[Optional[Any], Optional[List[str]], Optional[Dict[str, Any]]]:
    """
    Loads saved ML model, feature columns, and metadata.
    
    Returns:
        Tuple of (model, feature_cols, metadata) or (None, None, None) if not found.
    """
    model_path = os.path.join(model_dir, "ml_forecast_model.joblib")
    all_models_path = os.path.join(model_dir, "all_ml_models.joblib")
    config_path = os.path.join(model_dir, "ml_feature_config.json")
    meta_path = os.path.join(model_dir, "ml_metadata.json")
    
    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        return None, None, None
        
    try:
        model = joblib.load(model_path)
        with open(config_path, "r") as f:
            config = json.load(f)
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                
        if os.path.exists(all_models_path):
            try:
                metadata['all_models'] = joblib.load(all_models_path)
            except Exception as e:
                print(f"[NOTE] Could not load all_ml_models: {e}")
                metadata['all_models'] = {metadata.get('model_name', 'Random Forest'): model}
        else:
            metadata['all_models'] = {metadata.get('model_name', 'Random Forest'): model}
            
        return model, config['feature_cols'], metadata
    except Exception as e:
        print(f"[WARNING] Failed to load ML artifacts: {e}")
        return None, None, None


def forecast_future_ml(
    model: Any,
    historical_daily_df: pd.DataFrame,
    feature_cols: List[str],
    horizon_days: int = 30,
    target_col: str = 'Daily_energy_kWh'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates multi-step future electricity forecasts using an autoregressive ML pipeline.
    Recursively updates the rolling lag buffer with previous step predictions.
    
    Parameters:
        model: Fitted regression model (RandomForest, XGBoost, etc.).
        historical_daily_df: Historical daily series (at least 35 days of history).
        feature_cols: Exact list of feature column names expected by the model.
        horizon_days: Number of future days to forecast (7, 14, 30).
        target_col: Name of target energy column.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Forecast DataFrame and summary dictionary.
    """
    # Create rolling buffer from history
    buffer_df = historical_daily_df[[target_col]].copy().sort_index()
    last_date = buffer_df.index[-1]
    
    future_dates = pd.date_range(
        start=pd.to_datetime(last_date) + pd.Timedelta(days=1),
        periods=horizon_days,
        freq='D'
    )
    
    forecast_values = []
    
    for current_date in future_dates:
        # Build features for current_date from buffer_df
        feat_dict = {}
        
        # 1. Calendar
        feat_dict['day'] = current_date.day
        feat_dict['day_of_week'] = current_date.dayofweek
        feat_dict['month'] = current_date.month
        feat_dict['year'] = current_date.year
        feat_dict['quarter'] = current_date.quarter
        feat_dict['day_of_year'] = current_date.dayofyear
        feat_dict['is_weekend'] = 1 if current_date.dayofweek >= 5 else 0
        
        # 2. Cyclical
        feat_dict['sin_dow'] = np.sin(2 * np.pi * feat_dict['day_of_week'] / 7.0)
        feat_dict['cos_dow'] = np.cos(2 * np.pi * feat_dict['day_of_week'] / 7.0)
        feat_dict['sin_month'] = np.sin(2 * np.pi * feat_dict['month'] / 12.0)
        feat_dict['cos_month'] = np.cos(2 * np.pi * feat_dict['month'] / 12.0)
        
        # 3. Lags from buffer
        series_vals = buffer_df[target_col].values
        lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
        for l in lag_days:
            idx = -l
            feat_dict[f'lag_{l}'] = series_vals[idx] if len(series_vals) >= l else series_vals[0]
            
        # 4. Rolling stats from buffer (shift 1)
        for w in [7, 14, 30]:
            sub_window = series_vals[-w:] if len(series_vals) >= w else series_vals
            feat_dict[f'rolling_mean_{w}'] = float(np.mean(sub_window))
            feat_dict[f'rolling_std_{w}'] = float(np.std(sub_window)) if len(sub_window) > 1 else 0.0
            feat_dict[f'rolling_min_{w}'] = float(np.min(sub_window))
            feat_dict[f'rolling_max_{w}'] = float(np.max(sub_window))
            
        # 5. Differencing & Ratios
        feat_dict['diff_lag1_lag2'] = feat_dict['lag_1'] - feat_dict['lag_2']
        feat_dict['diff_lag1_lag7'] = feat_dict['lag_1'] - feat_dict['lag_7']
        feat_dict['ratio_lag1_rolling7'] = feat_dict['lag_1'] / (feat_dict['rolling_mean_7'] + 1e-5)
        
        # Construct input DataFrame with exact feature order
        X_step = pd.DataFrame([feat_dict])[feature_cols]
        
        # Predict
        pred_val = float(model.predict(X_step)[0])
        pred_val = max(0.0, pred_val)
        forecast_values.append(pred_val)
        
        # Append prediction to buffer for subsequent recursive steps
        new_row = pd.DataFrame({target_col: [pred_val]}, index=[current_date])
        buffer_df = pd.concat([buffer_df, new_row])
        
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast_kWh': np.round(forecast_values, 2),
        'Day_Name': future_dates.day_name(),
        'Is_Weekend': [1 if d >= 5 else 0 for d in future_dates.dayofweek]
    }).set_index('Date')
    
    summary = {
        'horizon_days': horizon_days,
        'expected_avg_kWh': round(float(forecast_df['Forecast_kWh'].mean()), 2),
        'total_expected_kWh': round(float(forecast_df['Forecast_kWh'].sum()), 2),
        'max_forecast_kWh': round(float(forecast_df['Forecast_kWh'].max()), 2),
        'max_forecast_date': str(forecast_df['Forecast_kWh'].idxmax().strftime('%Y-%m-%d (%a)')),
        'min_forecast_kWh': round(float(forecast_df['Forecast_kWh'].min()), 2),
        'min_forecast_date': str(forecast_df['Forecast_kWh'].idxmin().strftime('%Y-%m-%d (%a)'))
    }
    
    return forecast_df, summary
