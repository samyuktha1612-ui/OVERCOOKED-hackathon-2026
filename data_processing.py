"""
data_processing.py
------------------
Robust data preprocessing, auto-format detection, cleaning, feature engineering,
and time-series sequence generation pipeline for Household Electricity Consumption Forecasting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, Optional, Union, List
import io


UCI_NUMERIC_COLUMNS = [
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3'
]


def detect_and_load_data(source: Union[str, io.BytesIO, pd.DataFrame]) -> pd.DataFrame:
    """
    Intelligently detects dataset format (UCI semicolon format or standard daily CSV)
    and loads it into a clean DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
        
    if isinstance(source, str) and not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")
        
    # Read sample to detect delimiter and columns
    if isinstance(source, str):
        with open(source, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
    else:
        first_line = source.readline().decode('utf-8', errors='ignore')
        source.seek(0)
        
    sep = ';' if ';' in first_line else ','
    
    df = pd.read_csv(
        source,
        sep=sep,
        na_values=['?', 'NA', 'null', 'None', ''],
        low_memory=False
    )
    return df


def clean_and_prepare_daily(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Cleans raw dataset and transforms it into a standard daily time-series DataFrame.
    Automatically detects dataset structure:
      - If raw minute-level UCI dataset: combines Date+Time, interpolates, resamples to Daily_energy_kWh.
      - If daily dataset (with Weather/Occupancy): parses Date, handles missing lags, sets target.
      
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], str]: (daily_df, summary_stats, target_column_name)
    """
    initial_records = len(df)
    cols = df.columns.tolist()
    
    # Case A: UCI Minute Dataset (contains Global_active_power and Time)
    if 'Global_active_power' in cols and 'Time' in cols:
        print("[INFO] Detected UCI minute-level power dataset...")
        
        # Parse Datetime
        df['Datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )
        df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['Datetime'], keep='first')
        
        # Convert numeric columns
        for col in UCI_NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        initial_missing = int(df[UCI_NUMERIC_COLUMNS].isnull().any(axis=1).sum())
        df[UCI_NUMERIC_COLUMNS] = df[UCI_NUMERIC_COLUMNS].interpolate(method='linear').bfill().ffill()
        df = df.set_index('Datetime')
        
        # Resample daily
        daily = df.resample('D').agg({
            'Global_active_power': ['sum', 'mean', 'max', 'min', 'std'],
            'Global_reactive_power': 'mean',
            'Voltage': 'mean',
            'Global_intensity': 'mean',
            'Sub_metering_1': 'sum',
            'Sub_metering_2': 'sum',
            'Sub_metering_3': 'sum'
        })
        daily.columns = [
            'Global_active_power_sum', 'Global_active_power_mean', 'Global_active_power_max',
            'Global_active_power_min', 'Global_active_power_std', 'Global_reactive_power_mean',
            'Voltage_mean', 'Global_intensity_mean', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
        ]
        
        # Target in kWh
        daily['Daily_energy_kWh'] = daily['Global_active_power_sum'] / 60.0
        total_wh = daily['Daily_energy_kWh'] * 1000.0
        sub_sum = daily['Sub_metering_1'] + daily['Sub_metering_2'] + daily['Sub_metering_3']
        daily['Sub_metering_remainder'] = np.maximum(0.0, total_wh - sub_sum)
        target_col = 'Daily_energy_kWh'
        
        summary = {
            'dataset_type': 'UCI Minute-Level Resampled',
            'initial_records': initial_records,
            'missing_records_filled': initial_missing,
            'clean_daily_records': len(daily),
            'start_date': str(daily.index.min().date()),
            'end_date': str(daily.index.max().date()),
            'target_column': target_col
        }

    # Case B: Daily Telemetry Dataset (with Weather, Occupancy, or Daily consumption)
    else:
        print("[INFO] Detected daily-level telemetry dataset...")
        # Find date column
        date_col = next((c for c in cols if 'date' in c.lower()), cols[0])
        df['Datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['Datetime'], keep='first')
        df = df.set_index('Datetime')
        
        # Find primary electricity target
        target_candidates = [
            'Previous electricity consumption (kWh)',
            'Daily_energy_kWh',
            'Electricity consumption (kWh)',
            'Consumption (kWh)',
            'consumption_kwh',
            'Power (kW)',
            'Appliance usage (kWh)'
        ]
        target_col = next((c for c in target_candidates if c in df.columns), None)
        if target_col is None:
            # Fallback to first float column
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    target_col = col
                    break
        if target_col is None:
            target_col = df.columns[1]
            
        # Standardize target column alias
        df['Daily_energy_kWh'] = pd.to_numeric(df[target_col], errors='coerce')
        df['Daily_energy_kWh'] = df['Daily_energy_kWh'].interpolate(method='linear').bfill().ffill()
        daily = df
        target_col = 'Daily_energy_kWh'
        
        summary = {
            'dataset_type': 'Daily Telemetry / Weather Multi-Feature',
            'initial_records': initial_records,
            'missing_records_filled': int(daily['Daily_energy_kWh'].isnull().sum()),
            'clean_daily_records': len(daily),
            'start_date': str(daily.index.min().date()),
            'end_date': str(daily.index.max().date()),
            'target_column': target_col
        }

    # Feature Engineering across all datasets
    daily['date'] = daily.index.date
    daily['day'] = daily.index.day
    daily['day_of_week'] = daily.index.dayofweek  # 0=Monday, 6=Sunday
    daily['day_name'] = daily.index.day_name()
    daily['is_weekend'] = daily['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    daily['month'] = daily.index.month
    daily['month_name'] = daily.index.strftime('%b')
    daily['year'] = daily.index.year
    daily['quarter'] = daily.index.quarter
    daily['day_of_year'] = daily.index.dayofyear
    
    # Lag Features (Lag 1, Lag 7, 7-day Rolling Mean)
    daily['Lag_1_kWh'] = daily[target_col].shift(1).bfill()
    daily['Lag_7_kWh'] = daily[target_col].shift(7).bfill()
    daily['Rolling_Mean_7'] = daily[target_col].rolling(window=7, min_periods=1).mean()
    
    daily = daily.dropna(subset=[target_col])
    return daily, summary, target_col


def create_sequences(
    values: np.ndarray,
    lookback_window: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates (X, y) sliding sequence pairs for LSTM training.
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)
        
    X, y = [], []
    for i in range(len(values) - lookback_window):
        X.append(values[i : i + lookback_window])
        y.append(values[i + lookback_window, 0])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


def prepare_time_series_data(
    daily_df: pd.DataFrame,
    target_col: str = 'Daily_energy_kWh',
    lookback_window: int = 30,
    train_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    Performs chronological train/test split, fits MinMaxScaler on train set only,
    and generates sequence pairs. Automatically adapts lookback window if dataset is compact.
    """
    total_len = len(daily_df)
    # Ensure lookback window is feasible
    if lookback_window >= total_len * 0.4:
        lookback_window = max(7, int(total_len * 0.15))
        print(f"[INFO] Adjusted lookback window to {lookback_window} days for dataset of size {total_len}.")
        
    train_size = int(total_len * train_ratio)
    train_df = daily_df.iloc[:train_size].copy()
    test_df = daily_df.iloc[train_size:].copy()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_values = train_df[[target_col]].values
    test_values = test_df[[target_col]].values
    
    scaler.fit(train_values)
    scaled_train = scaler.transform(train_values)
    scaled_test = scaler.transform(test_values)
    
    combined_test_values = np.vstack([
        scaled_train[-lookback_window:],
        scaled_test
    ])
    
    X_train, y_train = create_sequences(scaled_train, lookback_window=lookback_window)
    X_test, y_test = create_sequences(combined_test_values, lookback_window=lookback_window)
    
    val_split_idx = max(1, int(len(X_train) * 0.85))
    X_tr, y_tr = X_train[:val_split_idx], y_train[:val_split_idx]
    X_val, y_val = X_train[val_split_idx:], y_train[val_split_idx:]
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'scaler': scaler,
        'target_col': target_col,
        'lookback_window': lookback_window,
        'X_train_full': X_train,
        'y_train_full': y_train,
        'X_train': X_tr,
        'y_train': y_tr,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'test_dates': test_df.index,
        'actual_test_unscaled': test_values.flatten()
    }


# Backwards-compatible aliases
load_raw_data = detect_and_load_data
clean_data = lambda df: clean_and_prepare_daily(df)[:2]
resample_daily = lambda df: df if 'Daily_energy_kWh' in df.columns else clean_and_prepare_daily(df)[0]
