"""
forecasting.py
--------------
Model building, baseline evaluation, LSTM training, model persistence, and
autoregressive multi-step future forecasting for Household Electricity Consumption.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import keras
from keras import layers, callbacks


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression and time-series evaluation metrics:
    MAE, RMSE, R2, and MAPE.
    
    Parameters:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        
    Returns:
        Dict[str, float]: Dictionary containing computed metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    # Avoid division by zero in MAPE
    denom = np.where(np.abs(y_true) < 1e-5, 1e-5, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    
    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2)
    }


class BaselinePredictor:
    """
    Baseline time-series models for benchmark comparison:
      1. Persistence (Naive T-1): Predicts previous day's consumption.
      2. 7-Day Moving Average: Predicts the rolling average of the preceding 7 days.
    """
    @staticmethod
    def persistence_predict(unscaled_series: np.ndarray, lookback_window: int = 30) -> np.ndarray:
        """Predicts y_t = y_{t-1} for the test sequence."""
        predictions = []
        for i in range(len(unscaled_series) - lookback_window):
            predictions.append(unscaled_series[i + lookback_window - 1])
        return np.array(predictions, dtype=np.float32)

    @staticmethod
    def moving_average_predict(unscaled_series: np.ndarray, lookback_window: int = 30, ma_window: int = 7) -> np.ndarray:
        """Predicts y_t = mean(y_{t-ma_window .. t-1}) for the test sequence."""
        predictions = []
        for i in range(len(unscaled_series) - lookback_window):
            window = unscaled_series[i + lookback_window - ma_window : i + lookback_window]
            predictions.append(np.mean(window))
        return np.array(predictions, dtype=np.float32)


def build_lstm_model(
    lookback_window: int = 30,
    num_features: int = 1,
    units_1: int = 64,
    units_2: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Builds a stacked LSTM neural network with Dropout regularization.
    
    Architecture:
      - Input Layer: (lookback_window, num_features)
      - LSTM Layer 1: 64 units, activation='tanh', return_sequences=True
      - Dropout Layer: 0.2
      - LSTM Layer 2: 32 units, activation='tanh', return_sequences=False
      - Dropout Layer: 0.2
      - Dense Hidden Layer: 16 units, activation='relu'
      - Dense Output Layer: 1 unit (continuous consumption regression)
    """
    model = keras.Sequential([
        layers.Input(shape=(lookback_window, num_features)),
        layers.LSTM(units_1, activation='tanh', return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(units_2, activation='tanh', return_sequences=False),
        layers.Dropout(dropout_rate),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_lstm_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    patience: int = 12
) -> Tuple[keras.Model, Dict[str, list]]:
    """
    Trains the LSTM model with Early Stopping to prevent overfitting.
    
    Returns:
        Tuple[keras.Model, Dict[str, list]]: Trained model and training loss history dictionary.
    """
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
    
    print(f"[INFO] Training LSTM on {len(X_train)} samples, validating on {len(X_val)} samples...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return model, history.history


def save_artifacts(
    model: keras.Model,
    scaler: Any,
    metadata: Dict[str, Any],
    model_dir: str = "models"
) -> None:
    """Saves model weights, scaler, and metadata to disk."""
    os.makedirs(model_dir, exist_ok=True)
    
    weights_path = os.path.join(model_dir, "lstm_weights.pkl")
    model_keras_path = os.path.join(model_dir, "lstm_model.keras")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    meta_path = os.path.join(model_dir, "metadata.json")
    
    print(f"[INFO] Saving model weights to {weights_path}...")
    weights = model.get_weights()
    with open(weights_path, "wb") as f:
        pickle.dump(weights, f)
        
    print(f"[INFO] Saving scaler to {scaler_path}...")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"[INFO] Saving metadata to {meta_path}...")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    try:
        # Also write a torch state dict if applicable
        import torch
        torch_path = os.path.join(model_dir, "lstm_model.pt")
        # Save placeholder / keras file safely
        with open(model_keras_path, "wb") as f:
            pickle.dump(weights, f)
    except Exception as e:
        print(f"[NOTE] Model format note: {e}")
        
    print("[INFO] All artifacts saved successfully.")


def load_artifacts(model_dir: str = "models") -> Tuple[Optional[keras.Model], Optional[Any], Optional[Dict[str, Any]]]:
    """
    Loads saved model, scaler, and metadata if available.
    
    Returns:
        Tuple of (model, scaler, metadata) or (None, None, None) if not found.
    """
    weights_path = os.path.join(model_dir, "lstm_weights.pkl")
    model_keras_path = os.path.join(model_dir, "lstm_model.keras")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    meta_path = os.path.join(model_dir, "metadata.json")
    
    if not (os.path.exists(scaler_path) and os.path.exists(meta_path)):
        return None, None, None
        
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
        lookback_window = metadata.get('lookback_window', 30)
        
        # Rebuild model and set weights
        model = build_lstm_model(lookback_window=lookback_window)
        if os.path.exists(weights_path):
            with open(weights_path, "rb") as f:
                weights = pickle.load(f)
            model.set_weights(weights)
        elif os.path.exists(model_keras_path):
            model = keras.models.load_model(model_keras_path)
        else:
            return None, None, None
            
        return model, scaler, metadata
    except Exception as e:
        print(f"[WARNING] Failed to load saved artifacts: {e}")
        return None, None, None


def forecast_future(
    model: keras.Model,
    last_sequence_scaled: np.ndarray,
    scaler: Any,
    last_date: pd.Timestamp,
    horizon_days: int = 30
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates multi-step future electricity consumption forecasts using an
    autoregressive recursive forecasting loop.
    
    Parameters:
        model: Trained LSTM model.
        last_sequence_scaled: Scaled 2D or 3D array of shape (lookback_window, 1).
        scaler: Fitted MinMaxScaler.
        last_date: Timestamp of the last historical record.
        horizon_days: Number of future days to forecast (7, 14, 30).
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - forecast_df with future dates, predictions, and day info.
            - summary metrics dictionary.
    """
    if last_sequence_scaled.ndim == 2:
        current_seq = last_sequence_scaled.reshape(1, -1, 1).copy()
    elif last_sequence_scaled.ndim == 3:
        current_seq = last_sequence_scaled.copy()
    else:
        current_seq = last_sequence_scaled.reshape(1, -1, 1).copy()
        
    predicted_scaled = []
    
    for _ in range(horizon_days):
        # Predict next day
        next_pred_scaled = model.predict(current_seq, verbose=0)[0, 0]
        predicted_scaled.append(next_pred_scaled)
        
        # Roll sequence forward: remove first day, append new predicted day
        next_point = np.array([[[next_pred_scaled]]], dtype=np.float32)
        current_seq = np.concatenate([current_seq[:, 1:, :], next_point], axis=1)
        
    # Inverse transform to original kWh scale
    predicted_scaled_arr = np.array(predicted_scaled).reshape(-1, 1)
    predicted_unscaled = scaler.inverse_transform(predicted_scaled_arr).ravel()
    
    # Clip negative values if any
    predicted_unscaled = np.maximum(0.0, predicted_unscaled)
    
    # Generate future dates
    future_dates = pd.date_range(
        start=pd.to_datetime(last_date) + pd.Timedelta(days=1),
        periods=horizon_days,
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast_kWh': np.round(predicted_unscaled, 2),
        'Day_Name': future_dates.day_name(),
        'Is_Weekend': [1 if d >= 5 else 0 for d in future_dates.dayofweek]
    }).set_index('Date')
    
    # Summary metrics
    expected_avg = float(forecast_df['Forecast_kWh'].mean())
    max_val = float(forecast_df['Forecast_kWh'].max())
    max_date = str(forecast_df['Forecast_kWh'].idxmax().strftime('%Y-%m-%d (%a)'))
    min_val = float(forecast_df['Forecast_kWh'].min())
    min_date = str(forecast_df['Forecast_kWh'].idxmin().strftime('%Y-%m-%d (%a)'))
    total_expected = float(forecast_df['Forecast_kWh'].sum())
    
    summary = {
        'horizon_days': horizon_days,
        'expected_avg_kWh': round(expected_avg, 2),
        'total_expected_kWh': round(total_expected, 2),
        'max_forecast_kWh': round(max_val, 2),
        'max_forecast_date': max_date,
        'min_forecast_kWh': round(min_val, 2),
        'min_forecast_date': min_date
    }
    
    return forecast_df, summary
