# model_evaluation.py

import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate sMAPE with handling for zero denominators"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / np.where(denominator == 0, 1, denominator))

def mean_absolute_scaled_error(y_true, y_pred):
    """Calculate MASE with naive forecast (lag-1)"""
    naive_forecast = np.roll(y_true, 1)[1:]  # Shifted by 1 to create lag-1 forecast
    mae_naive = np.mean(np.abs(y_true[1:] - naive_forecast))  # Mean Absolute Error of naive forecast
    mae_model = mean_absolute_error(y_true, y_pred)
    return mae_model / mae_naive if mae_naive != 0 else None  # Avoid division by zero

def calculate_metrics(y_true, y_pred, model_type):
    if y_pred is None or len(y_pred) == 0:
        logging.error(f"Não há previsões para calcular métricas para {model_type}.")
        return None
    
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)
        smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        mase = mean_absolute_scaled_error(y_true, y_pred)

        metrics = {
            'model_type': model_type,
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'smape': smape,
            'mase': mase
        }
        
        logging.info(f"Métricas para {model_type}: RMSE={rmse}, MSE={mse}, MAE={mae}, MAPE={mape}%, R²={r2}, sMAPE={smape}%, MASE={mase}")
        return metrics
    except Exception as e:
        logging.error(f"Erro ao calcular métricas para {model_type}: {e}")
        return None

