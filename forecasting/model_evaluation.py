# model_evaluation.py

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import logging

def calculate_metrics(y_true, y_pred, model_type):
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        metrics = {
            'model_type': model_type,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        logging.info(f"Métricas para {model_type}: RMSE={rmse}, MAE={mae}, MAPE={mape}%")
        return metrics
    except Exception as e:
        logging.error(f"Erro ao calcular métricas para {model_type}: {e}")
        return None
