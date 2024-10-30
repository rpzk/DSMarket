# forecasting/model_evaluation.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calculate_metrics(actual, forecast):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    # Para calcular MAPE, evitar divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((actual - forecast) / actual)
        mape = np.where(np.isfinite(mape), mape, np.nan)
        mape = np.nanmean(mape) * 100
    return rmse, mae, mape
