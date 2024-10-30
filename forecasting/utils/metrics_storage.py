# forecasting/utils/metrics_storage.py

import os

def save_metrics(store, item, model_type, rmse, mae, mape, params, metrics_file_path="metrics/model_metrics.csv"):
    metrics_dir = os.path.dirname(metrics_file_path)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    with open(metrics_file_path, 'a') as f:
        # Adiciona parâmetros adicionais conforme necessário
        changepoint_prior_scale = params.get('changepoint_prior_scale', '')
        seasonality_mode = params.get('seasonality_mode', '')
        daily_seasonality = params.get('daily_seasonality', '')
        weekly_seasonality = params.get('weekly_seasonality', '')
        yearly_seasonality = params.get('yearly_seasonality', '')
        arima_order = params.get('order', '')
        arima_seasonal_order = params.get('seasonal_order', '')

        f.write(f"{store},{item},{model_type},{rmse},{mae},{mape},{changepoint_prior_scale},{seasonality_mode},{daily_seasonality},{weekly_seasonality},{yearly_seasonality},{arima_order},{arima_seasonal_order}\n")
