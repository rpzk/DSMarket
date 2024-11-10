# forecasting/utils/metrics_storage.py

import os

def save_metrics(store, item, model_type, rmse, mse, mae, mape, r2, smape, mase, params, metrics_file_path="metrics/model_metrics.csv"):
    metrics_dir = os.path.dirname(metrics_file_path)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Extrai os parâmetros específicos do modelo, deixando em branco os que não se aplicam
    changepoint_prior_scale = params.get('changepoint_prior_scale', '')
    seasonality_mode = params.get('seasonality_mode', '')
    arima_order = params.get('order', '')
    arima_seasonal_order = params.get('seasonal_order', '')

    # Escreve todas as métricas e parâmetros no arquivo com o número de colunas fixo
    with open(metrics_file_path, 'a') as f:
        f.write(f"{store},{item},{model_type},{rmse},{mse},{mae},{mape},{r2},{smape},{mase},"
                f"{changepoint_prior_scale},{seasonality_mode},{arima_order},{arima_seasonal_order}\n")

