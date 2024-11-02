# forecasting/model_storage.py

import os
import joblib
import logging

def save_trained_model(model, model_type, store, item, models_dir):
    # Criar diretório de modelos, se não existir
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Identificar o tipo de modelo para o nome do arquivo
    if model_type == "Prophet":
        model_filename = f"prophet_model_{store}_{item}.pkl"
    elif model_type == "SARIMA":
        model_filename = f"sarima_model_{store}_{item}.pkl"
    elif model_type == "XGBoost":
        model_filename = f"xgboost_model_{store}_{item}.pkl"
    else:
        model_filename = f"model_{store}_{item}.pkl"

    # Salvar o modelo
    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(model, model_path)
    logging.info(f"Modelo {model_type} salvo em {model_path}")

def save_forecast_results(data, forecasts_dir):
    # Criar diretório de previsões, se não existir
    if not os.path.exists(forecasts_dir):
        os.makedirs(forecasts_dir)
    forecast_file = os.path.join(forecasts_dir, "all_forecasts.csv")
    data.to_csv(forecast_file, index=False)
    logging.info(f"Previsões salvas em {forecast_file}")
