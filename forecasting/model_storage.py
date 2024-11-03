# forecasting/model_storage.py

import os
import joblib
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import pandas as pd

def save_trained_model(model, model_type, store, item, models_dir='forecasting/models'):
    """
    Salva um modelo treinado no diretório especificado.

    Args:
        model: O modelo a ser salvo.
        model_type (str): Tipo do modelo (e.g., 'Prophet', 'SARIMA', 'XGBoost').
        store (str): Identificador da loja.
        item (str): Identificador do item.
        models_dir (str): Diretório onde os modelos serão salvos.
    """
    # Criar diretório de modelos, se não existir
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Nome do arquivo com base no tipo do modelo
    if model_type.lower() == "prophet":
        model_filename = f"prophet_model_{store}_{item}.pkl"
    elif model_type.lower() == "sarima":
        model_filename = f"sarima_model_{store}_{item}.pkl"
    elif model_type.lower() == "xgboost":
        model_filename = f"xgboost_model_{store}_{item}.pkl"
    else:
        model_filename = f"model_{store}_{item}.pkl"

    # Caminho completo do arquivo
    model_path = os.path.join(models_dir, model_filename)

    # Salvar conforme o tipo do modelo
    try:
        if model_type.lower() == "sarima":
            model.save(model_path)  # Para SARIMA, utilize o método save
        else:
            joblib.dump(model, model_path)  # Para outros tipos, utilize joblib
        logging.info(f"Modelo {model_type} salvo em {model_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar o modelo {model_type} para {store}, {item}: {e}")

def save_forecast_results(data, forecasts_dir='forecasting/forecasts'):
    """
    Salva os resultados de previsão em um arquivo CSV no diretório especificado.

    Args:
        data (pd.DataFrame): DataFrame contendo as previsões a serem salvas.
        forecasts_dir (str): Diretório onde as previsões serão salvas.
    """
    # Criar diretório de previsões, se não existir
    if not os.path.exists(forecasts_dir):
        os.makedirs(forecasts_dir)

    forecast_file = os.path.join(forecasts_dir, "all_forecasts.csv")

    try:
        data.to_csv(forecast_file, index=False)
        logging.info(f"Previsões salvas em {forecast_file}")
    except Exception as e:
        logging.error(f"Erro ao salvar previsões: {e}")

