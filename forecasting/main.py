# forecasting/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import logging

app = FastAPI()

# Configuração do logging
logging.basicConfig(level=logging.INFO)

# Obter o diretório atual deste arquivo (api.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construir o caminho absoluto para o arquivo 'trained_models.pkl'
models_path = os.path.join(current_dir, 'models', 'trained_models.pkl')

# Carregar os modelos treinados
if os.path.exists(models_path):
    models = joblib.load(models_path)
    logging.info(f"Modelos carregados com sucesso de {models_path}")
else:
    logging.error(f"O arquivo de modelos treinados não foi encontrado em {models_path}.")
    models = {}

# Exibir o conteúdo de models
logging.info(f"Conteúdo de models: {models}")

# Carregar modelos individuais
for key, model_path in models.items():
    if isinstance(model_path, str) and model_path.endswith('.pkl'):
        full_model_path = os.path.join(current_dir, model_path)
        if os.path.exists(full_model_path):
            if 'sarima' in model_path.lower():
                # Carregar modelo SARIMA
                models[key] = SARIMAXResults.load(full_model_path)
                logging.info(f"Modelo SARIMA carregado para {key}")
            else:
                # Carregar modelo Prophet
                models[key] = joblib.load(full_model_path)
                logging.info(f"Modelo Prophet carregado para {key}")
        else:
            logging.error(f"Modelo não encontrado: {full_model_path}")
    else:
        logging.error(f"Caminho do modelo inválido para {key}: {model_path}")

class ForecastRequest(BaseModel):
    store: str
    item: str
    periods: int = 28  # Horizonte de previsão

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de Previsão de Vendas!"}

@app.post("/forecast/")
def get_forecast(request: ForecastRequest):
    store = request.store
    item = request.item
    periods = request.periods

    # Recuperar o modelo correspondente
    model = models.get((store, item))
    if model is None:
        return {"error": "Modelo não encontrado para a loja e item especificados."}

    try:
        # Gerar previsões
        if isinstance(model, Prophet):
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            forecast_values = forecast['yhat'][-periods:].values
        elif isinstance(model, SARIMAXResults):
            forecast_values = model.forecast(steps=periods)
        else:
            return {"error": "Tipo de modelo desconhecido."}

        # Preparar a resposta
        dates = pd.date_range(start=pd.Timestamp.today(), periods=periods)
        forecast_df = pd.DataFrame({
            'date': dates,
            'store': store,
            'item': item,
            'forecast_sales': forecast_values
        })
        forecast_list = forecast_df.to_dict(orient='records')
        return {"forecast": forecast_list}
    except Exception as e:
        logging.error(f"Erro ao gerar a previsão: {e}")
        return {"error": f"Ocorreu um erro ao gerar a previsão: {str(e)}"}
