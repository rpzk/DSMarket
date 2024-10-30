from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import logging
from datetime import date, timedelta
from typing import Optional

app = FastAPI()

# Configuração do logging
logging.basicConfig(level=logging.INFO)

# Habilitar CORS
origins = [
    "*"  # Permite todas as origens. Para produção, especifique os domínios permitidos.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obter o diretório atual deste arquivo (api.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construir o caminho absoluto para o diretório 'models'
models_dir = os.path.join(current_dir, 'forecasting/models')
models_path = os.path.join(models_dir, 'trained_models.pkl')

# Carregar os modelos treinados
if os.path.exists(models_path):
    models = joblib.load(models_path)
else:
    logging.error(f"O arquivo de modelos treinados não foi encontrado em {models_path}.")
    models = {}

# Carregar modelos SARIMA individualmente, se necessário
for key, model_path in models.items():
    if isinstance(model_path, str) and model_path.endswith('.pkl'):
        full_model_path = os.path.join(models_dir, os.path.basename(model_path))
        if os.path.exists(full_model_path):
            if 'sarima' in model_path.lower():
                models[key] = SARIMAXResults.load(full_model_path)
            else:
                models[key] = joblib.load(full_model_path)
        else:
            logging.error(f"Modelo não encontrado: {full_model_path}")

class ForecastRequest(BaseModel):
    store: str
    item: str
    periods: int = 28  # Horizonte de previsão
    start_date: Optional[date] = None  # Data inicial da previsão (opcional)

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de Previsão de Vendas!"}

@app.get("/last_date/")
def get_last_date(store: str, item: str):
    model_key = (store, item)
    model = models.get(model_key)

    if model is None:
        logging.error(f"Modelo não encontrado para a loja {store} e item {item}.")
        raise HTTPException(status_code=404, detail="Modelo não encontrado para a loja e item especificados.")

    if isinstance(model, Prophet):
        last_date = model.history['ds'].max()
    elif isinstance(model, SARIMAXResults):
        last_date = model.data.dates[-1]
    else:
        logging.error("Tipo de modelo desconhecido.")
        raise HTTPException(status_code=500, detail="Tipo de modelo desconhecido.")

    return {"last_date": last_date.strftime('%Y-%m-%d')}

@app.post("/forecast/")
def get_forecast(forecast_request: ForecastRequest):
    try:
        store = forecast_request.store
        item = forecast_request.item
        periods = forecast_request.periods
        start_date = forecast_request.start_date

        logging.info(f"Dados recebidos: Loja={store}, Item={item}, Períodos={periods}, Data Inicial={start_date}")

        if not store or not item or not isinstance(periods, int):
            return {"error": "Dados inválidos fornecidos."}

        model_key = (store, item)
        model = models.get(model_key)

        if model is None:
            logging.error(f"Modelo não encontrado para a loja {store} e item {item}.")
            return {"error": "Modelo não encontrado para a loja e item especificados."}

        if isinstance(model, Prophet):
            last_date = model.history['ds'].max()
        elif isinstance(model, SARIMAXResults):
            last_date = model.data.dates[-1]
        else:
            logging.error("Tipo de modelo desconhecido.")
            return {"error": "Tipo de modelo desconhecido."}

        # Se a data inicial não for fornecida, usar o dia seguinte à última data de treinamento
        if not start_date:
            future_start = last_date + timedelta(days=1)
        else:
            future_start = pd.to_datetime(start_date)
            if future_start <= last_date:
                future_start = last_date + timedelta(days=1)
                logging.warning("Data inicial fornecida é anterior ou igual à última data de treinamento. Usando o dia seguinte à última data.")

        # Calcular o número de períodos necessários
        total_days = (future_start - (last_date + timedelta(days=1))).days + periods
        steps = total_days if total_days > 0 else periods

        # Gerar previsões
        if isinstance(model, Prophet):
            future_dates = pd.date_range(start=future_start, periods=periods, freq='D')
            future = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future)
            forecast_values = forecast['yhat'].values
        elif isinstance(model, SARIMAXResults):
            forecast_values = model.forecast(steps=steps)[-periods:]
        else:
            logging.error("Tipo de modelo desconhecido.")
            return {"error": "Tipo de modelo desconhecido."}

        dates = pd.date_range(start=future_start, periods=periods, freq='D')

        forecast_df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'store': store,
            'item': item,
            'forecast_sales': forecast_values
        })
        forecast_list = forecast_df.to_dict(orient='records')
        return {"forecast": forecast_list}

    except Exception as e:
        logging.error(f"Erro ao gerar a previsão: {e}")
        return {"error": f"Ocorreu um erro ao gerar a previsão: {str(e)}"}
