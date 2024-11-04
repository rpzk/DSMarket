from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import timedelta
from typing import Optional

app = FastAPI()

# Configuração de templates e CORS
templates = Jinja2Templates(directory="app/templates")  # Ajuste o caminho conforme necessário
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diretório de modelos
MODELS_DIR = 'forecasting/models'

class ForecastRequest(BaseModel):
    store: str
    item: str
    steps: int

def load_model(store, item, model_type):
    # Define o caminho baseado no tipo do modelo
    model_path = os.path.join(MODELS_DIR, f"{model_type}_model_{store}_{item}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo '{model_type}' não encontrado para loja {store} e item {item}.")
    
    # Carrega o modelo usando joblib
    model = joblib.load(model_path)
    return model

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/last_data_date/")
async def get_last_data_date():
    # Defina a última data de dados aqui. Exemplo de última data: '2016-12-31'
    last_data_date = datetime(2016, 12, 31)
    return {"last_data_date": last_data_date.strftime('%Y-%m-%d')}

@app.post("/forecast/")
async def get_forecast(forecast_request: ForecastRequest):
    store = forecast_request.store.replace(" ", "_")
    item = forecast_request.item.replace(" ", "_")
    steps = forecast_request.steps
    
    try:
        last_date = pd.Timestamp("2016-12-31")  # Defina a última data dos dados reais

        # Tenta carregar o modelo Prophet
        try:
            model_type = "prophet"
            model = load_model(store, item, model_type)
            
            # Cria as datas futuras e adiciona os regressores e limites
            future_dates = pd.DataFrame({'ds': [last_date + timedelta(days=i) for i in range(1, steps + 1)]})
            future_dates['is_christmas'] = future_dates['ds'].apply(lambda x: 1 if x.month == 12 and x.day == 25 else 0)
            future_dates['is_holiday'] = future_dates['ds'].apply(
                lambda x: 1 if (x.month == 12 and x.day == 25) or (x.month == 1 and x.day == 1) else 0
            )
            future_dates['day_of_week'] = future_dates['ds'].apply(lambda x: x.weekday())
            future_dates['is_weekend'] = future_dates['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
            
            # Define floor e cap
            future_dates['floor'] = 0
            future_dates['cap'] = model.history['y'].max()  # Assume o valor máximo de 'y' no histórico como limite superior

            # Fazer a previsão
            forecast = model.predict(future_dates)
            forecast_values = forecast['yhat'].tolist()
        
        except FileNotFoundError:
            # Tenta carregar o modelo XGBoost se Prophet não for encontrado
            model_type = "xgboost"
            model = load_model(store, item, model_type)
            future_data = pd.DataFrame({"day_of_week": [(last_date + timedelta(days=i)).weekday() for i in range(1, steps + 1)]})
            forecast_values = model.predict(future_data).tolist()
        
        return {"store": store, "item": item, "model_type": model_type, "forecast": forecast_values}
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")
