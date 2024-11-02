# app/api.py

# Importações padrão e externas
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import logging
from datetime import date, timedelta
from typing import Optional

# Instanciar a aplicação FastAPI
app = FastAPI()

# Configuração de templates e CORS
templates = Jinja2Templates(directory="app/templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens. Para produção, especifique os domínios permitidos.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração do logging
logging.basicConfig(level=logging.INFO)

# Caminho do diretório de modelos
# Ajustado para apontar para a pasta correta
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'forecasting/models')

# Carregar todos os modelos Prophet e SARIMA na pasta 'forecasting/models'
models = {}
for model_file in os.listdir(models_dir):
    if model_file.endswith('.pkl'):
        model_path = os.path.join(models_dir, model_file)
        # Extrair as informações de loja e item a partir do nome do arquivo
        # Exemplo: "prophet_model_Harlem_SUPERMARKET_3_586.pkl" -> ("Harlem", "SUPERMARKET_3_586")
        store_item_key = model_file.replace('prophet_model_', '').replace('.pkl', '')
        store, item = store_item_key.split('_', 1)  # Dividir no primeiro underscore apenas
        models[(store, item)] = SARIMAXResults.load(model_path) if 'sarima' in model_file.lower() else joblib.load(model_path)

# Definir o esquema de entrada para previsão
class ForecastRequest(BaseModel):
    store: str
    item: str
    periods: int = 28  # Horizonte de previsão
    start_date: Optional[date] = None  # Data inicial da previsão (opcional)

# Rota principal
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Rota para obter a última data do modelo
@app.get("/last_date/")
def get_last_date(store: str, item: str):
    model_key = (store, item)
    model = models.get(model_key)

    if not model:
        logging.error(f"Modelo não encontrado para a loja {store} e item {item}.")
        raise HTTPException(status_code=404, detail="Modelo não encontrado para a loja e item especificados.")

    # Determinar a última data baseada no tipo de modelo
    if isinstance(model, Prophet):
        last_date = model.history['ds'].max()
    elif isinstance(model, SARIMAXResults):
        last_date = model.data.dates[-1]
    else:
        logging.error("Tipo de modelo desconhecido.")
        raise HTTPException(status_code=500, detail="Tipo de modelo desconhecido.")

    return {"last_date": last_date.strftime('%Y-%m-%d')}

# Rota para gerar previsões
@app.post("/forecast/")
def get_forecast(forecast_request: ForecastRequest):
    try:
        store, item, periods, start_date = (
            forecast_request.store,
            forecast_request.item,
            forecast_request.periods,
            forecast_request.start_date,
        )

        logging.info(f"Dados recebidos: Loja={store}, Item={item}, Períodos={periods}, Data Inicial={start_date}")

        model_key = (store, item)
        model = models.get(model_key)
        if not model:
            logging.error(f"Modelo não encontrado para a loja {store} e item {item}.")
            return {"error": "Modelo não encontrado para a loja e item especificados."}

        # Determinar a última data baseada no tipo de modelo
        last_date = model.history['ds'].max() if isinstance(model, Prophet) else model.data.dates[-1]

        # Ajustar data inicial de previsão
        future_start = pd.to_datetime(start_date or (last_date + timedelta(days=1)))
        if future_start <= last_date:
            future_start = last_date + timedelta(days=1)
            logging.warning("Data inicial fornecida é anterior ou igual à última data de treinamento. Usando o dia seguinte à última data.")

        # Calcular períodos necessários
        total_days = max((future_start - (last_date + timedelta(days=1))).days + periods, periods)

        # Gerar previsões
        if isinstance(model, Prophet):
            future_dates = pd.date_range(start=future_start, periods=periods, freq='D')
            future = pd.DataFrame({'ds': future_dates})
            
            # Adicionar os regressors 'is_christmas', 'is_holiday' e 'day_of_week' ao DataFrame
            future['is_christmas'] = future['ds'].apply(lambda x: 1 if x.month == 12 and x.day == 25 else 0)
            future['is_holiday'] = future['ds'].apply(lambda x: 1 if (x.month == 12 and x.day == 25) or (x.month == 1 and x.day == 1) else 0)
            future['day_of_week'] = future['ds'].apply(lambda x: x.weekday())  # 0=Monday, 6=Sunday
            future['is_weekend'] = future['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)  # 1 for Saturday and Sunday, 0 otherwise
            
            forecast = model.predict(future)
            forecast_values = forecast['yhat'].values
        else:
            logging.error("Tipo de modelo desconhecido.")
            return {"error": "Tipo de modelo desconhecido."}

        # Organizar o resultado
        dates = pd.date_range(start=future_start, periods=periods, freq='D')
        forecast_df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'store': store,
            'item': item,
            'forecast_sales': forecast_values
        })

        return {"forecast": forecast_df.to_dict(orient='records')}

    except Exception as e:
        logging.error(f"Erro ao gerar a previsão: {e}")
        return {"error": f"Ocorreu um erro ao gerar a previsão: {str(e)}"}
