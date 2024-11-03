# app/api.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import os
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import logging
from datetime import date, timedelta
from enum import Enum

# Instanciar a aplicação FastAPI
app = FastAPI()

# Configuração de templates e CORS
templates = Jinja2Templates(directory="app/templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para produção, especifique os domínios permitidos.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração do logging
logging.basicConfig(level=logging.INFO)

# Caminho do diretório de modelos
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'forecasting/models')

# Função para extrair o nome principal da loja
def extract_main_store_name(full_store_name: str) -> str:
    """
    Extrai o nome principal da loja removendo o sufixo 'SUPERMARKET X'.
    Exemplo: 'Back Bay SUPERMARKET 3' -> 'Back Bay'
    """
    return full_store_name.replace("_", " ").split(" SUPERMARKET")[0].strip()

# Função para extrair o código do item
def extract_item_code(full_item: str) -> str:
    """
    Extrai o código do item removendo quaisquer prefixos.
    Exemplo: 'SUPERMARKET_3_090' -> '090'
    """
    return full_item.split('_')[-1].strip()

# Definir um Enum para os tipos de modelos disponíveis
class ModelType(str, Enum):
    prophet = "prophet"
    sarima = "sarima"
    xgboost = "xgboost"

# Definir a estrutura de armazenamento para cada tipo de modelo
models = {
    'prophet': {},
    'sarima': {},
    'xgboost': {}
}

# Carregar todos os modelos Prophet, SARIMA e XGBoost na pasta 'forecasting/models'
for model_type in models.keys():
    model_type_dir = os.path.join(models_dir, model_type)
    if not os.path.exists(model_type_dir):
        logging.warning(f"Diretório para modelos {model_type} não existe: {model_type_dir}")
        continue

    for model_file in os.listdir(model_type_dir):
        if model_file.endswith('.pkl'):
            model_path = os.path.join(model_type_dir, model_file)
            try:
                # Extrair detalhes de loja e item
                _, store, item = model_file.replace('.pkl', '').split('_model_')
                store_main = extract_main_store_name(store)
                item_code = extract_item_code(item)

                # Carregar o modelo conforme o tipo
                if model_type == 'sarima':
                    model = SARIMAXResults.load(model_path)
                else:
                    model = joblib.load(model_path)

                # Carregar a última data
                last_date_path = model_path.replace('.pkl', '_last_date.txt')
                if os.path.exists(last_date_path):
                    with open(last_date_path, 'r') as f:
                        last_date_str = f.read().strip()
                        last_date = pd.to_datetime(last_date_str)
                else:
                    logging.error(f"Última data para o modelo {model_file} não está disponível.")
                    last_date = None

                models[model_type][(store_main, item_code)] = {
                    'model': model,
                    'last_date': last_date
                }

                logging.info(f"Modelo carregado: Tipo={model_type}, Loja={store_main}, Item={item_code}")
            except Exception as e:
                logging.error(f"Erro ao carregar o modelo {model_file}: {e}")

logging.info(f"Modelos carregados: {models}")

# Definir o esquema de entrada para previsão
class ForecastRequest(BaseModel):
    store: str
    item: str
    model_type: ModelType  # Especifica o tipo de modelo
    periods: int = 28  # Horizonte de previsão
    start_date: Optional[date] = None  # Data inicial da previsão (opcional)

# Rota principal
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Rota para obter a última data do modelo
@app.get("/last_date/")
def get_last_date(store: str, item: str, model_type: ModelType):
    # Extrair o nome principal da loja
    store_main = extract_main_store_name(store)

    # Extrair o código do item
    item_code = extract_item_code(item)

    logging.debug(f"Store main: {store_main}, Item code: {item_code}, Tipo Modelo: {model_type}")

    model_entry = models.get(model_type).get((store_main, item_code))

    if not model_entry:
        logging.error(f"Modelo {model_type} não encontrado para a loja {store_main} e item {item_code}.")
        raise HTTPException(status_code=404, detail="Modelo não encontrado para a loja e item especificados.")

    last_date = model_entry['last_date']
    if not last_date:
        logging.error(f"Última data para o modelo {model_type} ({store_main}, {item_code}) não disponível.")
        raise HTTPException(status_code=500, detail="Última data do modelo não disponível.")

    logging.info(f"Última data para ({store_main}, {item_code}) com modelo {model_type}: {last_date.strftime('%Y-%m-%d')}")
    return {"last_date": last_date.strftime('%Y-%m-%d')}

# Rota para gerar previsões
@app.post("/forecast/")
def get_forecast(forecast_request: ForecastRequest):
    try:
        # Ajuste do store e item para o formato correto
        store_main = extract_main_store_name(forecast_request.store)
        item_code = extract_item_code(forecast_request.item)
        periods = forecast_request.periods
        start_date = forecast_request.start_date
        model_type = forecast_request.model_type

        logging.info(f"Dados recebidos: Loja={store_main}, Item={item_code}, Tipo Modelo={model_type}, Períodos={periods}, Data Inicial={start_date}")

        model_entry = models.get(model_type).get((store_main, item_code))

        if not model_entry:
            available_stores = sorted({k[0] for k in models[model_type].keys() if k[1] == item_code})
            available_items = sorted({k[1] for k in models[model_type].keys() if k[0] == store_main})
            logging.error(f"Modelo {model_type} não encontrado para a loja {store_main} e item {item_code}.")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Modelo {model_type} não encontrado para a loja e item especificados.",
                    "available_stores_for_item": available_stores,
                    "available_items_for_store": available_items
                }
            )

        model = model_entry['model']
        last_date = model_entry['last_date']

        if not last_date:
            logging.error(f"Última data para o modelo {model_type} ({store_main}, {item_code}) não disponível.")
            raise HTTPException(status_code=500, detail="Última data do modelo não disponível.")

        logging.info(f"Última data para ({store_main}, {item_code}) com modelo {model_type}: {last_date.strftime('%Y-%m-%d')}")

        # Ajustar data inicial de previsão
        future_start = pd.to_datetime(start_date or (last_date + timedelta(days=1)))
        if future_start <= last_date:
            future_start = last_date + timedelta(days=1)
            logging.warning("Data inicial fornecida é anterior ou igual à última data de treinamento. Usando o dia seguinte à última data.")

        # Gerar previsões com base no tipo de modelo
        if model_type == 'prophet':
            future_dates = pd.date_range(start=future_start, periods=periods, freq='D')
            future = pd.DataFrame({'ds': future_dates})

            # Adicionar os regressors 'is_christmas', 'is_holiday', 'day_of_week' e 'is_weekend' ao DataFrame
            future['is_christmas'] = future['ds'].apply(lambda x: 1 if x.month == 12 and x.day == 25 else 0)
            future['is_holiday'] = future['ds'].apply(
                lambda x: 1 if (x.month == 12 and x.day == 25) or (x.month == 1 and x.day == 1) else 0
            )
            future['day_of_week'] = future['ds'].apply(lambda x: x.weekday())  # 0=Monday, 6=Sunday
            future['is_weekend'] = future['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)  # 1 para sábado e domingo

            forecast = model.predict(future)
            forecast_values = forecast['yhat'].values
        elif model_type == 'sarima':
            # Para SARIMA, precisamos das variáveis exógenas
            future_dates = pd.date_range(start=future_start, periods=periods, freq='D')
            exog = {
                'is_christmas': [1 if (x.month == 12 and x.day == 25) else 0 for x in future_dates],
                'is_holiday': [1 if (x.month == 12 and x.day == 25) or (x.month == 1 and x.day == 1) else 0 for x in future_dates],
                'day_of_week': [x.weekday() for x in future_dates],
                'is_weekend': [1 if x.weekday() >= 5 else 0 for x in future_dates]
            }
            exog_df = pd.DataFrame(exog)

            forecast_values = model.predict(n_periods=periods, exog=exog_df)
            forecast_values = [max(0, val) for val in forecast_values]  # Corrige valores negativos para zero
        elif model_type == 'xgboost':
            # Para XGBoost, precisamos das variáveis exógenas
            future_dates = pd.date_range(start=future_start, periods=periods, freq='D')
            exog = {
                'is_christmas': [1 if (x.month == 12 and x.day == 25) else 0 for x in future_dates],
                'is_holiday': [1 if (x.month == 12 and x.day == 25) or (x.month == 1 and x.day == 1) else 0 for x in future_dates],
                'day_of_week': [x.weekday() for x in future_dates],
                'is_weekend': [1 if x.weekday() >= 5 else 0 for x in future_dates]
            }
            exog_df = pd.DataFrame(exog)

            forecast_values = model.predict(exog_df)
            forecast_values = [max(0, val) for val in forecast_values]  # Corrige valores negativos para zero
        else:
            logging.error(f"Tipo de modelo não suportado: {model_type}")
            raise HTTPException(status_code=400, detail="Tipo de modelo não suportado.")

        # Organizar o resultado
        dates = pd.date_range(start=future_start, periods=periods, freq='D')
        forecast_df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'store': store_main,
            'item': item_code,
            'forecast_sales': forecast_values
        })

        logging.info(f"Previsão gerada para ({store_main}, {item_code}) com modelo {model_type}: {forecast_df.head().to_dict(orient='records')}")

        return {"forecast": forecast_df.to_dict(orient='records')}

    except HTTPException as http_exc:
        raise http_exc  # Re-raise HTTPExceptions para serem tratadas pelo FastAPI
    except Exception as e:
        logging.error(f"Erro ao gerar a previsão: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao gerar a previsão: {str(e)}")
