# main.py

import logging
import os
from data_extraction import initialize_bigquery_client, get_historical_data_from_bigquery
from feature_engineering import prepare_sales_data, filter_zero_sales, detect_and_remove_outliers
from model_training import fit_and_forecast_prophet_optimized, fit_and_forecast_sarima_optimized, fit_and_forecast_xgboost
from model_evaluation import calculate_metrics
from model_storage import save_trained_model, save_forecast_results
from utils.metrics_storage import save_metrics
from utils.logging_config import setup_logging
from visualization import plot_forecast
from utils.helpers import get_top_items
import pandas as pd

# Obter o diretório do script atual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configurações do BigQuery
SERVICE_ACCOUNT_PATH = 'tfm-sa.json'  # Atualize com o caminho correto para o seu arquivo de credenciais
PROJECT_ID = 'perseverance-332400'    # Atualize com o seu ID de projeto
DATASET_ID = 'TFM'
DATA_TABLE_ID = 'ds_market'
FORECAST_TABLE_ID = 'ds_market_forecast'
FULL_DATA_TABLE_ID = f'{PROJECT_ID}.{DATASET_ID}.{DATA_TABLE_ID}'
FORECAST_TABLE_FULL_ID = f'{PROJECT_ID}.{DATASET_ID}.{FORECAST_TABLE_ID}'

# Caminhos para salvar modelos e métricas
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_FILE_PATH = os.path.join(BASE_DIR, 'metrics', 'model_metrics.csv')
FORECASTS_DIR = os.path.join(BASE_DIR, 'forecasts')
LOG_FILE_PATH = os.path.join(BASE_DIR, 'logs', 'forecast_pipeline.log')

def process_store_item_forecast(store, item, store_data, test_period_days):
    logging.info(f"Processando item {item} na loja {store}...")
    item_data = store_data[store_data['item'] == item]

    # Preparar os dados e adicionar variáveis de evento
    prepared_data = prepare_sales_data(item_data, apply_log=False, aggregate_weekly=False)

    # Filtrar itens com alta proporção de vendas zero
    if not filter_zero_sales(prepared_data, threshold=0.9):
        logging.warning(f"Item {item} na loja {store} possui alta proporção de vendas zero. Pulando...")
        return None

    # Remover outliers
    prepared_data = detect_and_remove_outliers(prepared_data)
    prepared_data = prepared_data.asfreq('D').fillna(0)
    cutoff_date = prepared_data.index.max() - pd.Timedelta(days=test_period_days)
    train_data = prepared_data[prepared_data.index <= cutoff_date]
    test_data = prepared_data[prepared_data.index > cutoff_date]
    steps = len(test_data)

    # Preparar dados para Prophet e XGBoost
    regressors = ['is_christmas', 'is_holiday', 'day_of_week', 'is_weekend']
    train_df = train_data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
    test_df = test_data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})

    for reg in regressors:
        if reg not in train_df.columns:
            train_df[reg] = train_data[reg].values
        if reg not in test_df.columns:
            test_df[reg] = test_data[reg].values

    # Lista para armazenar previsões e métricas
    forecast_results = []

    # Otimizar e ajustar Prophet
    forecast_prophet, model_type, prophet_params, prophet_model = fit_and_forecast_prophet_optimized(
        train_df, test_df, steps, regressors, n_trials=50
    )
    if forecast_prophet is not None:
        save_trained_model(prophet_model, model_type, store, item, models_dir=MODELS_DIR)
        metrics_prophet = calculate_metrics(test_df['y'], forecast_prophet, model_type)
        forecast_results.append((forecast_prophet, model_type, metrics_prophet, prophet_params))

    # Ajustar SARIMA
    forecast_sarima, model_type, sarima_params, sarima_model = fit_and_forecast_sarima_optimized(
        train_data, test_data, steps
    )
    if forecast_sarima is not None:
        save_trained_model(sarima_model, model_type, store, item, models_dir=MODELS_DIR)
        metrics_sarima = calculate_metrics(test_df['y'], forecast_sarima, model_type)
        forecast_results.append((forecast_sarima, model_type, metrics_sarima, sarima_params))

    # Ajustar XGBoost
    forecast_xgboost, model_type, xgboost_params, xgboost_model = fit_and_forecast_xgboost(
        train_df, test_df, steps, regressors
    )
    if forecast_xgboost is not None:
        save_trained_model(xgboost_model, model_type, store, item, models_dir=MODELS_DIR)
        metrics_xgboost = calculate_metrics(test_df['y'], forecast_xgboost, model_type)
        forecast_results.append((forecast_xgboost, model_type, metrics_xgboost, xgboost_params))

    # Selecionar o melhor modelo baseado no MAPE
    best_forecast, best_model_type, best_metrics, best_params = min(forecast_results, key=lambda x: x[2]['mape'])
    save_metrics(store, item, best_model_type, best_metrics['rmse'], best_metrics['mse'], best_metrics['mae'], 
             best_metrics['mape'], best_metrics['r2'], best_metrics['smape'], best_metrics['mase'], 
             params=best_params, metrics_file_path=METRICS_FILE_PATH)

    # Salvar o gráfico de previsão do melhor modelo
    forecast_df = pd.DataFrame({
        'date': test_df['ds'],
        'forecast_sales': best_forecast,
        'actual_sales': test_df['y'].values,
        'store': store,
        'item': item
    })
    plot_path = os.path.join(BASE_DIR, 'metrics', f"forecast_{store}_{item}_{best_model_type}.png")
    plot_forecast(forecast_df, save_path=plot_path)
    logging.info(f"Gráfico salvo em {plot_path}")

    # Retornar o DataFrame de previsões
    return forecast_df

def run_forecast_pipeline_for_top_items_per_store(query, forecast_table_full_id, test_period_days=28, top_n=5):
    client = initialize_bigquery_client(PROJECT_ID, SERVICE_ACCOUNT_PATH)
    raw_data = get_historical_data_from_bigquery(query, client)

    stores = raw_data['store'].unique()
    all_forecasts = []  # Lista para coletar todas as previsões

    # Garantir que o diretório de métricas existe
    metrics_dir = os.path.dirname(METRICS_FILE_PATH)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Limpar o arquivo de métricas antes de iniciar
    if os.path.exists(METRICS_FILE_PATH):
        os.remove(METRICS_FILE_PATH)
    # No cabeçalho do CSV
    with open(METRICS_FILE_PATH, 'w') as f:
        f.write("store,item,model_type,rmse,mse,mae,mape,r2,smape,mase,changepoint_prior_scale,seasonality_mode,order,seasonal_order\n")


    for store in stores:
        logging.info(f"\nProcessando loja {store}...")
        store_data = raw_data[raw_data['store'] == store]
        top_items = get_top_items(store_data, top_n)  # Obter os top N itens

        for item in top_items:
            result = process_store_item_forecast(store, item, store_data, test_period_days)
            if result is not None:
                forecast_df = result
                all_forecasts.append(forecast_df)

    # Concatenar todas as previsões em um único DataFrame
    if all_forecasts:
        all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        # Salvar as previsões usando a função save_forecast_results
        save_forecast_results(all_forecasts_df, forecasts_dir=FORECASTS_DIR)
        # Realizar as agregações
        aggregate_and_save_forecasts(all_forecasts_df)
    else:
        logging.warning("Nenhuma previsão foi gerada.")

def aggregate_and_save_forecasts(forecast_data):
    if not os.path.exists(FORECASTS_DIR):
        os.makedirs(FORECASTS_DIR)

    # Agregar por loja
    store_forecasts = forecast_data.groupby(['date', 'store'])['forecast_sales'].sum().reset_index()
    store_forecasts.to_csv(os.path.join(FORECASTS_DIR, "store_forecasts.csv"), index=False)
    logging.info(f"Previsões agregadas por loja salvas em {os.path.join(FORECASTS_DIR, 'store_forecasts.csv')}")

    # Se houver informações de departamento
    if 'department' in forecast_data.columns:
        department_forecasts = forecast_data.groupby(['date', 'department'])['forecast_sales'].sum().reset_index()
        department_forecasts.to_csv(os.path.join(FORECASTS_DIR, "department_forecasts.csv"), index=False)
        logging.info(f"Previsões agregadas por departamento salvas em {os.path.join(FORECASTS_DIR, 'department_forecasts.csv')}")

    # Se houver informações de cidade
    if 'city' in forecast_data.columns:
        city_forecasts = forecast_data.groupby(['date', 'city'])['forecast_sales'].sum().reset_index()
        city_forecasts.to_csv(os.path.join(FORECASTS_DIR, "city_forecasts.csv"), index=False)
        logging.info(f"Previsões agregadas por cidade salvas em {os.path.join(FORECASTS_DIR, 'city_forecasts.csv')}")

def run_pipeline():
    setup_logging(LOG_FILE_PATH)
    logging.info("Iniciando o pipeline de previsão.")

    # Ajustar a consulta para incluir dados de 2011 a 2016 inclusive
    query = """
    SELECT store, item, date, sales 
    FROM `perseverance-332400.TFM.ds_market`
    """
    top_n = 10  # Defina o número de top itens que deseja processar por loja
    run_forecast_pipeline_for_top_items_per_store(query, FORECAST_TABLE_FULL_ID, test_period_days=28, top_n=top_n)

if __name__ == "__main__":
    run_pipeline()
