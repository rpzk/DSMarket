from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import os
from data_extraction import initialize_bigquery_client, get_historical_data_from_bigquery
from models import ProphetModel, ARIMAModel, calculate_metrics
from storage import store_forecast_results
from visualization import plot_forecast
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configuração dos caminhos e logging
log_file_path = "logs/forecast_pipeline.log"
metrics_file_path = "metrics/model_metrics.csv"
service_account_path = 'tfm-sa.json'
project_id = 'perseverance-332400'
dataset_id = 'TFM'
data_table_id = 'ds_market'
forecast_table_id = 'ds_market_forecast'
full_data_table_id = f'{project_id}.{dataset_id}.{data_table_id}'
forecast_table_full_id = f'{project_id}.{dataset_id}.{forecast_table_id}'

# Funções auxiliares
def setup_logging():
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def save_metrics(store, item, model_type, rmse, mae, mape, params):
    if not os.path.exists(os.path.dirname(metrics_file_path)):
        os.makedirs(os.path.dirname(metrics_file_path))
    with open(metrics_file_path, 'a') as f:
        f.write(f"{store},{item},{model_type},{rmse},{mae},{mape},{params['changepoint_prior_scale']},"
                f"{params['daily_seasonality']},{params['weekly_seasonality']},{params['yearly_seasonality']}\n")

# Função para preparar os dados de vendas
def prepare_sales_data(raw_data: pd.DataFrame, apply_log=False, aggregate_weekly=False) -> pd.DataFrame:
    raw_data = raw_data.copy()
    raw_data['date'] = pd.to_datetime(raw_data['date']).dt.tz_localize(None)
    raw_data['sales'] = raw_data['sales'].fillna(0)
    
    if aggregate_weekly:
        sales_data = raw_data.groupby('date').agg({'sales': 'sum'}).resample('W').sum().sort_index()
    else:
        sales_data = raw_data.groupby('date').agg({'sales': 'sum'}).sort_index()
    
    sales_data = sales_data.asfreq('D').fillna(0)
    
    if apply_log:
        sales_data['sales_log'] = np.log1p(sales_data['sales'])
    
    return sales_data

# Função para filtrar itens com alta proporção de vendas zero
def filter_zero_sales(data: pd.DataFrame, threshold=0.9) -> bool:
    zero_sales_ratio = (data['sales'] == 0).mean()
    if zero_sales_ratio > threshold:
        logging.warning(f"Item com {zero_sales_ratio*100:.2f}% de vendas zero. Considerar remoção ou tratamento especial.")
        return False
    return True

# Função para detectar e remover outliers
def detect_and_remove_outliers(data: pd.DataFrame, column='sales', threshold=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    cleaned_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return cleaned_data

def run_forecast_pipeline(**context):
    query = "SELECT store, item, date, sales FROM `perseverance-332400.TFM.ds_market` WHERE date BETWEEN '2015-01-01' AND '2016-01-01'"
    client = initialize_bigquery_client(project_id, service_account_path)
    raw_data = get_historical_data_from_bigquery(query, client)
    stores = raw_data['store'].unique()
    top_n = 3
    test_period_days = 30

    for store in stores:
        store_data = raw_data[raw_data['store'] == store]
        total_sales_per_item = store_data.groupby('item')['sales'].sum().reset_index()
        top_items = total_sales_per_item.sort_values(by='sales', ascending=False).head(top_n)['item']

        for item in top_items:
            item_data = store_data[store_data['item'] == item]
            prepared_data = prepare_sales_data(item_data)
            if not filter_zero_sales(prepared_data, threshold=0.9):
                continue

            prepared_data = detect_and_remove_outliers(prepared_data)
            cutoff_date = prepared_data.index.max() - pd.Timedelta(days=test_period_days)
            train_data = prepared_data[prepared_data.index <= cutoff_date]
            test_data = prepared_data[prepared_data.index > cutoff_date]
            steps = len(test_data)

            try:
                # Modelagem Prophet
                prophet = Prophet(changepoint_prior_scale=0.1)
                prophet.fit(train_data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'}))
                future = prophet.make_future_dataframe(periods=steps)
                forecast = prophet.predict(future)
                forecast_values = forecast['yhat'][-steps:].values
                model_type = "Prophet"
            except Exception:
                # Modelagem SARIMA
                sarima = SARIMAX(train_data['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                sarima_fit = sarima.fit(disp=False)
                forecast_values = sarima_fit.forecast(steps=steps)
                model_type = "SARIMA"

            # Calcular métricas e salvar
            actual_values = test_data['sales'].values
            rmse, mae, mape = calculate_metrics(actual_values, forecast_values)
            save_metrics(store, item, model_type, rmse, mae, mape, {"changepoint_prior_scale": 0.1, "daily_seasonality": True, "weekly_seasonality": True, "yearly_seasonality": True})

# Definindo a DAG
with DAG(
    "forecast_pipeline_dag",
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": datetime(2024, 1, 1),
        "retries": 1,
    },
    schedule_interval="0 0 * * *",
    catchup=False,
) as dag:

    setup_logging_task = PythonOperator(
        task_id="setup_logging",
        python_callable=setup_logging
    )

    run_forecast_task = PythonOperator(
        task_id="run_forecast_pipeline",
        python_callable=run_forecast_pipeline,
        provide_context=True
    )

    setup_logging_task >> run_forecast_task
