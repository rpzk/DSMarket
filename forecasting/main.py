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

# Configuração do logging para salvar em um arquivo
log_file_path = "logs/forecast_pipeline.log"
log_dir = os.path.dirname(log_file_path)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Caminho do arquivo para salvar métricas
metrics_file_path = "metrics/model_metrics.csv"
metrics_dir = os.path.dirname(metrics_file_path)

if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

if not os.path.exists(metrics_file_path):
    with open(metrics_file_path, 'w') as f:
        f.write("store,item,model_type,rmse,mae,mape,changepoint_prior_scale,daily_seasonality,weekly_seasonality,yearly_seasonality\n")

# Função para salvar as métricas e os parâmetros do modelo
def save_metrics(store, item, model_type, rmse, mae, mape, params):
    with open(metrics_file_path, 'a') as f:
        f.write(f"{store},{item},{model_type},{rmse},{mae},{mape},{params['changepoint_prior_scale']},{params['daily_seasonality']},{params['weekly_seasonality']},{params['yearly_seasonality']}\n")

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

# Configurações do BigQuery
service_account_path = 'tfm-sa.json'
project_id = 'perseverance-332400'
dataset_id = 'TFM'
data_table_id = 'ds_market'
forecast_table_id = 'ds_market_forecast'
full_data_table_id = f'{project_id}.{dataset_id}.{data_table_id}'
forecast_table_full_id = f'{project_id}.{dataset_id}.{forecast_table_id}'

# Pipeline completo para previsão de estoque
def run_forecast_pipeline_for_top_items_per_store(query: str, forecast_table_full_id: str, test_period_days=30, top_n=3):
    client = initialize_bigquery_client(project_id, service_account_path)
    raw_data = get_historical_data_from_bigquery(query, client)

    stores = raw_data['store'].unique()

    for store in stores:
        logging.info(f"\nProcessando loja {store}...")
        store_data = raw_data[raw_data['store'] == store]
        total_sales_per_item = store_data.groupby('item')['sales'].sum().reset_index()
        top_items = total_sales_per_item.sort_values(by='sales', ascending=False).head(top_n)['item']

        for item in top_items:
            logging.info(f"Processando item {item} na loja {store}...")
            item_data = store_data[store_data['item'] == item]
            prepared_data = prepare_sales_data(item_data, apply_log=False, aggregate_weekly=False)

            # Filtrar itens com alta proporção de vendas zero
            if not filter_zero_sales(prepared_data, threshold=0.9):
                logging.warning(f"Item {item} na loja {store} possui alta proporção de vendas zero. Pulando...")
                continue

            # Remover outliers
            prepared_data = detect_and_remove_outliers(prepared_data)
            prepared_data = prepared_data.asfreq('D').fillna(0)
            cutoff_date = prepared_data.index.max() - pd.Timedelta(days=test_period_days)
            train_data = prepared_data[prepared_data.index <= cutoff_date]
            test_data = prepared_data[prepared_data.index > cutoff_date]
            steps = len(test_data)

            # Ajuste do modelo Prophet com múltiplos valores de changepoint_prior_scale
            params = {
                "changepoint_prior_scale": 0.1,
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": True
            }
            try:
                prophet = Prophet(
                    changepoint_prior_scale=params["changepoint_prior_scale"],
                    daily_seasonality=params["daily_seasonality"],
                    weekly_seasonality=params["weekly_seasonality"],
                    yearly_seasonality=params["yearly_seasonality"]
                )
                train_df = train_data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
                prophet.fit(train_df)
                future = prophet.make_future_dataframe(periods=steps)
                forecast = prophet.predict(future)
                forecast_values = forecast['yhat'][-steps:].values
                model_type = "Prophet"
            except Exception as e:
                logging.error(f"Erro ao usar Prophet para {item} na loja {store}: {e}")
                continue

            # Alternativa com SARIMA se Prophet falhar
            try:
                sarima = SARIMAX(train_data['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                sarima_fit = sarima.fit(disp=False)
                forecast_values = sarima_fit.forecast(steps=steps)
                model_type = "SARIMA"
            except Exception as e:
                logging.error(f"Erro ao usar SARIMA para {item} na loja {store}: {e}")
                continue

            # Calcular métricas de erro
            actual_values = test_data['sales'].values
            rmse, mae, mape = calculate_metrics(actual_values, forecast_values)
            save_metrics(store, item, model_type, rmse, mae, mape, params)

            # Salvar gráfico de previsão
            forecast_df = pd.DataFrame({
                'date': test_data.index,
                'forecast_sales': forecast_values,
                'actual_sales': actual_values,
                'store': store,
                'item': item
            })
            plot_path = f"metrics/forecast_{store}_{item}.png"
            plot_forecast(forecast_df, save_path=plot_path)
            logging.info(f"Gráfico salvo em {plot_path}")

# Função principal
def run_pipeline():
    query = "SELECT store, item, date, sales FROM `perseverance-332400.TFM.ds_market` WHERE date BETWEEN '2015-01-01' AND '2016-01-01'"
    run_forecast_pipeline_for_top_items_per_store(query, forecast_table_full_id)

if __name__ == "__main__":
    run_pipeline()
