# forecasting/previsão/main.py

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

# Importar funções e classes dos módulos
from data_extraction import initialize_bigquery_client, get_historical_data_from_bigquery
from models import ProphetModel, ARIMAModel, calculate_metrics
from storage import store_forecast_results
from visualization import plot_forecast_interactive  # Se desejar usar plotly

# Configurar logging com nível DEBUG para maior detalhamento
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para preparar os dados de vendas
def prepare_sales_data(raw_data: pd.DataFrame, apply_log=False, aggregate_weekly=False) -> pd.DataFrame:
    raw_data = raw_data.copy()
    raw_data['date'] = pd.to_datetime(raw_data['date']).dt.tz_localize(None)  # Remove timezone
    logging.debug(f"Timezones após remoção: {raw_data['date'].dt.tz}")  # Deve ser None
    raw_data['sales'] = raw_data['sales'].fillna(0)
    
    if aggregate_weekly:
        sales_data = raw_data.groupby('date').agg({'sales': 'sum'}).resample('W').sum().sort_index()
        logging.info("Dados agregados semanalmente.")
    else:
        sales_data = raw_data.groupby('date').agg({'sales': 'sum'}).sort_index()
    
    sales_data = sales_data.asfreq('D').fillna(0)
    logging.info("Dados de vendas preparados e limpos.")
    
    if apply_log:
        sales_data['sales_log'] = np.log1p(sales_data['sales'])
        logging.debug("Transformação logarítmica aplicada.")
    
    return sales_data


# Função para filtrar itens com alta proporção de vendas zero
def filter_zero_sales(data: pd.DataFrame, threshold=0.9) -> bool:
    zero_sales_ratio = (data['sales'] == 0).mean()
    logging.debug(f"Proporção de vendas zero: {zero_sales_ratio*100:.2f}%")
    if zero_sales_ratio > threshold:
        logging.warning(f"Item com {zero_sales_ratio*100:.2f}% de vendas zero. Considerar remoção ou tratamento especial.")
        return False
    return True


# Função para detectar outliers nos dados de vendas
def detect_outliers(data: pd.DataFrame, column='sales', threshold=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers


# Configurações do BigQuery
service_account_path = 'tfm-sa.json'
project_id = 'perseverance-332400'
dataset_id = 'TFM'
data_table_id = 'ds_market'
forecast_table_id = 'ds_market_forecast'
full_data_table_id = f'{project_id}.{dataset_id}.{data_table_id}'
forecast_table_full_id = f'{project_id}.{dataset_id}.{forecast_table_id}'

# Função de preparação de dados, detecção de outliers, visualização e filtragem de vendas zero
# (mantém as funções fornecidas no seu código)

# Pipeline completo para previsão de estoque
def run_forecast_pipeline_for_top_items_per_store(query: str, forecast_table_full_id: str, model_type='Prophet', test_period_days=30, top_n=3, save_csv=False, csv_path='forecast_results.csv'):
    client = initialize_bigquery_client(project_id, service_account_path)
    raw_data = get_historical_data_from_bigquery(query, client)

    # Verificar colunas necessárias
    required_columns = {'store', 'item', 'date', 'sales'}
    if not required_columns.issubset(raw_data.columns):
        missing = required_columns - set(raw_data.columns)
        raise ValueError(f"As colunas a seguir estão faltando nos dados: {missing}")

    # Processamento das lojas
    stores = raw_data['store'].unique()
    all_forecasts = []

    for store in stores:
        logging.info(f"\nProcessando loja {store}...")
        store_data = raw_data[raw_data['store'] == store]
        total_sales_per_item = store_data.groupby('item')['sales'].sum().reset_index()
        top_items = total_sales_per_item.sort_values(by='sales', ascending=False).head(top_n)['item']

        for item in top_items:
            logging.info(f"Processando item {item} na loja {store}...")
            item_data = store_data[store_data['item'] == item]
            prepared_data = prepare_sales_data(item_data, apply_log=False, aggregate_weekly=False)

            # Filtrar itens com muitas vendas zero
            if not filter_zero_sales(prepared_data, threshold=0.9):
                logging.warning(f"Item {item} na loja {store} possui alta proporção de vendas zero. Pulando...")
                continue

            # Detectar e remover outliers
            outliers = detect_outliers(prepared_data, threshold=1.5)
            if not outliers.empty:
                lower_bound = prepared_data['sales'].quantile(0.25) - 1.5 * (prepared_data['sales'].quantile(0.75) - prepared_data['sales'].quantile(0.25))
                upper_bound = prepared_data['sales'].quantile(0.75) + 1.5 * (prepared_data['sales'].quantile(0.75) - prepared_data['sales'].quantile(0.25))
                prepared_data = prepared_data[(prepared_data['sales'] >= lower_bound) & (prepared_data['sales'] <= upper_bound)]

            # Preparar dados para frequência diária
            prepared_data = prepared_data.asfreq('D').fillna(0)

            # Dividir dados em treino e teste
            cutoff_date = prepared_data.index.max() - pd.Timedelta(days=test_period_days)
            train_data = prepared_data[prepared_data.index <= cutoff_date]
            test_data = prepared_data[prepared_data.index > cutoff_date]
            steps = len(test_data)

            # Verificar tipo de modelo e realizar previsão
            try:
                if model_type == 'Prophet':
                    prophet = ProphetModel(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                    train_df = train_data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
                    prophet.fit(train_df)
                    forecast_values = prophet.predict(steps)
                elif model_type == 'ARIMA':
                    arima = ARIMAModel(seasonal=False, stepwise=True, suppress_warnings=True)
                    arima.fit(train_data['sales'])
                    forecast_values = arima.predict(steps)

                    if arima.model.order[1] > 0:
                        forecast_values = np.cumsum(forecast_values) + train_data['sales'].iloc[-1]

                forecast_values = np.maximum(forecast_values, 0)
                actual_values = test_data['sales'].values

                # Calcular métricas
                rmse, mae, mape = calculate_metrics(actual_values, forecast_values)

                forecast_df = pd.DataFrame({
                    'date': test_data.index,
                    'forecast_sales': forecast_values,
                    'actual_sales': actual_values,
                    'store': store,
                    'item': item,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'model': model_type
                })
                all_forecasts.append(forecast_df)

            except Exception as e:
                logging.error(f"Erro ao processar item {item} na loja {store}: {e}")

    # Armazenar previsões no BigQuery (opcional)
    if all_forecasts:
        full_forecast_df = pd.concat(all_forecasts, ignore_index=True)
        store_forecast_results(full_forecast_df, forecast_table_full_id, client)

# Função principal
def run_pipeline():
    logging.info("Iniciando o pipeline de previsão.")
    client = initialize_bigquery_client(project_id, service_account_path)
    query = f"SELECT store, item, date, sales FROM `{full_data_table_id}`"
    raw_data = get_historical_data_from_bigquery(query, client)
    run_forecast_pipeline_for_top_items_per_store(query, forecast_table_full_id, model_type='Prophet', test_period_days=28)

if __name__ == "__main__":
    run_pipeline()
