# forecasting/main.py

import pandas as pd
import numpy as np
import logging
import os
import joblib
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Função para inicializar o cliente do BigQuery
def initialize_bigquery_client(project_id, service_account_path):
    from google.cloud import bigquery
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(service_account_path)
    client = bigquery.Client(credentials=credentials, project=project_id)
    return client

# Função para obter dados históricos do BigQuery
def get_historical_data_from_bigquery(query, client):
    query_job = client.query(query)
    results = query_job.result()
    data = results.to_dataframe()
    return data

# Função para calcular métricas
def calculate_metrics(actual, forecast):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    # Para calcular MAPE, evitar divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((actual - forecast) / actual)
        mape = np.where(np.isfinite(mape), mape, np.nan)
        mape = np.nanmean(mape) * 100
    return rmse, mae, mape

# Função para plotar a previsão
def plot_forecast(forecast_df, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(forecast_df['date'], forecast_df['actual_sales'], label='Vendas Reais')
    plt.plot(forecast_df['date'], forecast_df['forecast_sales'], label='Previsão de Vendas')
    plt.xlabel('Data')
    plt.ylabel('Vendas')
    plt.title(f"Previsão vs Vendas Reais - Loja {forecast_df['store'].iloc[0]}, Item {forecast_df['item'].iloc[0]}")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

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
        f.write(f"{store},{item},{model_type},{rmse},{mae},{mape},{params.get('changepoint_prior_scale','')},{params.get('daily_seasonality','')},{params.get('weekly_seasonality','')},{params.get('yearly_seasonality','')}\n")

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
service_account_path = 'tfm-sa.json'  # Atualize com o caminho correto para o seu arquivo de credenciais
project_id = 'perseverance-332400'    # Atualize com o seu ID de projeto
dataset_id = 'TFM'
data_table_id = 'ds_market'
forecast_table_id = 'ds_market_forecast'
full_data_table_id = f'{project_id}.{dataset_id}.{data_table_id}'
forecast_table_full_id = f'{project_id}.{dataset_id}.{forecast_table_id}'

# Função para obter os top N itens mais vendidos em uma loja
def get_top_items(store_data: pd.DataFrame, top_n: int) -> pd.Series:
    """
    Obtém os top N itens com base nas vendas totais em uma loja.

    Args:
        store_data (pd.DataFrame): Dados de vendas da loja.
        top_n (int): Número de itens a selecionar.

    Returns:
        pd.Series: Série contendo os IDs dos top N itens.
    """
    total_sales_per_item = store_data.groupby('item')['sales'].sum().reset_index()
    top_items = total_sales_per_item.sort_values(by='sales', ascending=False).head(top_n)['item']
    return top_items

def fit_and_forecast_prophet(train_data, steps):
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
        return forecast_values, model_type, params, prophet
    except Exception as e:
        logging.error(f"Erro ao usar Prophet: {e}")
        return None, None, None, None

def fit_and_forecast_sarima(train_data, steps):
    params = {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 7)
    }
    try:
        sarima = SARIMAX(train_data['sales'], order=params["order"], seasonal_order=params["seasonal_order"])
        sarima_fit = sarima.fit(disp=False)
        forecast_values = sarima_fit.forecast(steps=steps)
        model_type = "SARIMA"
        return forecast_values, model_type, params, sarima_fit
    except Exception as e:
        logging.error(f"Erro ao usar SARIMA: {e}")
        return None, None, None, None

def process_store_item_forecast(store, item, store_data, test_period_days):
    logging.info(f"Processando item {item} na loja {store}...")
    item_data = store_data[store_data['item'] == item]
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

    # Tentar ajustar e prever com Prophet
    forecast_values, model_type, params, trained_model = fit_and_forecast_prophet(train_data, steps)

    if forecast_values is None:
        # Se Prophet falhar, tentar com SARIMA
        forecast_values, model_type, params, trained_model = fit_and_forecast_sarima(train_data, steps)
        if forecast_values is None:
            logging.error(f"Ambos os modelos falharam para o item {item} na loja {store}.")
            return None

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

    # Retornar o DataFrame de previsões e o modelo treinado
    return forecast_df, trained_model, model_type

def run_forecast_pipeline_for_top_items_per_store(query: str, forecast_table_full_id: str, test_period_days=28, top_n=5):
    client = initialize_bigquery_client(project_id, service_account_path)
    raw_data = get_historical_data_from_bigquery(query, client)

    stores = raw_data['store'].unique()
    all_forecasts = []  # Lista para coletar todas as previsões
    trained_models = {}  # Dicionário para armazenar os modelos treinados

    for store in stores:
        logging.info(f"\nProcessando loja {store}...")
        store_data = raw_data[raw_data['store'] == store]
        top_items = get_top_items(store_data, top_n)  # Obter os top N itens

        for item in top_items:
            result = process_store_item_forecast(store, item, store_data, test_period_days)
            if result is not None:
                forecast_df, trained_model, model_type = result
                all_forecasts.append(forecast_df)
                # Salvar o modelo treinado
                models_dir = 'models'
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                if model_type == "Prophet":
                    # Salvar modelo Prophet com joblib
                    model_filename = f"prophet_model_{store}_{item}.pkl"
                    joblib.dump(trained_model, os.path.join(models_dir, model_filename))
                    trained_models[(store, item)] = os.path.join(models_dir, model_filename)
                elif model_type == "SARIMA":
                    # Salvar modelo SARIMA usando save
                    model_filename = f"sarima_model_{store}_{item}.pkl"
                    trained_model.save(os.path.join(models_dir, model_filename))
                    trained_models[(store, item)] = os.path.join(models_dir, model_filename)

    # Salvar os modelos treinados
    if trained_models:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        joblib.dump(trained_models, os.path.join(models_dir, 'trained_models.pkl'))
        logging.info("Modelos treinados salvos em models/trained_models.pkl")
    else:
        logging.warning("Nenhum modelo foi treinado.")

    # Concatenar todas as previsões em um único DataFrame
    if all_forecasts:
        all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        # Salvar as previsões em um arquivo CSV
        forecasts_dir = "forecasts"
        if not os.path.exists(forecasts_dir):
            os.makedirs(forecasts_dir)
        all_forecasts_df.to_csv(os.path.join(forecasts_dir, "all_forecasts.csv"), index=False)
        logging.info("Previsões individuais salvas em forecasts/all_forecasts.csv")

        # Realizar as agregações
        aggregate_and_save_forecasts(all_forecasts_df)
    else:
        logging.warning("Nenhuma previsão foi gerada.")

def aggregate_and_save_forecasts(forecast_data):
    forecasts_dir = "forecasts"
    if not os.path.exists(forecasts_dir):
        os.makedirs(forecasts_dir)

    # Agregar por loja
    store_forecasts = forecast_data.groupby(['date', 'store'])['forecast_sales'].sum().reset_index()
    store_forecasts.to_csv(os.path.join(forecasts_dir, "store_forecasts.csv"), index=False)
    logging.info("Previsões agregadas por loja salvas em forecasts/store_forecasts.csv")

    # Se houver informações de departamento
    if 'department' in forecast_data.columns:
        department_forecasts = forecast_data.groupby(['date', 'department'])['forecast_sales'].sum().reset_index()
        department_forecasts.to_csv(os.path.join(forecasts_dir, "department_forecasts.csv"), index=False)
        logging.info("Previsões agregadas por departamento salvas em forecasts/department_forecasts.csv")

    # Se houver informações de cidade
    if 'city' in forecast_data.columns:
        city_forecasts = forecast_data.groupby(['date', 'city'])['forecast_sales'].sum().reset_index()
        city_forecasts.to_csv(os.path.join(forecasts_dir, "city_forecasts.csv"), index=False)
        logging.info("Previsões agregadas por cidade salvas em forecasts/city_forecasts.csv")

# Função para calcular a quantidade de reabastecimento (opcional)
def calculate_reorder_quantity(forecast_df, current_stock_data, safety_stock_data, lead_time_data):
    # Implementar se necessário
    pass

# Função principal
def run_pipeline():
    # Ajustar a consulta para incluir dados de 2011 a 2016 inclusive
    query = """
    SELECT store, item, date, sales 
    FROM `perseverance-332400.TFM.ds_market` 
    WHERE date BETWEEN '2011-01-01' AND '2016-12-31'
    """
    top_n = 5  # Defina o número de top itens que deseja processar por loja
    run_forecast_pipeline_for_top_items_per_store(query, forecast_table_full_id, test_period_days=28, top_n=top_n)

if __name__ == "__main__":
    run_pipeline()
