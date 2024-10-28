# results_viewer.py

from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import matplotlib.pyplot as plt

# Configurações do BigQuery
PROJECT_ID = 'perseverance-332400'
DATASET_ID = 'TFM'
TABLE_ID = 'ds_market_forecast'
SERVICE_ACCOUNT_PATH = 'tfm-sa.json'  # Caminho para seu arquivo de credenciais

def initialize_bigquery_client():
    """Inicializa o cliente do BigQuery."""
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
    return client

def fetch_forecast_data(store=None, item=None, limit=1000):
    """Busca os dados de previsão do BigQuery com filtros opcionais."""
    client = initialize_bigquery_client()
    
    # Construindo a query SQL
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
    filters = []
    if store:
        filters.append(f"store = '{store}'")
    if item:
        filters.append(f"item = '{item}'")
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += f" LIMIT {limit}"
    
    # Executando a query
    query_job = client.query(query)
    data = query_job.to_dataframe()
    return data

def plot_forecast(data):
    """Gera gráficos para as previsões com base nos dados disponíveis."""
    if data.empty:
        print("Nenhum dado disponível para exibir.")
        return
    
    # Exibindo colunas para debug
    print("Colunas disponíveis no DataFrame:", data.columns.tolist())
    
    # Converte a coluna 'date' para o tipo datetime
    data['date'] = pd.to_datetime(data['date'])

    # Plotando apenas a Previsão (forecast_sales)
    if 'forecast_sales' in data.columns:
        plt.figure(figsize=(14, 7))
        plt.plot(data['date'], data['forecast_sales'], label='Previsão', color='orange')
        plt.title(f"Previsão de Vendas (Store: {data['store'].iloc[0]}, Item: {data['item'].iloc[0]})")
        plt.xlabel("Data")
        plt.ylabel("Previsão de Vendas")
        plt.legend()
        plt.show()
    else:
        print("Coluna 'forecast_sales' não encontrada no DataFrame.")

    # Aviso sobre ausência de vendas reais
    print("A coluna 'actual_sales' está ausente. Gráficos de vendas reais e resíduos não podem ser gerados.")


if __name__ == "__main__":
    # Exemplo: buscar dados para uma loja e item específicos
    store = 'Harlem'  # Defina a loja desejada
    item = 'SUPERMARKET_3_586'  # Defina o item desejado
    data = fetch_forecast_data(store=store, item=item)
    plot_forecast(data)

print("Colunas disponíveis:", data.columns)
