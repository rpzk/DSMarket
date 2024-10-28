# forecasting/previsão/data_extraction.py

from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import logging

def initialize_bigquery_client(project_id: str, service_account_path: str):
    try:
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        client = bigquery.Client(project=project_id, credentials=credentials)
        logging.info("Cliente BigQuery inicializado com sucesso.")
        return client
    except Exception as e:
        logging.error(f"Erro ao inicializar o cliente BigQuery: {e}")
        raise

def get_historical_data_from_bigquery(query: str, client) -> pd.DataFrame:
    try:
        query_job = client.query(query)
        data = query_job.to_dataframe()
        logging.info(f"Dados extraídos do BigQuery com {data.shape[0]} linhas e {data.shape[1]} colunas.")
        logging.info(f"Colunas disponíveis nos dados extraídos: {data.columns.tolist()}")
        if data.empty:
            logging.warning("A consulta retornou um DataFrame vazio.")
        return data
    except Exception as e:
        logging.error(f"Erro ao extrair dados do BigQuery: {e}")
        return pd.DataFrame()  # Retorna um DataFrame vazio para evitar falhas subsequentes
