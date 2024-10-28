# forecasting/previsão/storage.py

from google.cloud import bigquery
import pandas as pd
import logging

def store_forecast_results(data: pd.DataFrame, table_id: str, client):
    logging.info(f"Carregando previsões para a tabela {table_id} no BigQuery...")
    
    # Remover coluna 'actual_sales' se ela não for necessária na tabela de previsões
    if 'actual_sales' in data.columns:
        data = data.drop(columns=['actual_sales'])
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
    )  # Evita sobrescrever a tabela original
    
    try:
        job = client.load_table_from_dataframe(data, table_id, job_config=job_config)
        job.result()  # Espera a conclusão do job
        logging.info("Previsões carregadas com sucesso no BigQuery.")
    except Exception as e:
        logging.error(f"Erro ao carregar previsões no BigQuery: {e}")
