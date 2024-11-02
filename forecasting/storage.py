# forecasting/storage.py

from google.cloud import bigquery
import pandas as pd
import logging

def store_forecast_results(data: pd.DataFrame, table_id: str, client: bigquery.Client, write_disposition: str = "WRITE_APPEND"):
    """
    Carrega os resultados das previsões para uma tabela no BigQuery.

    Args:
        data (pd.DataFrame): DataFrame contendo os resultados das previsões.
        table_id (str): ID completo da tabela no BigQuery (dataset.table).
        client (bigquery.Client): Cliente BigQuery autenticado.
        write_disposition (str): Comportamento de escrita (e.g., "WRITE_APPEND", "WRITE_TRUNCATE").

    Raises:
        Exception: Se ocorrer um erro ao carregar os dados.
    """
    logging.info(f"Carregando previsões para a tabela {table_id} no BigQuery...")

    # Remover coluna 'actual_sales' se ela não for necessária na tabela de previsões
    if 'actual_sales' in data.columns:
        data = data.drop(columns=['actual_sales'])

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
    )

    try:
        job = client.load_table_from_dataframe(data, table_id, job_config=job_config)
        job.result()  # Espera a conclusão do job
        logging.info("Previsões carregadas com sucesso no BigQuery.")
    except Exception as e:
        logging.error(f"Erro ao carregar previsões no BigQuery: {e}")
        raise
