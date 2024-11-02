from google.oauth2 import service_account
from google.cloud import bigquery

# Caminho para o arquivo JSON com as credenciais da Service Account
key_path = 'eda/tfm-sa.json'

# Cria as credenciais da Service Account
credentials = service_account.Credentials.from_service_account_file(key_path)

# Inicializa o cliente BigQuery
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Lista todos os datasets no projeto
datasets = list(client.list_datasets())  # Retorna um iterador de datasets

if datasets:
    print("Datasets no projeto {}: ".format(client.project))
    for dataset in datasets:
        print(" - {}".format(dataset.dataset_id))
else:
    print("Nenhum dataset encontrado.")

