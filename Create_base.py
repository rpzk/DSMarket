import pandas as pd
from sqlalchemy import create_engine, types

# Definir variáveis e caminhos
item_sales_file = 'data_dsmarket/item_sales.csv'
daily_calendar_file = 'data_dsmarket/daily_calendar_with_events.csv'
output_csv = 'data_dsmarket/item_sales_with_calendar.csv'
CHUNK_SIZE = 100000  # Tamanho do chunk para processar em blocos

# Credenciais de conexão PostgreSQL
USER = "postgres"
PASSWORD = "sk0400"
HOST = "localhost"
PORT = "5433"
DB_NAME = "brunomatos"  # Nome do banco de dados
table_name = "item_sales_with_calendar"  # Nome da tabela no PostgreSQL

# Criar a string de conexão para o PostgreSQL
connection_string = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}"
engine = create_engine(connection_string)

# Definir os tipos de dados para a tabela no banco de dados
dtype_mapping = {
    'event': types.TEXT,
    'id': types.String(),  # Ou o tipo adequado
    'date': types.Date(),  # Certifique-se de que a data está no formato correto
    # Adicione outros mapeamentos de tipos, se necessário
}

# Função para carregar, transformar e salvar o CSV
def transform_and_save_csv():
    print("Iniciando a transformação dos arquivos CSV...")

    # Passo 1: Carregar os arquivos CSV
    item_sales_df = pd.read_csv(item_sales_file)
    daily_calendar_df = pd.read_csv(daily_calendar_file)

    # Passo 2: Derreter (melt) a tabela item_sales para transformar colunas de dias em linhas
    item_sales_melted = pd.melt(item_sales_df, 
                                id_vars=['id', 'item', 'category', 'department', 'store', 'store_code', 'region'], 
                                var_name='day', 
                                value_name='sales')

    # Passo 3: Realizar a junção com o daily_calendar_df para adicionar as informações de calendário
    item_sales_with_calendar = pd.merge(item_sales_melted, 
                                        daily_calendar_df, 
                                        how='left', 
                                        left_on='day', 
                                        right_on='d')

    # Passo 4: Visualizar as primeiras linhas do novo dataframe
    print(item_sales_with_calendar.head())

    # Passo 5: Salvar o novo dataframe como um arquivo CSV
    item_sales_with_calendar.to_csv(output_csv, index=False)
    print(f"Processo concluído e arquivo salvo como '{output_csv}'.")

# Função para inserir os dados no banco de dados PostgreSQL
def load_csv_to_postgres():
    print("Iniciando a inserção no banco de dados PostgreSQL...")

    # Ler e carregar o CSV em blocos (chunks) e inserir no PostgreSQL
    for chunk in pd.read_csv(output_csv, chunksize=CHUNK_SIZE):
        chunk.to_sql(table_name, engine, index=False, if_exists='append', dtype=dtype_mapping)
        print(f"Chunk inserido com {len(chunk)} linhas.")

    # Otimizações no banco após o carregamento
    with engine.connect() as connection:
        # Criar índices para melhorar a performance das consultas
        connection.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name} (id);")
        connection.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name} (date);")
        print(f"Índices criados para a tabela {table_name}.")

    # Testar a inserção, selecionar os primeiros 5 registros
    query_result = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", engine)
    print("Primeiros registros da tabela após inserção:")
    print(query_result)

    # Fechar a conexão do SQLAlchemy
    engine.dispose()

# Execução principal
if __name__ == "__main__":
    # Realizar a transformação e salvar o arquivo CSV
    transform_and_save_csv()

    # Inserir o novo CSV no banco de dados PostgreSQL
    load_csv_to_postgres()
