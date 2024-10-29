from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import db_dtypes
import matplotlib.pyplot as plt
import dtale
import plotly.express as px
import plotly.graph_objects as go



def get_data():
    # Caminho para o arquivo de chave da conta de serviço - Lucas disponibilizou no Whatsapp do grupo
    service_account_path = 'tfm-sa.json'

    # Criar objeto de credenciais
    credentials = service_account.Credentials.from_service_account_file(service_account_path)

    project_id = 'perseverance-332400'
    dataset_id = 'TFM'
    table_id = 'ds_market'
    full_table_id = f'{project_id}.{dataset_id}.{table_id}'

    client = bigquery.Client(project='perseverance-332400', credentials=credentials)

    total_items_sold_overall = pd.DataFrame()
    total_sales_value_overall = pd.DataFrame()

    data_frames = []
    
    # Definir o intervalo de anos (de 2012 até 2016)
    for year in range(2012, 2017):  # O range vai até 2017 para incluir o ano de 2016
        print(f"Processando dados para o ano {year}...")
        
        # Definir a consulta SQL para o ano atual
        query = f"""
        SELECT
            id,
            item,
            category_x,
            category_y,
            department,
            store,
            store_code,
            region,
            d,
            sales,
            yearweek,
            date,
            event,
            sell_price
        FROM
            `perseverance-332400.TFM.ds_market`
        WHERE
            EXTRACT(YEAR FROM date) = {year} 
           
        """

        # Executar a consulta e obter o DataFrame
        df_year = client.query(query).to_dataframe()
        
        # Apendar o DataFrame do ano na lista
        data_frames.append(df_year)

    # Concatenar todos os DataFrames de cada ano em um único DataFrame
    df_all_years = pd.concat(data_frames, ignore_index=True)

    return df_all_years

def clean_data(df):
    # Renomear 'category_x' para 'category'
    df = df.rename(columns={'category_x': 'category'})

    # Remover a coluna 'category_y' se existir
    if 'category_y' in df.columns:
        df = df.drop('category_y', axis=1)

    # Remover a coluna 'd' se existir
    if 'd' in df.columns:
        df = df.drop('d', axis=1)

    # Preencher valores nulos na coluna 'event' com strings vazias
    if 'event' in df.columns:
        df['event'] = df['event'].fillna('')

    # Preencher valores nulos em 'sell_price' com zeros
    df['sell_price'] = df['sell_price'].fillna(0)

    # Converter 'sell_price' para float32 com 2 casas decimais
    df['sell_price'] = df['sell_price'].astype(float).round(2)
    df['sell_price'] = df['sell_price'].astype('float32')

    return df

# Função para clusterização de produtos por preço e volume de vendas
def cluster_produto_preco(df):
    vendas_preco_produto = df.groupby('item').agg({'sales': 'sum', 'sell_price': ['mean', 'std']}).reset_index()
    vendas_preco_produto.columns = ['item', 'total_sales', 'average_price', 'price_std']
    vendas_preco_produto['price_std'] = vendas_preco_produto['price_std'].fillna(0)

    variaveis_cluster = vendas_preco_produto[['total_sales', 'average_price', 'price_std']]
    scaler = StandardScaler()
    variaveis_cluster_scaled = scaler.fit_transform(variaveis_cluster)

    # Aplicar K-means com 5 clusters
    kmeans_produto_preco = KMeans(n_clusters=5, random_state=42)
    vendas_preco_produto['cluster'] = kmeans_produto_preco.fit_predict(variaveis_cluster_scaled)

    return vendas_preco_produto

# Função para plotar os clusters com Plotly Express
def plot_cluster_produto_preco(vendas_preco_produto):
    fig = px.scatter(vendas_preco_produto, x='average_price', y='total_sales', color='cluster', 
                     title='Clusterização de Produtos por Preço e Vendas (5 Clusters)', 
                     labels={'average_price': 'Preço Médio', 'total_sales': 'Total de Vendas'})
    fig.show()

# Função para exibir estatísticas dos clusters de produtos como tabela com Plotly
    cluster_stats = vendas_preco_produto.groupby('cluster').agg({
        'total_sales': ['mean', 'std'],
        'average_price': ['mean', 'std'],
        'price_std': ['mean', 'std']
    }).reset_index()

    # Exibir a tabela formatada com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(cluster_stats.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[cluster_stats[col] for col in cluster_stats.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title="Estatísticas por Cluster de Produto e Preço (5 Clusters)")
    fig.show()

# Função para clusterização de lojas
def cluster_lojas(df):
    vendas_loja = df.groupby('store').agg({'sales': 'sum', 'sell_price': ['mean', 'std']}).reset_index()
    vendas_loja.columns = ['store', 'total_sales', 'average_price', 'price_std']
    vendas_loja['price_std'] = vendas_loja['price_std'].fillna(0)

    variaveis_cluster = vendas_loja[['total_sales', 'average_price', 'price_std']]
    scaler = StandardScaler()
    variaveis_cluster_scaled = scaler.fit_transform(variaveis_cluster)

    # Aplicar K-means com 5 clusters
    kmeans_loja = KMeans(n_clusters=5, random_state=42)
    vendas_loja['cluster'] = kmeans_loja.fit_predict(variaveis_cluster_scaled)

    return vendas_loja

# Função para exibir estatísticas dos clusters de lojas como tabela com Plotly
    cluster_stats = vendas_loja.groupby('cluster').agg({
        'total_sales': ['mean', 'std'],
        'average_price': ['mean', 'std'],
        'price_std': ['mean', 'std']
    }).reset_index()

    # Exibir a tabela formatada com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(cluster_stats.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[cluster_stats[col] for col in cluster_stats.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title="Estatísticas por Cluster de Lojas (5 Clusters)")
    fig.show()

def plot_cluster_lojas(vendas_loja):
    fig = px.scatter(vendas_loja, x='average_price', y='total_sales', color='cluster', 
                     title='Clusterização de Lojas por Preço Médio e Vendas (5 Clusters)', 
                     labels={'average_price': 'Preço Médio', 'total_sales': 'Total de Vendas'},
                     hover_data=['store'])  # Adiciona informações de hover com o nome da loja
    fig.show()



def atribuir_eventos(df):
    # Converter a coluna 'date' para o tipo datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Atribuir eventos baseados nas datas de dezembro
    df.loc[(df['date'].dt.month == 12) & (df['date'].dt.day == 23), 'event'] = 'Natal D-2'
    df.loc[(df['date'].dt.month == 12) & (df['date'].dt.day == 24), 'event'] = 'Natal D-1'
    df.loc[(df['date'].dt.month == 12) & (df['date'].dt.day == 26), 'event'] = 'Boxing Day'

    # Remover linhas onde 'event' é nulo (desconsiderar dias sem eventos)
    df = df[df['event'].notna() & (df['event'] != '')]
    
    return df

# Função para clusterização de eventos
def cluster_eventos(df):
    df = atribuir_eventos(df)

    # Filtrar apenas os eventos que não são nulos
    df_eventos = df[df['event'].notna()]

    # Agrupar as vendas por evento e ano
    df_eventos['year'] = df_eventos['date'].dt.year
    vendas_por_evento_ano = df_eventos.groupby(['year', 'event'])['sales'].sum().reset_index()

    return vendas_por_evento_ano

#



def plot_cluster_eventos(vendas_por_evento_ano):
    fig = px.bar(vendas_por_evento_ano, x='event', y='sales', color='year', 
                 title='Vendas por Evento e Ano',
                 labels={'event': 'Evento', 'sales': 'Total de Vendas', 'year': 'Ano'},
                 hover_data=['year', 'sales'])  # Adiciona informações de hover com o ano e as vendas
    fig.show()
# Função para exibir estatísticas dos clusters de produtos como tabela com Plotly (formatada)
def estatisticas_cluster_produto_preco(vendas_preco_produto):
    # Agrupar as estatísticas de vendas e preço
    cluster_stats = vendas_preco_produto.groupby('cluster').agg({
        'total_sales': ['mean', 'std'],
        'average_price': ['mean', 'std'],
        'price_std': ['mean', 'std']
    }).reset_index()

    # Renomear colunas para português
    cluster_stats.columns = ['Cluster', 'Média Vendas', 'Desvio Padrão Vendas', 'Média Preço', 'Desvio Padrão Preço', 'Média Variação Preço', 'Desvio Padrão Variação Preço']

    # Aplicar formatação com 2 casas decimais
    cluster_stats = cluster_stats.round(2)

    # Adicionar a lista de produtos de cada cluster
    cluster_stats['Produtos'] = vendas_preco_produto.groupby('cluster')['item'].apply(lambda x: ', '.join(x.astype(str)))

    # Exibir a tabela formatada com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(cluster_stats.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[cluster_stats[col] for col in cluster_stats.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title="Estatísticas por Cluster de Produto e Preço (5 Clusters)")
    fig.show()

# Função para exibir estatísticas dos clusters de lojas como tabela com Plotly (formatada)
def estatisticas_cluster_lojas(vendas_loja):
    # Agrupar as estatísticas de vendas e preço
    cluster_stats = vendas_loja.groupby('cluster').agg({
        'total_sales': ['mean', 'std'],
        'average_price': ['mean', 'std'],
        'price_std': ['mean', 'std']
    }).reset_index()

    # Renomear colunas para português
    cluster_stats.columns = ['Cluster', 'Média Vendas', 'Desvio Padrão Vendas', 'Média Preço', 'Desvio Padrão Preço', 'Média Variação Preço', 'Desvio Padrão Variação Preço']

    # Aplicar formatação com 2 casas decimais
    cluster_stats = cluster_stats.round(2)

    # Adicionar a lista de lojas de cada cluster
    cluster_stats['Lojas'] = vendas_loja.groupby('cluster')['store'].apply(lambda x: ', '.join(x.astype(str)))

    # Exibir a tabela formatada com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(cluster_stats.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[cluster_stats[col] for col in cluster_stats.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title="Estatísticas por Cluster de Lojas (5 Clusters)")
    fig.show()

# Função para exibir estatísticas dos clusters de eventos como tabela com Plotly (formatada)
def estatisticas_cluster_eventos(vendas_por_evento_ano):
    # Agrupar as estatísticas de vendas
    cluster_stats = vendas_por_evento_ano.groupby('event').agg({
        'sales': ['mean', 'std']
    }).reset_index()

    # Renomear colunas para português
    cluster_stats.columns = ['Evento', 'Média Vendas', 'Desvio Padrão Vendas']

    # Aplicar formatação com 2 casas decimais
    cluster_stats = cluster_stats.round(2)

    # Adicionar a lista de anos de cada evento
    cluster_stats['Anos'] = vendas_por_evento_ano.groupby('event')['year'].apply(lambda x: ', '.join(x.astype(str)))

    # Exibir a tabela formatada com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(cluster_stats.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[cluster_stats[col] for col in cluster_stats.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title="Estatísticas por Evento")
    fig.show()
    cluster_stats = vendas_por_evento_ano.groupby('event').agg({
        'sales': ['mean', 'std']
    }).reset_index()

    # Renomear colunas para português
    cluster_stats.columns = ['Evento', 'Média Vendas', 'Desvio Padrão Vendas']

    # Aplicar formatação com 2 casas decimais
    cluster_stats = cluster_stats.round(2)

    # Exibir a tabela formatada com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(cluster_stats.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[cluster_stats[col] for col in cluster_stats.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title="Estatísticas por Evento")
    fig.show()

def main():
    df = get_data()  # Obter os dados
    df_clean = clean_data(df)  # Limpar os dados

    # Clusterização de Produtos
    print("\n--- Clusterização de Produtos ---")
    vendas_preco_produto = cluster_produto_preco(df_clean)
    plot_cluster_produto_preco(vendas_preco_produto)
    estatisticas_cluster_produto_preco(vendas_preco_produto)

    # Clusterização de Lojas
    print("\n--- Clusterização de Lojas ---")
    vendas_loja = cluster_lojas(df_clean)
    plot_cluster_lojas(vendas_loja)
    estatisticas_cluster_lojas(vendas_loja)

    # Clusterização de Eventos
    print("\n--- Clusterização de Eventos ---")
    vendas_eventos = cluster_eventos(df_clean)
    plot_cluster_eventos(vendas_eventos)
    estatisticas_cluster_eventos(vendas_eventos)

# Executar o código principal
if __name__ == "__main__":
    main()