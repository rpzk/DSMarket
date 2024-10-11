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

    year = 2012
    month = 1
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
        AND 
        EXTRACT(MONTH FROM date)= {month}
        
    """

    # Executar a consulta e obter o DataFrame
    df_year = client.query(query).to_dataframe()

    return df_year

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

def analise(df):
        # analise do data set

    # Carregar o dataset com o delimitador correto
    data = df



    # Converter a coluna 'date' para o formato datetime para análise temporal
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    # Agrupar as vendas por categoria
    vendas_por_categoria = data.groupby('category')['sales'].sum().reset_index()
    vendas_por_categoria = vendas_por_categoria.sort_values(by='sales', ascending=False)

    # Agrupar as vendas por loja
    vendas_por_loja = data.groupby('store')['sales'].sum().reset_index()
    vendas_por_loja = vendas_por_loja.sort_values(by='sales', ascending=False)

    # Agrupar as vendas por região
    vendas_por_regiao = data.groupby('region')['sales'].sum().reset_index()
    vendas_por_regiao = vendas_por_regiao.sort_values(by='sales', ascending=False)

    # Analisar produtos mais vendidos por região
    produtos_top_por_regiao = data.groupby(['region', 'item'])['sales'].sum().reset_index()
    produtos_top_por_regiao = produtos_top_por_regiao.sort_values(['region', 'sales'], ascending=[True, False])

    # Analisar variação de preço por região
    preco_por_regiao = data.groupby('region')['sell_price'].mean().reset_index()
    preco_por_regiao.columns = ['region', 'preco_medio']
    preco_por_regiao = preco_por_regiao.sort_values(by='preco_medio', ascending=False)

    # Analisar variação de vendas por categoria
    variacao_vendas_categoria = data.groupby('category')['sales'].std().reset_index()
    variacao_vendas_categoria.columns = ['category', 'desvio_padrao_vendas']
    variacao_vendas_categoria = variacao_vendas_categoria.sort_values(by='desvio_padrao_vendas', ascending=False)

    # Analisar quais lojas vendem mais produtos específicos
    produtos_especificos_por_loja = data.groupby(['store', 'item'])['sales'].sum().reset_index()
    produtos_especificos_por_loja = produtos_especificos_por_loja.sort_values(['store', 'sales'], ascending=[True, False])

    # Analisar variação de preço por loja
    variacao_preco_loja = data.groupby('store')['sell_price'].std().reset_index()
    variacao_preco_loja.columns = ['store', 'desvio_padrao_preco']
    variacao_preco_loja = variacao_preco_loja.sort_values(by='desvio_padrao_preco', ascending=False)

    # Comparar vendas por categoria e loja
    vendas_categoria_loja = data.groupby(['category', 'store'])['sales'].sum().reset_index()
    vendas_categoria_loja_pivot = vendas_categoria_loja.pivot(index='category', columns='store', values='sales').fillna(0)

    # Analisar variação de vendas ao longo do tempo
    vendas_tempo = data.groupby('date')['sales'].sum().reset_index()
    fig_vendas_tempo = px.line(vendas_tempo, x='date', y='sales', title='Variação de Vendas ao Longo do Tempo')
    fig_vendas_tempo.show()

    # Análise de vendas sazonais por categoria
    data['month'] = data['date'].dt.month
    vendas_sazonais_categoria = data.groupby(['category', 'month'])['sales'].sum().reset_index()
    fig_vendas_sazonais = px.line(vendas_sazonais_categoria, x='month', y='sales', color='category', title='Tendências Sazonais de Vendas por Categoria')
    fig_vendas_sazonais.show()

    # Agrupar as vendas por tipo de evento para analisar a variação
    vendas_por_evento = data.groupby('event')['sales'].sum().reset_index()

    # Ordenar os eventos pela quantidade de vendas em ordem decrescente
    vendas_por_evento_sorted = vendas_por_evento.sort_values(by='sales', ascending=False)

    # Visualizar as vendas por evento
    fig_vendas_por_evento = px.bar(vendas_por_evento_sorted, x='event', y='sales', title='Vendas por Evento')
    fig_vendas_por_evento.show()

    # Filtrar os dados para eventos que não são nulos
    eventos_com_categorias = data.dropna(subset=['event'])

    # Agrupar as vendas por categoria e evento para identificar quais categorias vendem mais durante eventos
    vendas_categorias_por_evento = eventos_com_categorias.groupby(['event', 'category'])['sales'].sum().reset_index()

    # Ordenar as vendas por evento e por quantidade em ordem decrescente
    vendas_categorias_por_evento_sorted = vendas_categorias_por_evento.sort_values(['event', 'sales'], ascending=[True, False])

    # Visualizar as vendas por categoria durante eventos
    fig_vendas_categorias_por_evento = px.bar(vendas_categorias_por_evento_sorted, x='event', y='sales', color='category', title='Vendas por Categoria Durante Eventos')
    fig_vendas_categorias_por_evento.show()

def verifica_qtd_cluster(df):
    ### clusters  metodo para determinar quantidade de clusters

    # Carregar o dataset com o delimitador correto
    data = df


    # Preparar os dados para clusterização: vendas por loja
    vendas_por_loja_cluster = data.groupby('store')['sales'].sum().reset_index()

    # Padronizar os dados para aplicar K-means
    scaler = StandardScaler()
    vendas_por_loja_cluster_scaled = scaler.fit_transform(vendas_por_loja_cluster[['sales']])

    # Determinar o número ideal de clusters usando o método Elbow
    inertia = []
    range_n_clusters = range(1, 11)  # Testando de 1 a 10 clusters
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(vendas_por_loja_cluster_scaled)
        inertia.append(kmeans.inertia_)

    # Criar um DataFrame com os resultados do método Elbow
    elbow_data = pd.DataFrame({'Número de Clusters': list(range_n_clusters), 'Inércia': inertia})

    # Plotar o gráfico do método Elbow usando Plotly Express
    fig_elbow = px.line(elbow_data, x='Número de Clusters', y='Inércia', title='Método Elbow para Determinar o Número Ideal de Clusters',
                        markers=True)
    fig_elbow.update_layout(xaxis_title='Número de Clusters', yaxis_title='Inércia')
    fig_elbow.show()

def vendas_por_loja(df):
    #Carregar o dataset novamente com o delimitador correto
    data = df

    # Preparar os dados para clusterização: vendas por loja
    vendas_por_loja_cluster = data.groupby('store')['sales'].sum().reset_index()

    # Padronizar os dados para aplicar K-means
    scaler = StandardScaler()
    vendas_por_loja_cluster_scaled = scaler.fit_transform(vendas_por_loja_cluster[['sales']])

    # Aplicar K-means com 2 clusters para as vendas por loja
    kmeans_2_clusters = KMeans(n_clusters=2, random_state=42)
    vendas_por_loja_cluster['cluster'] = kmeans_2_clusters.fit_predict(vendas_por_loja_cluster_scaled)

    # Visualizar os resultados com Plotly Express
    fig_clusters = px.scatter(vendas_por_loja_cluster, x='store', y='sales', color='cluster',
                            title='Clusterização de Vendas por Loja com 2 Clusters',
                            labels={'store': 'Loja', 'sales': 'Vendas'})
    fig_clusters.show()

    # Calcular estatísticas detalhadas para cada cluster
    cluster_stats = vendas_por_loja_cluster.groupby('cluster')['sales'].describe().reset_index()

    # Exibir as estatísticas detalhadas para cada cluster
    print(cluster_stats)
    # Filtrar as lojas que pertencem ao cluster 1
    lojas_cluster_1 = vendas_por_loja_cluster[vendas_por_loja_cluster['cluster'] == 1]
    lojas_cluster_0 = vendas_por_loja_cluster[vendas_por_loja_cluster['cluster'] == 0]
    # Exibir as lojas que estão no cluster 1
    Print(lojas_cluster_1)
    print(lojas_cluster_0)

    # Calcular a média de vendas por cluster
    media_vendas_por_cluster = vendas_por_loja_cluster.groupby('cluster')['sales'].mean().reset_index()

    # Renomear colunas para clareza
    media_vendas_por_cluster.columns = ['cluster', 'media_vendas']

    # Exibir a média de vendas por cluster
    print(media_vendas_por_cluster)


    fig_distribuicao_vendas = px.histogram(vendas_por_loja_cluster, x='sales', nbins=20,
                                        title='Distribuição de Vendas por Loja',
                                        labels={'sales': 'Vendas Totais por Loja'})
    fig_distribuicao_vendas.show()

def cluster_cat_loja(df):

    #1. **Clusterização por Categoria e Loja**:

    # Salvar uma amostra ainda menor do dataset localmente
    data_sample_min = df

    # Agrupar as vendas por "category_x" (categoria) e loja para o Método 1 com a amostra reduzida
    vendas_por_categoria_loja_min = data_sample_min.groupby(['category', 'store'])['sales'].sum().reset_index()

    # Pivotar os dados para ter uma matriz onde cada linha é uma loja e cada coluna é uma categoria, preenchendo com 0 os valores ausentes
    vendas_pivot_min = vendas_por_categoria_loja_min.pivot(index='store', columns='category', values='sales').fillna(0)

    # Padronizar os dados para aplicar K-means
    scaler = StandardScaler()
    vendas_pivot_scaled_min = scaler.fit_transform(vendas_pivot_min)

    # Aplicar K-means com 3 clusters para o exemplo
    kmeans_categoria_loja_min = KMeans(n_clusters=3, random_state=42)
    vendas_pivot_min['cluster'] = kmeans_categoria_loja_min.fit_predict(vendas_pivot_scaled_min)

    # Visualizar os resultados com Plotly Express
    fig_clusters_categoria_loja_min = px.scatter(vendas_pivot_min.reset_index(), x='ACCESORIES', y='SUPERMARKET', 
                                                color='cluster', title='Clusterização por Categoria e Loja (Amostra de 10.000)',
                                                labels={'ACCESORIES': 'Vendas de Acessórios', 'SUPERMARKET': 'Vendas de Supermercado'},
                                                hover_data=['store'])
    fig_clusters_categoria_loja_min.show()


def cluster_reg_periodo(df):
    #2. **Clusterização por Região e Período (Sazonalidade)**:
    # Converter a coluna 'date' para datetime para extrair o mês
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Extrair o mês e o ano da coluna 'date' para usar na análise sazonal
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Agrupar as vendas por região e mês
    vendas_por_regiao_mes = df_tratado.groupby(['region', 'year', 'month'])['sales'].sum().reset_index()

    # Pivotar os dados para ter uma matriz onde cada linha é uma região e cada coluna é um período (mês)
    vendas_pivot_regiao_mes = vendas_por_regiao_mes.pivot_table(index='region', columns=['year', 'month'], values='sales').fillna(0)

    # Padronizar os dados para aplicar K-means
    scaler = StandardScaler()
    vendas_pivot_regiao_mes_scaled = scaler.fit_transform(vendas_pivot_regiao_mes)

    # Aplicar K-means com 3 clusters como exemplo
    kmeans_regiao_mes = KMeans(n_clusters=3, random_state=42)
    vendas_pivot_regiao_mes['cluster'] = kmeans_regiao_mes.fit_predict(vendas_pivot_regiao_mes_scaled)

    # Exibir as estatísticas detalhadas para cada cluster
    cluster_stats_regiao_mes = vendas_pivot_regiao_mes.groupby('cluster').describe()

    print(cluster_stats_regiao_mes)

def produto_preco(df):
    #3. **Clusterização por Produto e Preço**:
    # Agrupar as vendas e calcular o preço médio por produto
    vendas_preco_produto = df.groupby('item').agg({'sales': 'sum', 'sell_price': ['mean', 'std']}).reset_index()
    vendas_preco_produto.columns = ['item', 'total_sales', 'average_price', 'price_std']

    # Substituir valores nulos na coluna de desvio padrão de preço por 0 (caso existam produtos com preço constante)
    vendas_preco_produto['price_std'] = vendas_preco_produto['price_std'].fillna(0)

    # Selecionar as variáveis para a clusterização
    variaveis_cluster = vendas_preco_produto[['total_sales', 'average_price', 'price_std']]

    # Substituir valores NaN nas variáveis 'total_sales', 'average_price', e 'price_std' pela média de cada coluna
    variaveis_cluster.fillna(variaveis_cluster.mean(), inplace=True)

    # Padronizar as variáveis para aplicar K-means
    scaler = StandardScaler()
    variaveis_cluster_scaled = scaler.fit_transform(variaveis_cluster)

    # Aplicar K-means com 3 clusters como exemplo
    kmeans_produto_preco = KMeans(n_clusters=3, random_state=42)
    vendas_preco_produto['cluster'] = kmeans_produto_preco.fit_predict(variaveis_cluster_scaled)

    # Exibir as estatísticas detalhadas para cada cluster
    cluster_stats_produto_preco = vendas_preco_produto.groupby('cluster').describe()
    print(cluster_stats_produto_preco)

def main():
    df = get_data()
    df_clean = clean_data(df)
    analise(df_clean)
    verifica_qtd_cluster(df_clean)
    cluster_cat_loja(df_clean)
    cluster_reg_periodo(df_clean)
    produto_preco(df_clean)
    
if __name__ == '__main__':
    main()