# %% [markdown]
# # Análise de Vendas no GCP com BigQuery - Lucas

# %% [markdown]
# Este notebook realiza diversas consultas SQL em uma base de dados grande hospedada no Google Cloud Platform (GCP) utilizando o BigQuery. As consultas abrangem análises de vendas por região, departamento, categoria, loja e outros parâmetros relevantes.

# %% [markdown]
# ## 1. Configuração e Importações

# %%
import os
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Integer, String, Float, DateTime
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Configurações para visualizações
%matplotlib inline
sns.set(style="whitegrid")

# %% [markdown]
# ## 2. Autenticação e Conexão com o BigQuery

# %%
# Caminho para o arquivo de chave da conta de serviço
service_account_path = 'tfm-sa.json'

# Criar objeto de credenciais
credentials = service_account.Credentials.from_service_account_file(service_account_path)

# Configurações do projeto e dataset
project_id = 'perseverance-332400'
dataset_id = 'TFM'
table_id = 'ds_market'
full_table_id = f'{project_id}.{dataset_id}.{table_id}'

# Criar cliente BigQuery
client = bigquery.Client(project=project_id, credentials=credentials)

# %% [markdown]
# ## 3. Listagem e Visualização das Colunas e Dados

# %% [markdown]
# ### 3.1. Listando as Colunas Disponíveis na Tabela

# %%
# Obter o esquema da tabela
table = client.get_table(full_table_id)  # Faz uma chamada API para obter a tabela

# Extrair os nomes das colunas
columns = [schema_field.name for schema_field in table.schema]

# Exibir as colunas
print("Colunas disponíveis na tabela `TFM.ds_market`:")
for column in columns:
    print(f"- {column}")

# %% [markdown]
# ### 3.2. Obtendo um Conjunto de Dados de Amostra

# %%
# Definir uma consulta para obter as primeiras 10 linhas da tabela
sample_query = f"""
SELECT *
FROM `{full_table_id}`
LIMIT 10;
"""

# Definir a função para executar consultas
def executar_consulta(sql, client):
    """
    Executa uma consulta SQL no BigQuery e retorna um DataFrame do Pandas.
    
    Args:
        sql (str): A consulta SQL a ser executada.
        client (bigquery.Client): Cliente do BigQuery.

    Returns:
        pd.DataFrame: Resultado da consulta.
    """
    query_job = client.query(sql)
    return query_job.to_dataframe()

# Executar a consulta
df_sample = executar_consulta(sample_query, client)

# Exibir o DataFrame de amostra
df_sample

# %% [markdown]
# ### 3.3. Inspecionando os Tipos de Dados das Colunas

# %%
print("Esquema da tabela `TFM.ds_market`:")
for schema_field in table.schema:
    print(f"- {schema_field.name}: {schema_field.field_type}")

# %% [markdown]
# ## 4. Definição das Consultas SQL Refinadas

# %%
# Consulta 1: Total de vendas por loja, região e departamento com contribuição percentual
query_total_vendas_loja = """
SELECT
  region,
  department,
  store,
  SUM(sales) AS total_sales,
  (SUM(sales) / SUM(SUM(sales)) OVER (PARTITION BY region, department)) * 100 AS sales_contribution_percentage
FROM
  `TFM.ds_market`
GROUP BY
  region,
  department,
  store;
"""

# Consulta 2 Refinada: Total de vendas por category_x, região e semana com contribuição percentual
query_vendas_category = """
SELECT
  region,
  yearweek,
  category_x,
  SUM(sales) AS total_sales,
  (SUM(sales) / SUM(SUM(sales)) OVER (PARTITION BY region, yearweek)) * 100 AS sales_contribution_percentage
FROM
  `TFM.ds_market`
GROUP BY
  region,
  yearweek,
  category_x;
"""

# Consulta 3: Os 5 itens com as maiores vendas para uma determinada região e semana do ano, com contribuição percentual
query_top_5_itens = """
SELECT
  region,
  yearweek,
  item,
  SUM(sales) AS total_sales,
  (SUM(sales) / SUM(SUM(sales)) OVER (PARTITION BY region, yearweek)) * 100 AS sales_contribution_percentage
FROM
  `TFM.ds_market`
WHERE
  region = 'US'
  AND yearweek = '201552'
GROUP BY
  region,
  yearweek,
  item
ORDER BY
  total_sales DESC
LIMIT
  5;
"""

# Consulta 4: Total de vendas por loja, região e semana do ano com variação percentual ano a ano
query_total_vendas_loja_variacao = """
SELECT
  region,
  yearweek,
  store,
  SUM(sales) AS total_sales,
  (SUM(sales) / SUM(SUM(sales)) OVER (PARTITION BY region)) * 100 AS sales_contribution_percentage
FROM
  `TFM.ds_market`
GROUP BY
  region,
  yearweek,
  store;
"""

# Consulta 5: As 5 lojas com as maiores vendas médias para uma determinada região e semana do ano
query_top_5_lojas_media = """
SELECT
  region,
  yearweek,
  store,
  AVG(sales) AS avg_sales
FROM
  `TFM.ds_market`
WHERE
  region = 'US'
  AND yearweek = '201552'
GROUP BY
  region,
  yearweek,
  store
ORDER BY
  avg_sales DESC
LIMIT
  5;
"""

# Consulta 6: Correlação entre vendas e preço de venda para cada região
query_correlacao_vendas_preco = """
SELECT
  region,
  CORR(sales, sell_price) AS sales_sell_price_corr
FROM
  `TFM.ds_market`
GROUP BY
  region;
"""

# Consulta 7: Os 5 itens com as maiores vendas para uma determinada região e semana do ano
query_top_5_itens_vendas = """
SELECT
  item,
  SUM(sales) AS total_sales
FROM
  `TFM.ds_market`
WHERE
  region = 'US'
  AND yearweek = '201552'
GROUP BY
  item
ORDER BY
  total_sales DESC
LIMIT
  5;
"""

# Consulta 8: Total de vendas por loja, região e departamento
query_total_vendas_loja_departamento = """
SELECT
  region,
  department,
  SUM(sales) AS total_sales
FROM
  `TFM.ds_market`
GROUP BY
  region,
  department;
"""

# Consulta 9 Refinada: Preço de venda máximo para cada região e loja
query_preco_max_category = """
SELECT
  region,
  store,
  MAX(sell_price) AS max_sell_price
FROM
  `TFM.ds_market`
GROUP BY
  region,
  store;
"""

# Consulta 10: Desvio padrão das vendas para cada loja, agrupadas por região e ano-semana
query_stddev_vendas = """
SELECT
  region,
  yearweek,
  store,
  STDDEV(sales) AS stddev_sales
FROM
  `TFM.ds_market`
GROUP BY
  region,
  yearweek,
  store;
"""

# Consulta 11: Média de vendas por região para cada ano-semana
query_media_vendas_regiao = """
SELECT
  region,
  yearweek,
  AVG(sales) AS average_sales
FROM
  `TFM.ds_market`
WHERE
  date BETWEEN '2022-01-01' AND '2022-12-31'
GROUP BY
  region,
  yearweek;
"""

# Consulta 12: Desvio padrão das vendas por departamento para o ano de 2022
query_stddev_departamento = """
SELECT
  department,
  STDDEV_SAMP(sales) AS sales_standard_deviation
FROM
  `TFM.ds_market`
WHERE
  date BETWEEN '2022-01-01' AND '2022-12-31'
GROUP BY
  department;
"""

# Consulta 13: 5 itens com o maior preço médio de venda
query_top_5_preco_medio = """
SELECT
  item,
  AVG(sell_price) AS average_sell_price
FROM
  `TFM.ds_market`
WHERE
  date BETWEEN '2022-01-01' AND '2022-12-31'
GROUP BY
  item
ORDER BY
  average_sell_price DESC
LIMIT
  5;
"""

# Consulta 14: Correlação entre vendas e preço de venda para o ano de 2022
query_correlacao_2022 = """
SELECT
  CORR(sales, sell_price) AS sales_sell_price_correlation
FROM
  `TFM.ds_market`
WHERE
  date BETWEEN '2022-01-01' AND '2022-12-31';
"""

# Consulta 15: Média de vendas por loja para cada combinação de região e ano-semana
query_media_vendas_loja = """
SELECT
  region,
  yearweek,
  store,
  AVG(sales) AS average_sales
FROM
  `TFM.ds_market`
WHERE
  date BETWEEN '2022-01-01' AND '2022-12-31'
GROUP BY
  region,
  yearweek,
  store;
"""

# Consulta 16: Preço médio de venda para cada combinação de category_x e código da loja
query_preco_medio_categoria_loja = """
SELECT
  category_x,
  store_code,
  AVG(sell_price) AS avg_sell_price
FROM
  `TFM.ds_market`
GROUP BY
  category_x,
  store_code;
"""

# Consulta 17: Variação percentual das vendas para cada ano-semana e category_x
query_variacao_percentual_vendas = """
SELECT
  region,
  yearweek,
  sales,
  (sales - LAG(sales, 1, 0) OVER (PARTITION BY region ORDER BY yearweek)) * 100.0 / LAG(sales, 1, 0) OVER (PARTITION BY region ORDER BY yearweek) AS sales_change_percent
FROM
  `TFM.ds_market`
WHERE
  yearweek > '201501';
"""

# Consulta 18 Refinada: Vendas acumuladas para cada loja ao longo do tempo
query_vendas_acumuladas = """
SELECT
  store_code,
  date,
  SUM(sales) OVER (PARTITION BY store_code ORDER BY date) AS cumulative_sales
FROM
  `TFM.ds_market`
ORDER BY
  store_code,
  date;
"""

# Consulta 19: Média móvel de 7 dias das vendas para cada região
query_media_movel_7dias = """
SELECT
  region,
  date,
  AVG(sales) OVER (PARTITION BY region ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS moving_avg_sales
FROM
  `TFM.ds_market`
ORDER BY
  region,
  date;
"""

# Consulta 20: Total de vendas por evento e departamento com classificação dentro de cada evento
query_vendas_evento_departamento = """
SELECT
  event,
  department,
  SUM(sales) AS total_sales,
  RANK() OVER (PARTITION BY event ORDER BY SUM(sales) DESC) AS sales_rank
FROM
  `TFM.ds_market`
GROUP BY
  event,
  department
ORDER BY
  event,
  sales_rank;
"""

# Consulta 21: As 3 regiões com o maior total de vendas para uma categoria específica
query_top_3_regioes_categoria = """
SELECT
  region,
  SUM(sales) AS total_sales
FROM
  `TFM.ds_market`
WHERE
  category_x = 'ACCESORIES'
GROUP BY
  region
ORDER BY
  total_sales DESC
LIMIT
  3;
"""

# Consulta 22: As 5 lojas com o maior preço médio de venda para um departamento específico
query_top_5_lojas_preco_medio = """
SELECT
  store_code,
  AVG(sell_price) AS avg_sell_price
FROM
  `TFM.ds_market`
WHERE
  department = 'ACCESORIES_1'
GROUP BY
  store_code
ORDER BY
  avg_sell_price DESC
LIMIT
  5;
"""

# Consulta 23: Os 3 eventos com o maior total de vendas para um item específico
query_top_3_eventos_item = """
SELECT
  event,
  SUM(sales) AS total_sales
FROM
  `TFM.ds_market`
WHERE
  item = 'ACCESORIES_1_001'
GROUP BY
  event
ORDER BY
  total_sales DESC
LIMIT
  3;
"""

# Consulta 24: As 5 categorias com a maior média de vendas para uma loja específica
query_top_5_categorias_media_loja = """
SELECT
  category_x,
  AVG(sales) AS avg_sales
FROM
  `TFM.ds_market`
WHERE
  store_code = 'BOS_1'
GROUP BY
  category_x
ORDER BY
  avg_sales DESC
LIMIT
  5;
"""

# %% [markdown]
# ## 5. Execução das Consultas Refinadas e Armazenamento dos Resultados

# %%
# Executando a Consulta 1
df_total_vendas_loja = executar_consulta(query_total_vendas_loja, client)
print("Consulta 1: Total de Vendas por Loja, Região e Departamento")
df_total_vendas_loja.head()

# %%
# Executando a Consulta 2 Refinada
df_vendas_category = executar_consulta(query_vendas_category, client)
print("Consulta 2: Total de Vendas por Category_X, Região e Semana")
df_vendas_category.head()

# %%
# Executando a Consulta 3
df_top_5_itens = executar_consulta(query_top_5_itens, client)
print("Consulta 3: Top 5 Itens com Maiores Vendas")
df_top_5_itens.head()

# %%
# Executando a Consulta 4
df_total_vendas_loja_variacao = executar_consulta(query_total_vendas_loja_variacao, client)
print("Consulta 4: Total de Vendas por Loja com Variação Percentual")
df_total_vendas_loja_variacao.head()

# %%
# Executando a Consulta 5
df_top_5_lojas_media = executar_consulta(query_top_5_lojas_media, client)
print("Consulta 5: Top 5 Lojas com Maiores Vendas Médias")
df_top_5_lojas_media.head()

# %%
# Executando a Consulta 6
df_correlacao_vendas_preco = executar_consulta(query_correlacao_vendas_preco, client)
print("Consulta 6: Correlação entre Vendas e Preço de Venda por Região")
df_correlacao_vendas_preco.head()

# %%
# Executando a Consulta 7
df_top_5_itens_vendas = executar_consulta(query_top_5_itens_vendas, client)
print("Consulta 7: Top 5 Itens com Maiores Vendas para US na Semana 201552")
df_top_5_itens_vendas.head()

# %%
# Executando a Consulta 8
df_total_vendas_loja_departamento = executar_consulta(query_total_vendas_loja_departamento, client)
print("Consulta 8: Total de Vendas por Loja, Região e Departamento")
df_total_vendas_loja_departamento.head()

# %%
# Executando a Consulta 9 Refinada
df_preco_max_category = executar_consulta(query_preco_max_category, client)
print("Consulta 9: Preço de Venda Máximo por Região e Loja")
df_preco_max_category.head()

# %%
# Executando a Consulta 10
df_stddev_vendas = executar_consulta(query_stddev_vendas, client)
print("Consulta 10: Desvio Padrão das Vendas por Loja, Região e Ano-Semana")
df_stddev_vendas.head()

# %%
# Executando a Consulta 11
df_media_vendas_regiao = executar_consulta(query_media_vendas_regiao, client)
print("Consulta 11: Média de Vendas por Região para Cada Ano-Semana")
df_media_vendas_regiao.head()

# %%
# Executando a Consulta 12
df_stddev_departamento = executar_consulta(query_stddev_departamento, client)
print("Consulta 12: Desvio Padrão das Vendas por Departamento para 2022")
df_stddev_departamento.head()

# %%
# Executando a Consulta 13
df_top_5_preco_medio = executar_consulta(query_top_5_preco_medio, client)
print("Consulta 13: Top 5 Itens com Maior Preço Médio de Venda")
df_top_5_preco_medio.head()

# %%
# Executando a Consulta 14
df_correlacao_2022 = executar_consulta(query_correlacao_2022, client)
print("Consulta 14: Correlação entre Vendas e Preço de Venda para 2022")
df_correlacao_2022.head()

# %%
# Executando a Consulta 15
df_media_vendas_loja = executar_consulta(query_media_vendas_loja, client)
print("Consulta 15: Média de Vendas por Loja para Cada Região e Ano-Semana")
df_media_vendas_loja.head()

# %%
# Executando a Consulta 16
df_preco_medio_categoria_loja = executar_consulta(query_preco_medio_categoria_loja, client)
print("Consulta 16: Preço Médio de Venda por Category_X e Código da Loja")
df_preco_medio_categoria_loja.head()

# %%
# Executando a Consulta 17
df_variacao_percentual_vendas = executar_consulta(query_variacao_percentual_vendas, client)
print("Consulta 17: Variação Percentual das Vendas por Ano-Semana e Category_X")
df_variacao_percentual_vendas.head()

# %%
# Executando a Consulta 18 Refinada
df_vendas_acumuladas = executar_consulta(query_vendas_acumuladas, client)
print("Consulta 18: Vendas Acumuladas por Código da Loja ao Longo do Tempo")
df_vendas_acumuladas.head()

# %%
# Executando a Consulta 19
df_media_movel_7dias = executar_consulta(query_media_movel_7dias, client)
print("Consulta 19: Média Móvel de 7 Dias das Vendas por Região")
df_media_movel_7dias.head()

# %%
# Executando a Consulta 20
df_vendas_evento_departamento = executar_consulta(query_vendas_evento_departamento, client)
print("Consulta 20: Total de Vendas por Evento e Departamento com Classificação")
df_vendas_evento_departamento.head()

# %%
# Executando a Consulta 21
df_top_3_regioes_categoria = executar_consulta(query_top_3_regioes_categoria, client)
print("Consulta 21: Top 3 Regiões com Maior Total de Vendas para 'ACCESORIES'")
df_top_3_regioes_categoria.head()

# %%
# Executando a Consulta 22
df_top_5_lojas_preco_medio = executar_consulta(query_top_5_lojas_preco_medio, client)
print("Consulta 22: Top 5 Lojas com Maior Preço Médio de Venda para 'ACCESORIES_1'")
df_top_5_lojas_preco_medio.head()

# %%
# Executando a Consulta 23
df_top_3_eventos_item = executar_consulta(query_top_3_eventos_item, client)
print("Consulta 23: Top 3 Eventos com Maior Total de Vendas para 'ACCESORIES_1_001'")
df_top_3_eventos_item.head()

# %%
# Executando a Consulta 24
df_top_5_categorias_media_loja = executar_consulta(query_top_5_categorias_media_loja, client)
print("Consulta 24: Top 5 Categorias com Maior Média de Vendas para 'BOS_1'")
df_top_5_categorias_media_loja.head()

# %% [markdown]
# ## 6. Análise e Visualização dos Dados

# %%
# Exemplo de visualização: Total de Vendas por Região e Departamento
plt.figure(figsize=(12, 8))
sns.barplot(data=df_total_vendas_loja, x='region', y='total_sales', hue='department')
plt.title('Total de Vendas por Região e Departamento')
plt.xlabel('Região')
plt.ylabel('Total de Vendas')
plt.legend(title='Departamento')
plt.show()

# %%
# Exemplo de visualização: Correlação entre Vendas e Preço de Venda por Região
plt.figure(figsize=(10, 6))
sns.barplot(data=df_correlacao_vendas_preco, x='region', y='sales_sell_price_corr')
plt.title('Correlação entre Vendas e Preço de Venda por Região')
plt.xlabel('Região')
plt.ylabel('Correlação')
plt.show()

# %%
# Exemplo de visualização: Top 5 Itens com Maiores Vendas para US na Semana 201552
plt.figure(figsize=(10, 6))
sns.barplot(data=df_top_5_itens, x='item', y='total_sales')
plt.title('Top 5 Itens com Maiores Vendas para US na Semana 201552')
plt.xlabel('Item')
plt.ylabel('Total de Vendas')
plt.show()

# %%
# Distribuição das Vendas
plt.figure(figsize=(8,6))
sns.histplot(df_sample['sales'], bins=10, kde=True)
plt.title('Distribuição das Vendas (Amostra)')
plt.xlabel('Vendas')
plt.ylabel('Frequência')
plt.show()

# %%
# Relação entre Preço de Venda e Vendas
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_sample, x='sell_price', y='sales', hue='region')
plt.title('Relação entre Preço de Venda e Vendas (Amostra)')
plt.xlabel('Preço de Venda')
plt.ylabel('Vendas')
plt.show()

# %% [markdown]
# ## 7. Modelos de Machine Learning

# %% [markdown]
# ### 7.1. BigQuery ML - Previsão de Vendas Futuras

# %% [markdown]
# #### 7.1.1. Criando e Treinando um Modelo de Classificação Logística com BigQuery ML

# %%
# Definir a consulta para criar o modelo de classificação logística
create_model_query = """
CREATE OR REPLACE MODEL `TFM.sales_predict_model`
OPTIONS(
  model_type='logistic_reg',
  input_label_cols=['sales']
) AS
SELECT
  sell_price,
  category_x,
  department,
  region,
  event,
  yearweek,
  d,
  store_code
FROM
  `TFM.ds_market`
WHERE
  date BETWEEN '2022-01-01' AND '2023-12-31';
"""

# Executar a consulta para criar o modelo
client.query(create_model_query).result()
print("Modelo de Classificação Logística criado com sucesso.")

# %%
# Definir a consulta para fazer previsões
predict_query = """
SELECT
  sell_price,
  category_x,
  department,
  region,
  event,
  yearweek,
  d,
  store_code,
  predicted_sales
FROM
  ML.PREDICT(MODEL `TFM.sales_predict_model`,
    (
      SELECT
        sell_price,
        category_x,
        department,
        region,
        event,
        yearweek,
        d,
        store_code
      FROM
        `TFM.ds_market`
      WHERE
        date BETWEEN '2024-01-01' AND '2024-12-31'
    )
  );
"""

# Executar a consulta de previsão
df_predictions = executar_consulta(predict_query, client)
print("Previsões realizadas com sucesso.")
df_predictions.head()

# %% [markdown]
# ### 7.2. scikit-learn - Classificação de Vendas

# %%
# Criando e Treinando um Modelo de Classificação com scikit-learn
# Carregar os dados necessários para o modelo
query_ml_data = """
SELECT
  sell_price,
  category_x,
  department,
  region,
  event,
  yearweek,
  d,
  store_code,
  sales
FROM
  `TFM.ds_market`
WHERE
  date BETWEEN '2022-01-01' AND '2023-12-31';
"""

df_ml = executar_consulta(query_ml_data, client)

# Remover registros com sell_price faltante
df_ml = df_ml.dropna(subset=['sell_price'])

# Definir Features e Target
X = df_ml[['sell_price', 'category_x', 'department', 'region', 'event', 'yearweek', 'd', 'store_code']]
y = df_ml['sales']

# Dividir em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definir colunas categóricas e numéricas
categorical_features = ['category_x', 'department', 'region', 'event', 'yearweek', 'd', 'store_code']
numeric_features = ['sell_price']

# Criar o pré-processamento para colunas categóricas e numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Criar o pipeline com pré-processamento e modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Fazer previsões
y_pred = pipeline.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# %%
from jupyter_dash import JupyterDash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Inicializar o aplicativo JupyterDash
app = JupyterDash(__name__)

# Definir o layout do dashboard
app.layout = html.Div([
    html.H1("Dashboard de Vendas"),
    html.Div([
        html.Label("Selecione a Região:"),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': region, 'value': region} for region in df_total_vendas_loja['region'].unique()],
            value='US',
            clearable=False
        )
    ], style={'width': '25%', 'display': 'inline-block', 'padding': '20px'}),
    dcc.Graph(id='vendas-por-departamento')
])

# Definir as callbacks para atualizar o gráfico
@app.callback(
    Output('vendas-por-departamento', 'figure'),
    [Input('region-dropdown', 'value')]
)
def update_graph(selected_region):
    # Filtrar os dados com base na região selecionada
    filtered_df = df_total_vendas_loja[df_total_vendas_loja['region'] == selected_region]
    
    # Criar o gráfico de barras
    fig = px.bar(
        filtered_df,
        x='department',
        y='total_sales',
        title=f'Total de Vendas por Departamento na Região {selected_region}',
        labels={'department': 'Departamento', 'total_sales': 'Total de Vendas'}
    )
    return fig

# Executar o aplicativo no modo inline
app.run_server(mode='inline')
