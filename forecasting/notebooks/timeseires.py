# forecasting.py

# %% [markdown]
# # Forecasting com Eventos (Incluindo Natal)

# %% [markdown]
# ## Importações e Configurações

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from dotenv import load_dotenv
import seaborn as sns

# Bibliotecas para modelagem
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

# Configurar parâmetros de exibição
%matplotlib inline
sns.set(style='whitegrid')

# %% [markdown]
# ## Conexão com o Banco de Dados

# %%
# Carregar variáveis de ambiente
load_dotenv()

# Obter variáveis de ambiente
user = os.environ.get('DB_USER')
password = os.environ.get('DB_PASSWORD')
host = os.environ.get('DB_HOST')
port = os.environ.get('DB_PORT', '5432')
database = os.environ.get('DB_NAME')

# Criar string de conexão
connection_string = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'

# Nome da tabela
table_name = 'ds_market'

# Criar conexão com o banco de dados
engine = create_engine(connection_string)

# %% [markdown]
# ## Carregamento e Preparação dos Dados

# %%
# Carregar dados da tabela 'ds_market'
query = f'SELECT * FROM {table_name}'
df = pd.read_sql(query, engine)

# Garantir que a coluna 'date' esteja em formato de data
df['date'] = pd.to_datetime(df['date'])

# Definir a coluna 'date' como índice
df.set_index('date', inplace=True)

# Verificar valores faltantes
print("Valores faltantes por coluna:")
print(df.isnull().sum())

# %% [markdown]
# ## Inclusão do Evento "Natal"

# %%
def adicionar_eventos(df):
    if 'event' not in df.columns:
        df['event'] = None
    natal_dates = pd.to_datetime([
        '2011-12-25', '2012-12-25', '2013-12-25',
        '2014-12-25', '2015-12-25', '2016-12-25'
    ])
    df.loc[df.index.isin(natal_dates), 'event'] = 'Natal'
    return df

# Adicionar o evento "Natal"
df = adicionar_eventos(df)

# Codificar a coluna 'event' usando One-Hot Encoding
df = pd.get_dummies(df, columns=['event'], dummy_na=True)

# %% [markdown]
# ## Selecionar os Top 10 Itens Mais Vendidos

# %%
# Calcular as vendas totais por item
top_items = df.groupby('item')['sales'].sum().sort_values(ascending=False).head(10).index

# Filtrar o DataFrame para incluir apenas os top 10 itens
df_top_items = df[df['item'].isin(top_items)]

# %% [markdown]
# ## Análise por Loja

# %%
# Selecionar uma loja específica (por exemplo, 'Queen_Village')
store_name = 'Queen_Village'
df_store = df_top_items[df_top_items['store'] == store_name]

# %% [markdown]
# ## Agregar Dados Mensalmente e Preparar Variáveis Exógenas

# %%
# Agregar as vendas mensalmente por item
monthly_sales = df_store.groupby([pd.Grouper(freq='M'), 'item']).sum().reset_index()

# Identificar variáveis exógenas (eventos)
exog_vars = [col for col in df.columns if 'event_' in col]

# Agregar variáveis exógenas mensalmente
monthly_exog = df_store[exog_vars].resample('M').sum().reset_index()

# Combinar vendas agregadas com variáveis exógenas
monthly_sales = monthly_sales.merge(monthly_exog, on='date', how='left')

# Separar ano e mês para facilitar
monthly_sales['year'] = monthly_sales['date'].dt.year
monthly_sales['month'] = monthly_sales['date'].dt.month

# %% [markdown]
# ## Funções Auxiliares

# %%
def verificar_estacionariedade(series):
    result = adfuller(series)
    print('Estatística ADF:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critério {}: {}'.format(key, value))

def preparar_variaveis_exogenas_futuras(exog_vars, start_date, periods, freq='M'):
    future_dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    future_exog = pd.DataFrame(index=future_dates)
    for var in exog_vars:
        future_exog[var] = 0  # Inicialmente zero
    # Ajustar eventos futuros (por exemplo, Natal em dezembro)
    future_exog['event_Natal'] = 0
    future_exog.loc[future_exog.index.month == 12, 'event_Natal'] = 1
    return future_exog

# %% [markdown]
# ## Função para Treinar o Modelo SARIMAX e Fazer Previsões

# %%
def train_and_forecast(item_df, item_name):
    # Configurar o índice de data
    item_df = item_df.set_index('date')
    
    # Separar as variáveis exógenas (eventos)
    exog_vars = [col for col in item_df.columns if 'event_' in col]
    
    # Dividir em treino e teste
    train = item_df[item_df['year'] < 2016]
    test = item_df[item_df['year'] == 2016]
    
    # Preparar as variáveis dependentes e independentes
    y_train = train['sales']
    X_train = train[exog_vars]
    y_test = test['sales']
    X_test = test[exog_vars]
    
    # Verificar estacionariedade
    print(f'\nVerificando estacionariedade para o item: {item_name}')
    verificar_estacionariedade(y_train)
    
    # Selecionar o melhor modelo usando auto_arima
    print(f'\nSelecionando o melhor modelo para o item: {item_name}')
    stepwise_fit = auto_arima(
        y_train,
        exogenous=X_train,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        m=12,  # Sazonalidade anual
        seasonal=True,
        d=1, D=1,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print(stepwise_fit.summary())
    
    # Fazer previsões no conjunto de teste
    y_pred = stepwise_fit.predict(n_periods=len(y_test), exogenous=X_test)
    
    # Avaliar o modelo
    mse = ((y_pred - y_test.values) ** 2).mean()
    mae = np.mean(np.abs(y_pred - y_test.values))
    print(f'\nItem: {item_name} - MSE: {mse:.2f}, MAE: {mae:.2f}')
    
    # Plotar resultados
    plt.figure(figsize=(12,6))
    plt.plot(y_train.index, y_train, label='Treino')
    plt.plot(y_test.index, y_test, label='Teste')
    plt.plot(y_test.index, y_pred, label='Previsão')
    plt.title(f'Previsão de Vendas para {item_name}')
    plt.legend()
    plt.show()
    
    # Fazer previsão para os próximos 12 meses
    future_exog = preparar_variaveis_exogenas_futuras(
        exog_vars, start_date=y_test.index[-1] + pd.DateOffset(months=1), periods=12
    )
    future_forecast = stepwise_fit.predict(n_periods=12, exogenous=future_exog)
    
    # Criar DataFrame com as previsões futuras
    future_dates = future_exog.index
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': future_forecast})
    forecast_df.set_index('date', inplace=True)
    
    # Plotar previsões futuras
    plt.figure(figsize=(12,6))
    plt.plot(y_train.index, y_train, label='Treino')
    plt.plot(y_test.index, y_test, label='Teste')
    plt.plot(y_test.index, y_pred, label='Previsão')
    plt.plot(forecast_df.index, forecast_df['forecast'], label='Previsão Futura')
    plt.title(f'Previsão de Vendas Futura para {item_name}')
    plt.legend()
    plt.show()
    
    return forecast_df

# %% [markdown]
# ## Aplicar a Função para Cada Item

# %%
# Criar um dicionário para armazenar as previsões futuras
future_forecasts = {}

for item in top_items:
    print(f'\nTreinando modelo para o item: {item}')
    item_data = monthly_sales[monthly_sales['item'] == item]
    forecast = train_and_forecast(item_data, item)
    future_forecasts[item] = forecast

# %% [markdown]
# ## Análise dos Resultados

# %%
# Exibir a previsão futura para um item específico
item_to_view = top_items[0]  # Primeiro item da lista
print(f'\nPrevisão para os próximos 12 meses para o item {item_to_view}:')
print(future_forecasts[item_to_view])

# %% [markdown]
# ## Previsão com Prophet

# %%
# Instalar o Prophet, se necessário
%pip install prophet

from prophet import Prophet

def train_and_forecast_prophet(item_df, item_name):
    # Preparar o DataFrame no formato esperado pelo Prophet
    df_prophet = item_df.reset_index()[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Adicionar eventos como regressores
    exog_vars = [col for col in item_df.columns if 'event_' in col]
    for var in exog_vars:
        df_prophet[var] = item_df[var].values
    
    # Dividir em treino e teste
    train = df_prophet[df_prophet['ds'].dt.year < 2016]
    test = df_prophet[df_prophet['ds'].dt.year == 2016]
    
    # Configurar o modelo Prophet
    model = Prophet()
    for var in exog_vars:
        model.add_regressor(var)
    
    # Treinar o modelo
    model.fit(train)
    
    # Fazer previsões
    future = model.make_future_dataframe(periods=12, freq='M')
    for var in exog_vars:
        future[var] = 0  # Inicialmente zero
    # Ajustar eventos futuros (por exemplo, Natal em dezembro)
    future['event_Natal'] = 0
    future.loc[future['ds'].dt.month == 12, 'event_Natal'] = 1
    
    forecast = model.predict(future)
    
    # Plotar os resultados
    model.plot(forecast)
    plt.title(f'Previsão de Vendas para {item_name} usando Prophet')
    plt.show()
    
    # Plotar componentes
    model.plot_components(forecast)
    plt.show()
    
    return forecast[['ds', 'yhat']]

# Aplicar a função para um item
print(f'\nPrevisão com Prophet para o item: {item_to_view}')
item_data = monthly_sales[monthly_sales['item'] == item_to_view]
forecast_prophet = train_and_forecast_prophet(item_data, item_to_view)

# %% [markdown]
# ## Conclusão

# %%
print("A previsão foi concluída com sucesso. Os resultados foram plotados e armazenados no dicionário 'future_forecasts'.")
