# visualization.py

import plotly.graph_objects as go

def plot_forecast_interactive(train_data, test_data, forecast, store, item):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data['sales'], mode='lines', name='Treinamento'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['sales'], mode='lines', name='Vendas Reais'))
    fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['forecast_sales'], mode='lines', name='Previsão'))
    fig.update_layout(title=f'Previsão vs. Vendas Reais - Loja {store} Item {item}',
                      xaxis_title='Data',
                      yaxis_title='Vendas')
    fig.show()
