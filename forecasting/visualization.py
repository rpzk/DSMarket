# forecasting/visualization.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(data: pd.DataFrame, save_path: str = None):
    """
    Gera e salva um gráfico de previsão com vendas reais e previstas.

    Args:
        data (pd.DataFrame): DataFrame contendo 'date', 'forecast_sales', e opcionalmente 'actual_sales', 'store', 'item'.
        save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico não é salvo.
    """
    if data.empty:
        logging.warning("Nenhum dado disponível para exibir.")
        return
    
    # Verificar se 'date' está no formato datetime
    if not np.issubdtype(data['date'].dtype, np.datetime64):
        data['date'] = pd.to_datetime(data['date'])
    
    plt.figure(figsize=(14, 7))

    # Plota vendas reais, se disponíveis
    if 'actual_sales' in data.columns:
        plt.plot(data['date'], data['actual_sales'], label='Vendas Reais', color='blue')

    # Plota previsões de vendas
    plt.plot(data['date'], data['forecast_sales'], label='Previsão', color='orange')

    # Título e labels
    title = "Previsão de Vendas"
    if 'store' in data.columns and 'item' in data.columns:
        title += f" (Loja: {data['store'].iloc[0]}, Item: {data['item'].iloc[0]})"
    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Vendas")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Salvar ou mostrar o gráfico
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()

    plt.close()  # Fechar o gráfico para liberar memória
