# forecasting/visualization.py

import matplotlib.pyplot as plt

def plot_forecast(data, save_path=None):
    """Gera e salva um gráfico de previsão com vendas reais e previstas."""
    if data.empty:
        print("Nenhum dado disponível para exibir.")
        return
    
    # Configurar o gráfico
    plt.figure(figsize=(14, 7))
    plt.plot(data['date'], data['forecast_sales'], label='Previsão', color='orange')
    if 'actual_sales' in data.columns:
        plt.plot(data['date'], data['actual_sales'], label='Vendas Reais', color='blue')

    plt.xlabel("Data")
    plt.ylabel("Vendas")
    plt.title(f"Previsão de Vendas para {data['store'].iloc[0]} - {data['item'].iloc[0]}")
    plt.legend()
    plt.grid(True)

    # Salvar o gráfico se `save_path` for especificado
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em: {save_path}")
    
    plt.close()  # Fechar o gráfico para evitar exibição
