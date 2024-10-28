import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(data, save_path="forecast_plot.png"):
    """Gera e salva um gráfico de previsão com vendas reais e previstas."""
    if data.empty:
        print("Nenhum dado disponível para exibir.")
        return
    
    # Convertendo a coluna 'date' para datetime, se necessário
    data['date'] = pd.to_datetime(data['date'])

    plt.figure(figsize=(14, 7))

    # Verifica e plota vendas reais, se disponíveis
    if 'actual_sales' in data.columns:
        plt.plot(data['date'], data['actual_sales'], label='Vendas Reais', color='blue')

    # Plota previsões de vendas
    if 'forecast_sales' in data.columns:
        plt.plot(data['date'], data['forecast_sales'], label='Previsão', color='orange')

    plt.title(f"Previsão de Vendas (Store: {data['store'].iloc[0]}, Item: {data['item'].iloc[0]})")
    plt.xlabel("Data")
    plt.ylabel("Vendas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)  # Salva o gráfico como imagem
    plt.show()
    print(f"Gráfico salvo em: {save_path}")
