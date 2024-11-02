from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd

class ProphetModel:
    """
    Classe para encapsular o modelo Prophet com opções adicionais de sazonalidade e pontos de mudança.
    """
    def __init__(self, changepoint_prior_scale=0.1, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, custom_seasonalities=None):
        # Configuração do Prophet com sazonalidades ajustáveis e escalas de pontos de mudança
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality
        )
        
        # Adiciona sazonalidades customizadas, se fornecidas
        if custom_seasonalities:
            for season in custom_seasonalities:
                self.model.add_seasonality(name=season['name'], period=season['period'], fourier_order=season['fourier_order'])
    
    def fit(self, df: pd.DataFrame):
        """
        Ajusta o modelo Prophet aos dados fornecidos.
        
        Args:
            df (pd.DataFrame): DataFrame com colunas 'ds' e 'y'.
        """
        self.model.fit(df)
    
    def predict(self, periods: int) -> np.ndarray:
        """
        Gera previsões para um número específico de períodos.
        
        Args:
            periods (int): Número de períodos futuros para prever.
        
        Returns:
            np.ndarray: Valores previstos.
        """
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-periods:]
    
class ARIMAModel:
    """
    Classe para encapsular o modelo ARIMA com opções adicionais de ajuste de parâmetros.
    """
    def __init__(self, seasonal=True, **kwargs):
        self.model = None
        self.model_params = kwargs
        self.seasonal = seasonal
    
    def fit(self, series: pd.Series):
        """
        Ajusta o modelo ARIMA aos dados fornecidos.
        
        Args:
            series (pd.Series): Série temporal dos dados.
        """
        self.model = pm.auto_arima(series, seasonal=self.seasonal, **self.model_params)
    
    def predict(self, periods: int) -> np.ndarray:
        """
        Gera previsões para um número específico de períodos.
        
        Args:
            periods (int): Número de períodos futuros para prever.
        
        Returns:
            np.ndarray: Valores previstos.
        """
        forecast = self.model.predict(n_periods=periods)
        return forecast

# Funções auxiliares para calcular métricas
def calculate_metrics(true_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    mape = mean_absolute_percentage_error(true_values, predicted_values) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
