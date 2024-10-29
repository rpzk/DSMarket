# forecasting/models.py

from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd

class ProphetModel:
    """
    Classe para encapsular o modelo Prophet.
    """
    def __init__(self, **kwargs):
        self.model = Prophet(**kwargs)
    
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
        return forecast['yhat'][-periods:].values

class ARIMAModel:
    """
    Classe para encapsular o modelo ARIMA.
    """
    def __init__(self, **kwargs):
        self.model = None
        self.model_params = kwargs
    
    def fit(self, series: pd.Series):
        """
        Ajusta o modelo ARIMA aos dados fornecidos.

        Args:
            series (pd.Series): Série temporal dos dados.
        """
        self.model = pm.auto_arima(series, **self.model_params)
    
    def predict(self, periods: int) -> np.ndarray:
        """
        Gera previsões para um número específico de períodos.

        Args:
            periods (int): Número de períodos futuros para prever.

        Returns:
            np.ndarray: Valores previstos.
        """
        return self.model.predict(n_periods=periods)

def calculate_metrics(actual: np.ndarray, forecast: np.ndarray) -> tuple:
    """
    Calcula métricas de erro entre valores reais e previstos.

    Args:
        actual (np.ndarray): Valores reais.
        forecast (np.ndarray): Valores previstos.

    Returns:
        tuple: RMSE, MAE e MAPE.
    """
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    mask = actual != 0
    if not np.any(mask):
        mape = np.nan
    else:
        mape = mean_absolute_percentage_error(actual[mask], forecast[mask]) * 100
    return rmse, mae, mape
