# forecasting/previsão/models.py

from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd

class ProphetModel:
    def __init__(self, **kwargs):
        self.model = Prophet(**kwargs)
    
    def fit(self, df: pd.DataFrame):
        self.model.fit(df)
    
    def predict(self, periods: int):
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        forecast = self.model.predict(future)
        return forecast['yhat'][-periods:].values

class ARIMAModel:
    def __init__(self, **kwargs):
        self.model = pm.auto_arima(**kwargs)
    
    def fit(self, series: pd.Series):
        self.model.fit(series)
    
    def predict(self, periods: int):
        return self.model.predict(n_periods=periods)

def calculate_metrics(actual, forecast):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    mask = actual != 0
    if not np.any(mask):
        mape = np.nan
    else:
        mape = mean_absolute_percentage_error(actual[mask], forecast[mask]) * 100
    return rmse, mae, mape
