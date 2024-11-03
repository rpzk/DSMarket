# forecasting/model_training.py

from prophet import Prophet
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import logging
import warnings
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

warnings.filterwarnings("ignore")

# Configuração de logging
logging.basicConfig(level=logging.INFO)

# Função Objetivo para Optuna (Prophet)
def prophet_objective(trial, train_df, test_df, regressors):
    changepoint_prior_scale = trial.suggest_loguniform('changepoint_prior_scale', 0.001, 1.0)
    seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
    yearly_seasonality = trial.suggest_categorical('yearly_seasonality', [True, False])
    weekly_seasonality = trial.suggest_categorical('weekly_seasonality', [True, False])
    daily_seasonality = trial.suggest_categorical('daily_seasonality', [True, False])

    try:
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            growth='logistic'
        )

        # Adicionar regressores
        for reg in regressors:
            model.add_regressor(reg)

        # Definir limites
        train_df['cap'] = train_df['y'].max()
        train_df['floor'] = 0

        model.fit(train_df)

        # Preparar o DataFrame futuro
        future = model.make_future_dataframe(periods=len(test_df))
        future = future.merge(test_df[['ds'] + regressors], on='ds', how='left')
        future[regressors] = future[regressors].fillna(0)
        future['cap'] = train_df['cap'].iloc[0]
        future['floor'] = train_df['floor'].iloc[0]

        forecast = model.predict(future)
        forecast_values = forecast['yhat'][-len(test_df):].values
        mape = mean_absolute_percentage_error(test_df['y'], forecast_values) * 100
        return mape
    except Exception as e:
        logging.error(f"Erro na otimização do Prophet: {e}")
        return float('inf')  # Penalização se ocorrer erro

# Função para otimizar Prophet usando Optuna
def optimize_prophet(train_df, test_df, regressors, n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: prophet_objective(trial, train_df, test_df, regressors), n_trials=n_trials)
    return study.best_params, study.best_value

# Função para ajustar e prever com Prophet (com otimização)
def fit_and_forecast_prophet_optimized(train_df, test_df, steps, regressors, n_trials=50):
    try:
        best_params, best_mape = optimize_prophet(train_df, test_df, regressors, n_trials)
        logging.info(f"Melhores parâmetros Prophet: {best_params} com MAPE: {best_mape}")

        model = Prophet(
            changepoint_prior_scale=best_params['changepoint_prior_scale'],
            seasonality_mode=best_params['seasonality_mode'],
            yearly_seasonality=best_params['yearly_seasonality'],
            weekly_seasonality=best_params['weekly_seasonality'],
            daily_seasonality=best_params['daily_seasonality'],
            growth='logistic'
        )

        # Adicionar regressores
        for reg in regressors:
            model.add_regressor(reg)

        # Definir limites
        train_df['cap'] = train_df['y'].max()
        train_df['floor'] = 0

        model.fit(train_df)

        # Preparar o DataFrame futuro
        future = model.make_future_dataframe(periods=steps)
        future = future.merge(test_df[['ds'] + regressors], on='ds', how='left')
        future[regressors] = future[regressors].fillna(0)
        future['cap'] = train_df['cap'].iloc[0]
        future['floor'] = train_df['floor'].iloc[0]

        # Faça a previsão
        forecast = model.predict(future)
        forecast_values = forecast['yhat'][-steps:].values
        model_type = "prophet"
        return forecast_values, model_type, best_params, model
    except Exception as e:
        logging.error(f"Erro ao usar Prophet otimizado: {e}")
        return None, None, None, None

# Função para ajustar e prever com SARIMA otimizado
def fit_and_forecast_sarima_optimized(train_data, test_data, steps):
    from pmdarima import auto_arima

    exog_train = train_data[['is_christmas', 'is_holiday', 'day_of_week', 'is_weekend']]
    exog_test = test_data[['is_christmas', 'is_holiday', 'day_of_week', 'is_weekend']]

    try:
        sarima_model = auto_arima(
            train_data['sales'],
            exogenous=exog_train,
            seasonal=True,
            m=7,  # Ajuste conforme a sazonalidade dos dados (7 para semanal)
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        sarima_model.fit(train_data['sales'], exogenous=exog_train)
        forecast_values = sarima_model.predict(n_periods=steps, exogenous=exog_test)
        forecast_values = [max(0, val) for val in forecast_values]  # Corrige valores negativos para zero
        model_type = "sarima"
        params = {
            "order": sarima_model.order,
            "seasonal_order": sarima_model.seasonal_order
        }
        return forecast_values, model_type, params, sarima_model
    except Exception as e:
        logging.error(f"Erro ao usar SARIMA otimizado: {e}")
        return None, None, None, None

# Função para ajustar e prever com XGBoost
def fit_and_forecast_xgboost(train_df, test_df, steps, regressors):
    try:
        X_train = train_df[regressors]
        y_train = train_df['y']

        # Treinamento do modelo XGBoost
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)

        # Previsão nos dados de teste
        X_test = test_df[regressors]
        forecast_values = model.predict(X_test)[:steps]
        forecast_values = [max(0, val) for val in forecast_values]  # Corrige valores negativos para zero

        model_type = "xgboost"
        params = {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth
        }
        return forecast_values, model_type, params, model
    except Exception as e:
        logging.error(f"Erro ao usar XGBoost: {e}")
        return None, None, None, None

# Função para salvar o modelo e a última data
def save_model_and_date(model, model_type, store, item, models_dir, last_date):
    model_filename = f"{model_type}_model_{store.replace(' ', '_')}_{item}.pkl"
    model_path = os.path.join(models_dir, model_type, model_filename)
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Salvar o modelo
    if model_type == "sarima":
        model.save(model_path)
    else:
        joblib.dump(model, model_path)
    
    # Salvar a última data
    last_date_path = model_path.replace('.pkl', '_last_date.txt')
    with open(last_date_path, 'w') as f:
        f.write(last_date.strftime('%Y-%m-%d'))
    
    logging.info(f"Modelo salvo: {model_path}")
    logging.info(f"Última data salva: {last_date_path}")

# Exemplo de Treinamento e Salvamento
def train_and_save_models(data, models_dir):
    # Supondo que 'data' seja um DataFrame com colunas: ['store', 'item', 'date', 'sales', 'is_christmas', 'is_holiday', 'day_of_week', 'is_weekend']
    stores = data['store'].unique()
    items = data['item'].unique()
    regressors = ['is_christmas', 'is_holiday', 'day_of_week', 'is_weekend']

    for store in stores:
        for item in items:
            subset = data[(data['store'] == store) & (data['item'] == item)].copy()
            if subset.empty:
                continue

            # Preparar dados para Prophet
            prophet_df = subset.rename(columns={'date': 'ds', 'sales': 'y'})
            prophet_df['cap'] = prophet_df['y'].max()
            prophet_df['floor'] = 0

            # Dividir em treino e teste
            train_size = int(len(prophet_df) * 0.8)
            train_prophet = prophet_df.iloc[:train_size]
            test_prophet = prophet_df.iloc[train_size:]

            # Treinar e salvar Prophet
            forecast_values, model_type, params, model = fit_and_forecast_prophet_optimized(
                train_prophet, test_prophet, steps=28, regressors=regressors, n_trials=10
            )
            if model:
                last_date = subset['date'].max()
                save_model_and_date(model, model_type, store, item, models_dir, last_date)

            # Preparar dados para SARIMA
            sarima_df = subset.rename(columns={'date': 'date', 'sales': 'sales'})
            sarima_df.set_index('date', inplace=True)

            # Dividir em treino e teste
            train_size = int(len(sarima_df) * 0.8)
            train_sarima = sarima_df.iloc[:train_size]
            test_sarima = sarima_df.iloc[train_size:]

            # Treinar e salvar SARIMA
            forecast_values, model_type, params, model = fit_and_forecast_sarima_optimized(
                train_sarima, test_sarima, steps=28
            )
            if model:
                last_date = subset['date'].max()
                save_model_and_date(model, model_type, store, item, models_dir, last_date)

            # Preparar dados para XGBoost
            xgb_df = subset.copy()
            xgb_df.rename(columns={'sales': 'y'}, inplace=True)

            # Dividir em treino e teste
            train_size = int(len(xgb_df) * 0.8)
            train_xgb = xgb_df.iloc[:train_size]
            test_xgb = xgb_df.iloc[train_size:]

            # Treinar e salvar XGBoost
            forecast_values, model_type, params, model = fit_and_forecast_xgboost(
                train_xgb, test_xgb, steps=28, regressors=regressors
            )
            if model:
                last_date = subset['date'].max()
                save_model_and_date(model, model_type, store, item, models_dir, last_date)

# Exemplo de uso
if __name__ == "__main__":
    # Carregar seus dados aqui
    data = pd.read_csv('path_to_your_data.csv')  # Substitua pelo caminho real dos seus dados
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    train_and_save_models(data, models_dir)
