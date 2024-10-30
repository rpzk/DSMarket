# forecasting/feature_engineering.py

import pandas as pd
import numpy as np
import logging

def add_event_features(df):
    df['is_christmas'] = df.apply(lambda x: 1 if x['date'].month == 12 and x['date'].day in [23, 24, 26] else 0, axis=1)
    df['is_holiday'] = df.apply(lambda x: 1 if x['date'].month == 12 and x['date'].day == 25 else 0, axis=1)
    # Variáveis temporais adicionais
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def prepare_sales_data(raw_data: pd.DataFrame, apply_log=False, aggregate_weekly=False) -> pd.DataFrame:
    raw_data = raw_data.copy()
    raw_data['date'] = pd.to_datetime(raw_data['date']).dt.tz_localize(None)
    raw_data['sales'] = raw_data['sales'].fillna(0)

    # Adicionar variáveis de evento e temporais
    raw_data = add_event_features(raw_data)

    if aggregate_weekly:
        sales_data = raw_data.groupby('date').agg({
            'sales': 'sum',
            'is_christmas': 'first',
            'is_holiday': 'first',
            'day_of_week': 'first',
            'is_weekend': 'first'
        }).resample('W').sum().sort_index()
    else:
        sales_data = raw_data.groupby('date').agg({
            'sales': 'sum',
            'is_christmas': 'first',
            'is_holiday': 'first',
            'day_of_week': 'first',
            'is_weekend': 'first'
        }).sort_index()

    sales_data = sales_data.asfreq('D').fillna(0)

    if apply_log:
        sales_data['sales'] = np.log1p(sales_data['sales'])

    return sales_data

def filter_zero_sales(data: pd.DataFrame, threshold=0.9) -> bool:
    zero_sales_ratio = (data['sales'] == 0).mean()
    if zero_sales_ratio > threshold:
        logging.warning(f"Item com {zero_sales_ratio*100:.2f}% de vendas zero. Considerar remoção ou tratamento especial.")
        return False
    return True

def detect_and_remove_outliers(data: pd.DataFrame, column='sales', z_thresh=3):
    data = data.copy()
    data['z_score'] = (data[column] - data[column].mean()) / data[column].std()
    cleaned_data = data[data['z_score'].abs() <= z_thresh]
    cleaned_data = cleaned_data.drop(columns=['z_score'])
    return cleaned_data
