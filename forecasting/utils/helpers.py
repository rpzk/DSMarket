# forecasting/utils/helpers.py

import pandas as pd

def get_top_items(store_data: pd.DataFrame, top_n: int) -> pd.Series:
    """
    Obtém os top N itens com base nas vendas totais em uma loja.

    Args:
        store_data (pd.DataFrame): Dados de vendas da loja.
        top_n (int): Número de itens a selecionar.

    Returns:
        pd.Series: Série contendo os IDs dos top N itens.
    """
    total_sales_per_item = store_data.groupby('item')['sales'].sum().reset_index()
    top_items = total_sales_per_item.sort_values(by='sales', ascending=False).head(top_n)['item']
    return top_items
