from flask import Blueprint, jsonify, request
import pandas as pd

forecast_bp = Blueprint('forecast_bp', __name__)

# Carregar previsões do arquivo CSV
forecast_data = pd.read_csv("metrics/model_metrics.csv")

@forecast_bp.route('/', methods=['GET'])
def home():
    return "Bem-vindo à API de Previsão DSMarket. Use /api/forecast para acessar as previsões."

@forecast_bp.route('/forecast', methods=['GET'])
def get_forecast():
    store = request.args.get('store')
    item = request.args.get('item')
    
    filtered_data = forecast_data
    if store:
        filtered_data = filtered_data[filtered_data['store'] == store]
    if item:
        filtered_data = filtered_data[filtered_data['item'] == item]
    
    result = filtered_data.to_dict(orient="records")
    return jsonify(result)
