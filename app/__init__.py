from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Importar e registrar as rotas
    from .routes import forecast_bp
    app.register_blueprint(forecast_bp, url_prefix='/api')

    return app
