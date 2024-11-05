# Usando a imagem oficial do Python
FROM python:3.10-slim

# Definindo o diretório de trabalho
WORKDIR /app

# Copiando os arquivos de requisitos e instalando as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiando o código da aplicação
COPY ./app ./app

# Expondo a porta 8000
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
