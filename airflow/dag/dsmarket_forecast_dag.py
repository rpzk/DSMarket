from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'DSMarket',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'forecasting_dag',
    default_args=default_args,
    description='Executa previsões para DSMarket',
    schedule_interval=timedelta(days=1),  # Executa diariamente
    start_date=datetime(2024, 10, 28),
    catchup=False,
)

run_forecast = BashOperator(
    task_id='run_forecast',
    bash_command='python /path/to/DSMarket/forecasting/main.py',  # Atualize o caminho para o seu `main.py`
    dag=dag,
)
