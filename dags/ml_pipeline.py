import pendulum
from datetime import timedelta

# Import des classes et opérateurs nécessaires d'Airflow
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# ========================
# Paramètres du DAG
# ========================
default_args = {
    'owner': 'Elias',
    'depends_on_past': False,
    'start_date': pendulum.datetime(2025, 9, 1, tz="UTC"),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ========================
# Définition du DAG
# ========================
with DAG(
    dag_id='mlops_traffic_cycliste',
    default_args=default_args,
    description='Pipeline d\'entraînement de modèle de trafic cycliste',
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'cyclisme', 'pipeline'],
) as dag:

    # ========================
    # Définition des Tâches
    # ========================

    # Tâche 1: Démarrer le pipeline
    start_pipeline = DummyOperator(
        task_id='start_pipeline',
    )

    # Tâche 2: Exécuter le service de préparation des données (ml_data_dev)
    run_data_preparation = DockerOperator(
        task_id='run_data_preparation',
        image='avr25-mle-trafic-cycliste-data:dev',
        api_version='auto',
        auto_remove=True,
        docker_conn_id='docker_default',
        command='--raw-path /app/data/raw/comptage_cycliste_rue_de_rivoli_2023.csv '
                '--site Rivoli --orientation Ouest-Est --range-start 2023-01-01 '
                '--range-end 2023-01-31 --timestamp-col "date_et_heure_de_comptage" '
                '--sub-dir 2023-01-rivoli-west --interim-name rivoli-west-2023-01',
        mount_tmp_dir=False,
        mounts=[
            Mount(source='/Users/elias/Downloads/Projet/avr25-mle-trafic-cycliste/data', target='/app/data', type='bind'),
            Mount(source='/Users/elias/Downloads/Projet/avr25-mle-trafic-cycliste/logs', target='/app/logs', type='bind')
        ]
    )

    # Tâche 3: Exécuter le service de feature engineering (ml_features_dev)
    run_feature_engineering = DockerOperator(
        task_id='run_feature_engineering',
        image='avr25-mle-trafic-cycliste-features:dev',
        api_version='auto',
        auto_remove=True,
        docker_conn_id='docker_default',
        command='--interim-path /app/data/interim/2023-01-rivoli-west/rivoli-west-2023-01.parquet '
                '--sub-dir 2023-01-rivoli-west --processed-name rivoli-west-2023-01-processed '
                '--timestamp-col "date_et_heure_de_comptage"',
        mount_tmp_dir=False,
        mounts=[
            Mount(source='/Users/elias/Downloads/Projet/avr25-mle-trafic-cycliste/data', target='/app/data', type='bind'),
            Mount(source='/Users/elias/Downloads/Projet/avr25-mle-trafic-cycliste/logs', target='/app/logs', type='bind')
        ]
    )

    # Tâche 4: Exécuter l'entraînement du modèle (ml_models_dev)
    run_model_training = DockerOperator(
        task_id='run_model_training',
        image='avr25-mle-trafic-cycliste-models:dev',
        api_version='auto',
        auto_remove=True,
        docker_conn_id='docker_default',
        command='--processed-path /app/data/processed/2023-01-rivoli-west/rivoli-west-2023-01-processed.parquet '
                '--sub-dir 2023-01-rivoli-west --target-col "comptage_horaire" '
                '--ts-col-utc "date_et_heure_de_comptage_utc" '
                '--ts-col-local "date_et_heure_de_comptage_local" --ar 2 --mm 2 --roll 3 '
                '--test-ratio 0.2 --grid-iter 10',
        mount_tmp_dir=False,
        mounts=[
            Mount(source='/Users/elias/Downloads/Projet/avr25-mle-trafic-cycliste/models', target='/app/models', type='bind'),
            Mount(source='/Users/elias/Downloads/Projet/avr25-mle-trafic-cycliste/data', target='/app/data', type='bind'),
            Mount(source='/Users/elias/Downloads/Projet/avr25-mle-trafic-cycliste/logs', target='/app/logs', type='bind')
        ]
    )

    # Tâche 5: Fin du pipeline
    end_pipeline = DummyOperator(
        task_id='end_pipeline',
    )

    # ========================
    # Définition de l'ordre d'exécution
    # ========================
    start_pipeline >> run_data_preparation >> run_feature_engineering >> run_model_training >> end_pipeline