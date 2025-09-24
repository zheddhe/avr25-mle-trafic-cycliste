# src/airflow/dags/bike_traffic_orchestrator_dag.py
from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from common.utils import _list_counters_payload


DEFAULT_CONFIG_PATH = "/opt/airflow/config/bike_dag_config.json"


# --------------------------------------------------------------------------- #
# Orchestrator DAG
# --------------------------------------------------------------------------- #
with DAG(
    dag_id="bike_traffic_orchestrator",
    description="Parent DAG orchestrating init and daily DAGs for all counters.",
    start_date=datetime(2025, 9, 20),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["ops", "orchestration", "bike"],
) as dag:
    # Get list of counters from shared config (utils)
    get_counters = PythonOperator(
        task_id="get_counters",
        python_callable=_list_counters_payload,
    )

    # Trigger init DAG for each counter
    run_init = TriggerDagRunOperator.partial(
        task_id="run_init",
        trigger_dag_id="bike_traffic_init",
        poke_interval=5,  # 5s poke interval
        wait_for_completion=True,
        pool="sequential_counters",
    ).expand(conf=get_counters.output)

    # Trigger daily DAG for each counter
    run_daily = TriggerDagRunOperator.partial(
        task_id="run_daily",
        trigger_dag_id="bike_traffic_daily",
        poke_interval=5,  # 5s poke interval
        wait_for_completion=True,
        pool="sequential_counters",
    ).expand(conf=get_counters.output)

    # Orchestration order: counters → init → daily
    get_counters >> run_init >> run_daily  # type: ignore
