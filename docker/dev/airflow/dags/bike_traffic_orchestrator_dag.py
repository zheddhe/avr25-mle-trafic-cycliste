# src/airflow/dags/bike_traffic_orchestrator_dag.py
from __future__ import annotations

import logging
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from common.utils import _list_counters_payload

logger = logging.getLogger("airflow.task")

# --------------------------------------------------------------------------- #
# Orchestrator DAG
# --------------------------------------------------------------------------- #
with DAG(
    dag_id="bike_traffic_orchestrator",
    description="Parent DAG orchestrating dev init and daily DAGs for counters.",
    start_date=datetime(2025, 9, 20),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["ops", "orchestration", "bike", "dev"],
) as dag:
    logger.info("[orchestrator] DAG loaded")

    get_counters = PythonOperator(
        task_id="get_counters",
        python_callable=_list_counters_payload,
    )

    run_init = TriggerDagRunOperator.partial(
        task_id="run_init",
        trigger_dag_id="bike_traffic_init",
        poke_interval=5,
        wait_for_completion=True,
        pool="sequential_counters",
    ).expand(conf=get_counters.output)

    run_daily = TriggerDagRunOperator.partial(
        task_id="run_daily",
        trigger_dag_id="bike_traffic_daily",
        poke_interval=5,
        wait_for_completion=True,
        pool="sequential_counters",
    ).expand(conf=get_counters.output)

    get_counters >> run_init >> run_daily  # type: ignore
