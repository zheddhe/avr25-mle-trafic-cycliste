"""Development orchestrator DAG for bike traffic pipelines."""

from __future__ import annotations

import logging
from datetime import datetime

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import DAG

from common.utils import _list_counters_payload, _load_concurrency_config

logger = logging.getLogger("airflow.task")
logger.info("[orchestrator] Development DAG loaded")

concurrency = _load_concurrency_config()


with DAG(
    dag_id="bike_traffic_orchestrator",
    description="Parent DAG orchestrating dev init and daily DAGs for counters.",
    start_date=datetime(2025, 9, 1),
    schedule=None,
    catchup=False,
    max_active_runs=concurrency.orchestrator_max_active_runs,
    max_active_tasks=concurrency.orchestrator_max_active_tasks,
    tags=["ops", "orchestration", "bike", "dev"],
) as dag:
    counters = _list_counters_payload()

    run_init = TriggerDagRunOperator.partial(
        task_id="run_init",
        trigger_dag_id="bike_traffic_init",
        wait_for_completion=True,
        reset_dag_run=True,
    ).expand(conf=counters)

    run_daily = TriggerDagRunOperator.partial(
        task_id="run_daily",
        trigger_dag_id="bike_traffic_daily",
        wait_for_completion=True,
        reset_dag_run=True,
    ).expand(conf=counters)

    run_init >> run_daily  # type: ignore
