# dags/bike_traffic_orchestrator_dag.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DEFAULT_CONFIG_PATH = "/opt/airflow/config/bike_dag_config.json"


def _load_cfg_and_default_counter() -> Tuple[Dict, str]:
    cfg_json = Variable.get("bike_dag_config", default_var="")
    default_counter = Variable.get(
        "bike_counter_id", default_var="Sebastopol_N-S_mlops"
    )
    if cfg_json:
        return json.loads(cfg_json), default_counter
    path = Variable.get(
        "bike_dag_config_path", default_var=DEFAULT_CONFIG_PATH
    )
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")), default_counter
    return {"counters": {}}, default_counter


def list_counters_payload() -> List[Dict[str, str]]:
    cfg, default_counter = _load_cfg_and_default_counter()
    counters = list((cfg.get("counters") or {}).keys())
    if not counters:
        counters = [default_counter]
    return [{"counter_id": cid} for cid in counters]


with DAG(
    dag_id="bike_traffic_orchestrator",
    start_date=datetime(2025, 9, 20),
    schedule="@daily",     # parent planifie
    catchup=False,
    max_active_runs=1,
    tags=["ops", "orchestration", "bike"],
) as dag:
    get_counters = PythonOperator(
        task_id="get_counters",
        python_callable=list_counters_payload,
    )

    run_init = TriggerDagRunOperator.partial(
        task_id="run_init",
        trigger_dag_id="bike_traffic_init",
        wait_for_completion=True,
        pool="sequential_counters",
    ).expand(conf=get_counters.output)

    run_daily = TriggerDagRunOperator.partial(
        task_id="run_daily",
        trigger_dag_id="bike_traffic_daily",
        wait_for_completion=True,
        pool="sequential_counters",
    ).expand(conf=get_counters.output)

    get_counters >> run_init >> run_daily  # type: ignore
