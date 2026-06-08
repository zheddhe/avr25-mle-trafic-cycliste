"""Production-like Airflow DAG using job-runner-api typed ML steps."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from airflow.sdk import DAG
except ImportError:  # pragma: no cover - Airflow 2 compatibility fallback.
    from airflow import DAG

from airflow.operators.python import PythonOperator

CONFIG_PATH = Path("/opt/airflow/config/bike_dag_config.json")
JOB_RUNNER_URL = "http://job-runner-api:10080"
API_URL = "http://api-dev:10000"
MANIFEST_ROOT = "/app/artifacts/manifests"
DAG_ID = "prod_ml_step_pipeline"


def _read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {error.code} while calling {url}: {body}"
        ) from error


def _submit_runner_job(payload: dict[str, Any]) -> dict[str, Any]:
    status = _post_json(f"{JOB_RUNNER_URL}/jobs", payload)
    state = status.get("state")
    if state != "succeeded":
        error = status.get("error") or {}
        message = error.get("message") or "Runner job did not succeed."
        raise RuntimeError(f"Runner job failed with state={state}: {message}")

    return status


def _refresh_api() -> dict[str, Any]:
    return _post_json(f"{API_URL}/admin/refresh", {})


def _counter_paths(
    counter_id: str,
    counter_config: dict[str, Any],
) -> dict[str, str]:
    sub_dir = counter_id
    interim_name = counter_config["interim_name"]
    processed_name = counter_config["processed_name"]
    final_basename = counter_config["final_basename"]

    return {
        "raw": f"/app/data/raw/{counter_config['raw_file_name']}",
        "interim": f"/app/data/interim/{sub_dir}/{interim_name}",
        "processed": f"/app/data/processed/{sub_dir}/{processed_name}",
        "prediction": f"/app/data/final/{sub_dir}/{final_basename}",
        "model": f"/app/models/{sub_dir}",
        "sub_dir": sub_dir,
    }


def _build_ingest_payload(
    counter_id: str,
    counter_config: dict[str, Any],
) -> dict[str, Any]:
    paths = _counter_paths(counter_id, counter_config)
    return {
        "job_type": "ingest",
        "run_id": "{{ run_id }}-ingest-" + counter_id,
        "counter_id": counter_id,
        "dag_id": DAG_ID,
        "task_id": "ingest_" + counter_id,
        "manifest_root": MANIFEST_ROOT,
        "raw_path": paths["raw"],
        "site": counter_config["site"],
        "orientation": counter_config["orientation"],
        "range_start": 0.0,
        "range_end": 100.0,
        "timestamp_col": "date_et_heure_de_comptage",
        "sub_dir": paths["sub_dir"],
        "interim_name": counter_config["interim_name"],
        "interim_output_path": paths["interim"],
    }


def _build_features_payload(
    counter_id: str,
    counter_config: dict[str, Any],
) -> dict[str, Any]:
    paths = _counter_paths(counter_id, counter_config)
    return {
        "job_type": "features",
        "run_id": "{{ run_id }}-features-" + counter_id,
        "counter_id": counter_id,
        "dag_id": DAG_ID,
        "task_id": "features_" + counter_id,
        "manifest_root": MANIFEST_ROOT,
        "interim_input_path": paths["interim"],
        "processed_output_path": paths["processed"],
        "processed_name": counter_config["processed_name"],
        "timestamp_col": "date_et_heure_de_comptage",
    }


def _build_models_payload(
    counter_id: str,
    counter_config: dict[str, Any],
) -> dict[str, Any]:
    paths = _counter_paths(counter_id, counter_config)
    modeling = counter_config["modeling"]
    return {
        "job_type": "models",
        "run_id": "{{ run_id }}-models-" + counter_id,
        "counter_id": counter_id,
        "dag_id": DAG_ID,
        "task_id": "models_" + counter_id,
        "manifest_root": MANIFEST_ROOT,
        "processed_input_path": paths["processed"],
        "prediction_output_path": paths["prediction"],
        "model_output_path": paths["model"],
        "target_col": "comptage_horaire",
        "ts_col_utc": "date_et_heure_de_comptage_utc",
        "ts_col_local": "date_et_heure_de_comptage_local",
        "ar": modeling["ar"],
        "mm": modeling["mm"],
        "roll": modeling["roll"],
        "test_ratio": modeling["test_ratio"],
        "grid_iter": modeling["grid_iter"],
        "mlflow_uri": "http://mlflow-server:5000",
    }


def _create_counter_tasks(
    dag: DAG,
    counter_id: str,
    counter_config: dict[str, Any],
) -> PythonOperator:
    ingest = PythonOperator(
        task_id="ingest_" + counter_id,
        python_callable=_submit_runner_job,
        op_kwargs={"payload": _build_ingest_payload(counter_id, counter_config)},
        dag=dag,
    )
    features = PythonOperator(
        task_id="features_" + counter_id,
        python_callable=_submit_runner_job,
        op_kwargs={"payload": _build_features_payload(counter_id, counter_config)},
        dag=dag,
    )
    models = PythonOperator(
        task_id="models_" + counter_id,
        python_callable=_submit_runner_job,
        op_kwargs={"payload": _build_models_payload(counter_id, counter_config)},
        dag=dag,
    )
    ingest >> features >> models
    return models


config = _read_config()
with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 0, "retry_delay": timedelta(minutes=1)},
    tags=["prod", "ml", "job-runner"],
) as dag:
    terminal_tasks = [
        _create_counter_tasks(dag, counter_id, counter_config)
        for counter_id, counter_config in config["counters"].items()
    ]
    refresh_api = PythonOperator(
        task_id="refresh_api_after_models",
        python_callable=_refresh_api,
    )
    terminal_tasks >> refresh_api
