"""Production-like init and daily DAGs using job-runner-api."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow.exceptions import AirflowException
from airflow.models import TaskInstance
from airflow.models.param import ParamsDict
from airflow.providers.http.operators.http import HttpOperator
from airflow.providers.standard.operators.python import (
    PythonOperator,
    ShortCircuitOperator,
)
from airflow.sdk import DAG, Param, TaskGroup, Variable
from common.utils import (
    CounterCfg,
    _load_concurrency_config,
    _load_config,
    get_airflow_task_logger,
)

LOGGER = get_airflow_task_logger()
concurrency = _load_concurrency_config()

JOB_RUNNER_URL = "http://job-runner-api:10080"
DATA_ROOT = "/app/data"
MODEL_ROOT = "/app/models"
MANIFEST_ROOT = "/app/artifacts/manifests"

DAG_PARAMS: ParamsDict = ParamsDict(
    {
        "counter_id": Param(
            type="string",
            description="Counter to process for this run (REQUIRED).",
        )
    }
)


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
        LOGGER.error(
            "[pipeline] Runner API HTTP error: url=%s code=%s body=%s",
            url,
            error.code,
            body,
        )
        raise AirflowException(
            f"HTTP {error.code} while calling {url}: {body}"
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        LOGGER.error("[pipeline] Runner API call failed: url=%s error=%s", url, error)
        raise AirflowException(f"Failed to call {url}: {error}") from error


def _extract_counter_id(ctx: Mapping[str, Any]) -> str | None:
    params_obj = ctx.get("params")
    if isinstance(params_obj, Mapping):
        value = params_obj.get("counter_id")
        if isinstance(value, str) and value:
            return value
    dag_run = ctx.get("dag_run")
    conf_obj = getattr(dag_run, "conf", None)
    if isinstance(conf_obj, Mapping):
        value = conf_obj.get("counter_id")
        if isinstance(value, str) and value:
            return value
    return None


def _prepare_args_callable(mode: str):
    def _callable(**ctx) -> dict[str, Any]:
        ti: TaskInstance = ctx["ti"]
        cfg, default_counter_id = _load_config()
        counter_id = _extract_counter_id(ctx) or default_counter_id
        counter = cfg.counters[counter_id]
        sched = cfg.scheduling
        exec_date = ctx["data_interval_end"]
        run_id = (
            f"{exec_date.strftime('%Y%m%d')}_"
            f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        )
        delta_days = 0
        window = {"start": 0.0, "end": 75.0}
        proceed = True
        if mode == "daily":
            anchor_dt = datetime.fromisoformat(sched.initial_anchor_date).date()
            delta_days = (exec_date.date() - anchor_dt).days
            shift = max(0.0, delta_days * float(sched.daily_increment_pct))
            window = {
                "start": round(max(0.0, min(100.0, shift)), 2),
                "end": round(max(0.0, min(100.0, 75.0 + shift)), 2),
            }
            proceed = window["end"] > 75.0
        context = _build_context(counter_id, counter, run_id, delta_days, window)
        context["PROCEED"] = proceed
        LOGGER.info(
            "[pipeline] Prepared %s run: counter=%s run_id=%s "
            "window=%s proceed=%s",
            mode,
            counter_id,
            run_id,
            window,
            proceed,
        )
        for key, value in context.items():
            ti.xcom_push(key=key, value=value)
        return context

    return _callable


def _build_context(
    counter_id: str,
    counter: CounterCfg,
    run_id: str,
    delta_days: int,
    window: dict[str, float],
) -> dict[str, Any]:
    sub_dir = f"{counter_id}_day{delta_days}"
    return {
        "RUN_ID": run_id,
        "COUNTER_ID": counter_id,
        "SUB_DIR": sub_dir,
        "WINDOW": window,
        "RAW_FILE_NAME": counter.raw_file_name,
        "SITE": counter.site,
        "SITE_SHORT": counter_id,
        "ORIENTATION": counter.orientation,
        "INTERIM_NAME": counter.interim_name,
        "PROCESSED_NAME": counter.processed_name,
        "FINAL_BASENAME": counter.final_basename,
        "INTERIM_PATH": f"{DATA_ROOT}/interim/{sub_dir}/{counter.interim_name}",
        "PROCESSED_PATH": f"{DATA_ROOT}/processed/{sub_dir}/{counter.processed_name}",
        "PREDICTION_PATH": f"{DATA_ROOT}/final/{sub_dir}/{counter.final_basename}",
        "MODEL_PATH": f"{MODEL_ROOT}/{sub_dir}",
        "AR": counter.modeling.ar,
        "MM": counter.modeling.mm,
        "ROLL": counter.modeling.roll,
        "TEST_RATIO": counter.modeling.test_ratio,
        "GRID_ITER": counter.modeling.grid_iter,
    }


def _submit_runner_job(job_type: str, **ctx) -> dict[str, Any]:
    ti: TaskInstance = ctx["ti"]
    payload = _build_job_payload(job_type, ti, ctx)
    LOGGER.info(
        "[pipeline] Submitting runner job: job_type=%s counter=%s run_id=%s",
        job_type,
        payload["counter_id"],
        payload["run_id"],
    )
    status = _post_json(f"{JOB_RUNNER_URL}/jobs", payload)
    result = status.get("result") or {}
    metrics = result.get("metrics") or {}
    metrics_reference = metrics.get("metrics_reference")
    runner_job_id = status.get("job_id")
    if status.get("state") != "succeeded":
        error = status.get("error") or {}
        message = error.get("message") or "Runner job did not succeed."
        LOGGER.error(
            "[pipeline] Runner job failed: job_id=%s job_type=%s "
            "counter=%s run_id=%s state=%s message=%s",
            runner_job_id,
            job_type,
            payload["counter_id"],
            payload["run_id"],
            status.get("state"),
            message,
        )
        raise AirflowException(
            f"Runner job failed with state={status.get('state')}: {message}"
        )
    LOGGER.info(
        "[pipeline] Runner job succeeded: job_id=%s job_type=%s "
        "counter=%s run_id=%s metrics_reference=%s",
        runner_job_id,
        job_type,
        payload["counter_id"],
        payload["run_id"],
        metrics_reference,
    )
    manifest = result.get("manifest") or {}
    if manifest:
        ti.xcom_push(key=f"{job_type.upper()}_MANIFEST", value=manifest)
    return status


def _build_job_payload(
    job_type: str,
    ti: TaskInstance,
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "job_type": job_type,
        "run_id": _pull(ti, "RUN_ID"),
        "counter_id": _pull(ti, "COUNTER_ID"),
        "dag_id": str(ctx["dag"].dag_id),
        "task_id": str(ctx["task"].task_id),
        "try_number": int(ti.try_number),
        "manifest_root": MANIFEST_ROOT,
    }
    if job_type == "ingest":
        base.update(_ingest_args(ti))
    elif job_type == "features":
        base.update(_features_args(ti))
    elif job_type == "models":
        base.update(_models_args(ti))
    else:
        raise AirflowException(f"Unsupported job_type={job_type}")
    return base


def _ingest_args(ti: TaskInstance) -> dict[str, Any]:
    window = _pull(ti, "WINDOW")
    return {
        "raw_path": f"{DATA_ROOT}/raw/{_pull(ti, 'RAW_FILE_NAME')}",
        "site": _pull(ti, "SITE"),
        "orientation": _pull(ti, "ORIENTATION"),
        "range_start": window["start"],
        "range_end": window["end"],
        "timestamp_col": "date_et_heure_de_comptage",
        "sub_dir": _pull(ti, "SUB_DIR"),
        "interim_name": _pull(ti, "INTERIM_NAME"),
        "interim_output_path": _pull(ti, "INTERIM_PATH"),
    }


def _features_args(ti: TaskInstance) -> dict[str, Any]:
    return {
        "interim_input_path": _manifest_or_xcom_path(
            ti,
            task_id="etl.ingest",
            manifest_key="INGEST_MANIFEST",
            fallback_key="INTERIM_PATH",
        ),
        "processed_output_path": _pull(ti, "PROCESSED_PATH"),
        "processed_name": _pull(ti, "PROCESSED_NAME"),
        "timestamp_col": "date_et_heure_de_comptage",
    }


def _models_args(ti: TaskInstance) -> dict[str, Any]:
    return {
        "processed_input_path": _manifest_or_xcom_path(
            ti,
            task_id="etl.features",
            manifest_key="FEATURES_MANIFEST",
            fallback_key="PROCESSED_PATH",
        ),
        "prediction_output_path": _pull(ti, "PREDICTION_PATH"),
        "model_output_path": _pull(ti, "MODEL_PATH"),
        "target_col": "comptage_horaire",
        "ts_col_utc": "date_et_heure_de_comptage_utc",
        "ts_col_local": "date_et_heure_de_comptage_local",
        "ar": _pull(ti, "AR"),
        "mm": _pull(ti, "MM"),
        "roll": _pull(ti, "ROLL"),
        "test_ratio": _pull(ti, "TEST_RATIO"),
        "grid_iter": _pull(ti, "GRID_ITER"),
        "mlflow_uri": "http://mlflow-server:5000",
    }


def _pull(ti: TaskInstance, key: str) -> Any:
    return ti.xcom_pull(task_ids="etl.prepare_args", key=key)


def _manifest_or_xcom_path(
    ti: TaskInstance,
    *,
    task_id: str,
    manifest_key: str,
    fallback_key: str,
) -> str:
    manifest = ti.xcom_pull(task_ids=task_id, key=manifest_key)
    if isinstance(manifest, Mapping):
        storage = manifest.get("storage")
        if isinstance(storage, Mapping) and storage.get("local_path"):
            return f"/app/{storage['local_path']}"
    return _pull(ti, fallback_key)


def _read_manifests(**ctx) -> dict[str, Any]:
    ti: TaskInstance = ctx["ti"]
    counter_id = _pull(ti, "COUNTER_ID")
    manifests = {
        "INGEST_MANIFEST": _read_current_manifest("interim_dataset", counter_id),
        "FEATURES_MANIFEST": _read_current_manifest("feature_dataset", counter_id),
        "MODELS_MANIFEST": _read_current_manifest("predictions", counter_id),
    }
    for key, value in manifests.items():
        if value is not None:
            ti.xcom_push(key=key, value=value)
    return manifests


def _read_current_manifest(
    artifact_type: str,
    counter_id: str,
) -> dict[str, Any] | None:
    path = Path(MANIFEST_ROOT) / artifact_type / counter_id / "current.json"
    if not path.is_file():
        LOGGER.debug("[pipeline] Manifest not found yet: %s", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _daily_gate_callable(**ctx) -> bool:
    ti: TaskInstance = ctx["ti"]
    return bool(_pull(ti, "PROCEED"))


def _init_gate_callable(**ctx) -> bool:
    counter_id = _extract_counter_id(ctx) or "UNKNOWN"
    already_done = Variable.get(f"bike_init_done__{counter_id}", default="0") == "1"
    LOGGER.info("[pipeline] Init gate counter=%s done=%s", counter_id, already_done)
    return not already_done


def _mark_init_done_callable(**ctx) -> None:
    counter_id = _extract_counter_id(ctx) or "UNKNOWN"
    Variable.set(f"bike_init_done__{counter_id}", "1")
    LOGGER.info("[pipeline] Init marked done for counter=%s", counter_id)


def build_etl_group(dag: DAG, mode: str) -> TaskGroup:
    with TaskGroup(group_id="etl", dag=dag) as etl:
        prepare_args = PythonOperator(
            task_id="prepare_args",
            python_callable=_prepare_args_callable(mode),
            dag=dag,
        )
        upstream = prepare_args
        if mode == "daily":
            gate = ShortCircuitOperator(
                task_id="gate",
                python_callable=_daily_gate_callable,
                dag=dag,
            )
            prepare_args >> gate  # type: ignore
            upstream = gate
        ingest = _runner_task("ingest", dag)
        read_after_ingest = _read_task("read_manifests_after_ingest", dag)
        features = _runner_task("features", dag)
        read_after_features = _read_task("read_manifests_after_features", dag)
        models = _runner_task("models", dag)
        read_manifests = _read_task("read_manifests", dag)
        (
            upstream
            >> ingest
            >> read_after_ingest
            >> features
            >> read_after_features
            >> models
            >> read_manifests
        )  # type: ignore
    return etl


def _runner_task(job_type: str, dag: DAG) -> PythonOperator:
    return PythonOperator(
        task_id=job_type,
        python_callable=_submit_runner_job,
        op_kwargs={"job_type": job_type},
        dag=dag,
    )


def _read_task(task_id: str, dag: DAG) -> PythonOperator:
    return PythonOperator(task_id=task_id, python_callable=_read_manifests, dag=dag)


def _api_refresh_task() -> HttpOperator:
    return HttpOperator(
        task_id="api_refresh",
        http_conn_id="api_prod",
        endpoint="/admin/refresh",
        method="POST",
        log_response=True,
    )


with DAG(
    dag_id="bike_traffic_init",
    description="Production-like historical bootstrap through job-runner-api.",
    start_date=datetime(2025, 9, 1),
    schedule=None,
    catchup=False,
    max_active_runs=concurrency.child_max_active_runs,
    max_active_tasks=concurrency.child_max_active_tasks,
    tags=["mlops", "init", "bike", "prod"],
    params=DAG_PARAMS,
) as init_dag:
    init_gate = ShortCircuitOperator(
        task_id="init_gate",
        python_callable=_init_gate_callable,
    )
    etl = build_etl_group(init_dag, mode="init")
    mark_done = PythonOperator(
        task_id="mark_init_done",
        python_callable=_mark_init_done_callable,
    )
    api_refresh = _api_refresh_task()
    init_gate >> etl >> mark_done >> api_refresh  # type: ignore


with DAG(
    dag_id="bike_traffic_daily",
    description="Production-like daily sliding window through job-runner-api.",
    start_date=datetime(2025, 9, 1),
    schedule=None,
    catchup=False,
    max_active_runs=concurrency.child_max_active_runs,
    max_active_tasks=concurrency.child_max_active_tasks,
    tags=["mlops", "daily", "bike", "prod"],
    params=DAG_PARAMS,
) as daily_dag:
    etl = build_etl_group(daily_dag, mode="daily")
    api_refresh = _api_refresh_task()
    etl >> api_refresh  # type: ignore
