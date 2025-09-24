# src/airflow/dags/bike_traffic_pipeline_dag.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Mapping, Optional

from airflow import DAG
from airflow.models import Variable, TaskInstance
from airflow.models.param import Param, ParamsDict
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.task_group import TaskGroup
from docker.types import Mount
from common.utils import _load_config


# --------------------------------------------------------------------------- #
# Infra Variables
# --------------------------------------------------------------------------- #
# Docker images and runtime network
IMG_INGEST = Variable.get("docker_image_ingest", default_var="ml-ingest:dev")
IMG_FEATS = Variable.get("docker_image_features", default_var="ml-features:dev")
IMG_MODELS = Variable.get("docker_image_models", default_var="ml-models:dev")
DOCKER_NET = Variable.get("docker_network", default_var="mlops_net")

# Host and container paths
HOST_REPO = Variable.get("host_repo_root", default_var="/")
AIRFLOW_REPO = Variable.get("airflow_repo_root", default_var="/opt/airflow")
CONT_REPO = Variable.get("container_repo_root", default_var="/app")
MOUNTS = [
    Mount(source=f"{HOST_REPO}/data", target=f"{CONT_REPO}/data", type="bind"),
    Mount(source=f"{HOST_REPO}/logs", target=f"{CONT_REPO}/logs", type="bind"),
    Mount(source=f"{HOST_REPO}/models", target=f"{CONT_REPO}/models", type="bind"),
]

# MLflow / MinIO configuration
MLFLOW_TRACKING_URI = Variable.get("mlflow_tracking_uri", default_var="http://mlflow-server:5000")
MLFLOW_S3_ENDPOINT_URL = Variable.get(
    "mlflow_s3_endpoint_url",
    default_var="http://mlflow-minio:9000"
)
AWS_ACCESS_KEY_ID = Variable.get("aws_access_key_id", default_var="minio")
AWS_SECRET_ACCESS_KEY = Variable.get("aws_secret_access_key", default_var="minio123")
AWS_DEFAULT_REGION = Variable.get("aws_default_region", default_var="us-east-1")

# Airflow runtime configuration
TZ = Variable.get("tz", default_var="Europe/Paris")
AIRFLOW_UID = Variable.get("airflow_uid", default_var="0")
AIRFLOW_GID = Variable.get("airflow_gid", default_var="0")

# DAG parameters
DAG_PARAMS: ParamsDict = ParamsDict(
    {
        "counter_id": Param(
            default="Sebastopol_N-S_airflow",
            type="string",
            description="Counter to process for this run",
        )
    }
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_env(sub_dir: str, run_id: str, window: Dict[str, float]) -> Dict[str, str]:
    """
    Build environment variables for Docker tasks.

    Parameters
    ----------
    sub_dir : str
        The subdirectory used for run artifacts.
    run_id : str
        Unique identifier of the run.
    window : Dict[str, float]
        Window percentages (start, end).

    Returns
    -------
    Dict[str, str]
        Dictionary of environment variables.
    """
    artifact_root = f"{CONT_REPO}/logs/run/{sub_dir}/{run_id}"
    return {
        "RUN_ID": run_id,
        "SUB_DIR": sub_dir,
        "RANGE_START": str(window["start"]),
        "RANGE_END": str(window["end"]),
        "ARTIFACT_RUN_ROOT": artifact_root,
        "MANIFEST_INGEST": f"{artifact_root}/ingest/manifest.json",
        "MANIFEST_FEATS": f"{artifact_root}/features/manifest.json",
        "MANIFEST_MODELS": f"{artifact_root}/models/manifest.json",
        "TZ": TZ,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "MLFLOW_S3_ENDPOINT_URL": MLFLOW_S3_ENDPOINT_URL,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
        "AIRFLOW_UID": str(AIRFLOW_UID),
        "AIRFLOW_GID": str(AIRFLOW_GID),
    }


def _read_manifests(**ctx) -> Dict[str, Any]:
    """
    Read manifests (ingest, features, models) from previous tasks or build defaults.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys "ingest", "features" and "models".
    """
    ti: TaskInstance = ctx["ti"]
    run_id: str = ti.xcom_pull(key="RUN_ID", task_ids="etl.prepare_args")
    sub_dir: str = ti.xcom_pull(key="SUB_DIR", task_ids="etl.prepare_args")

    artifact_root = Path(HOST_REPO) / "data" / "runs" / sub_dir / run_id

    def _try_load(p: Path):
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    ing = _try_load(artifact_root / "ingest" / "manifest.json")
    fea = _try_load(artifact_root / "features" / "manifest.json")
    mod = _try_load(artifact_root / "models" / "manifest.json")

    if ing is None:
        ing = {
            "outputs": {
                "interim_path": f"{CONT_REPO}/data/interim/{sub_dir}/"
                                f"{ti.xcom_pull('etl.prepare_args', key='INTERIM_NAME')}"
            }
        }
    if fea is None:
        fea = {
            "inputs": {"interim_path": ing["outputs"]["interim_path"]},
            "outputs": {
                "processed_path": f"{CONT_REPO}/data/processed/{sub_dir}/"
                                  f"{ti.xcom_pull('etl.prepare_args', key='PROCESSED_NAME')}"
            },
        }
    if mod is None:
        mod = {
            "inputs": {"processed_path": fea["outputs"]["processed_path"]},
            "outputs": {
                "table": f"{CONT_REPO}/data/final/{sub_dir}/"
                         f"{ti.xcom_pull('etl.prepare_args', key='FINAL_BASENAME')}"
            },
        }

    ti.xcom_push(key="INGEST_MANIFEST", value=ing)
    ti.xcom_push(key="FEATURES_MANIFEST", value=fea)
    ti.xcom_push(key="MODELS_MANIFEST", value=mod)
    return {"ingest": ing, "features": fea, "models": mod}


# --------------------------------------------------------------------------- #
# Prepare args
# --------------------------------------------------------------------------- #
def _prepare_args_common(
        ti: TaskInstance, exec_date: datetime,
        mode: str, counter_override=None
) -> Dict[str, Any]:
    """
    Compute run context (IDs, window, subdir) and push it to XCom.

    Returns
    -------
    Dict[str, Any]
        Environment dictionary injected into Docker operators.
    """
    cfg, counter_id = _load_config()
    if counter_override:
        counter_id = counter_override
    sched = cfg.scheduling
    counter = cfg.counters[counter_id]

    run_id = (
        f"{exec_date.strftime('%Y%m%d')}_"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    )

    if mode == "init":
        delta_days = 0
        window = {"start": 0.0, "end": 75.0}
        proceed = True
    else:
        anchor_dt = datetime.fromisoformat(sched.initial_anchor_date).date()
        delta_days = (exec_date.date() - anchor_dt).days
        x = max(0.0, delta_days * float(sched.daily_increment_pct))
        start = max(0.0, min(100.0, 0.0 + x))
        end = max(0.0, min(100.0, 75.0 + x))
        window = {"start": round(start, 2), "end": round(end, 2)}
        proceed = end > 75.0

    effective_sub_dir = f"{counter_id}_day{delta_days}"

    # Push run context in XCom
    ti.xcom_push(key="RUN_ID", value=run_id)
    ti.xcom_push(key="WINDOW", value=window)
    ti.xcom_push(key="PROCEED", value=proceed)
    ti.xcom_push(key="SUB_DIR", value=effective_sub_dir)

    # Push container args in XCom
    ti.xcom_push(key="RAW_FILE_NAME", value=counter.raw_file_name)
    ti.xcom_push(key="SITE", value=counter.site)
    ti.xcom_push(key="ORIENTATION", value=counter.orientation)
    ti.xcom_push(key="INTERIM_NAME", value=counter.interim_name)
    ti.xcom_push(key="PROCESSED_NAME", value=counter.processed_name)
    ti.xcom_push(key="FINAL_BASENAME", value=counter.final_basename)

    ti.xcom_push(key="AR", value=str(counter.modeling.ar))
    ti.xcom_push(key="MM", value=str(counter.modeling.mm))
    ti.xcom_push(key="ROLL", value=str(counter.modeling.roll))
    ti.xcom_push(key="TEST_RATIO", value=str(counter.modeling.test_ratio))
    ti.xcom_push(key="GRID_ITER", value=str(counter.modeling.grid_iter))

    return _make_env(effective_sub_dir, run_id, window)


def _prepare_args_callable(mode: str):
    """
    Build a callable for prepare_args depending on mode (init or daily).
    """
    def _callable(**ctx):
        ti = ctx["ti"]
        exec_date = ctx["data_interval_end"]
        # Priority: DAG param, then DAG run conf (parent orchestrator)
        counter_override = (ctx.get("params") or {})["counter_id"]
        if not counter_override:
            counter_override = (ctx.get("dag_run") or {})["conf"]["counter_id"]
        return _prepare_args_common(
            ti=ti, exec_date=exec_date, mode=mode, counter_override=counter_override
        )
    return _callable


# --------------------------------------------------------------------------- #
# Build ETL group
# --------------------------------------------------------------------------- #
def build_etl_group(dag: DAG, mode: str) -> TaskGroup:
    """
    Build the ETL task group: prepare_args → ingest → features → models → read_manifests.
    """
    with TaskGroup(group_id="etl", dag=dag) as etl:
        prepare_args = PythonOperator(
            task_id="prepare_args",
            python_callable=_prepare_args_callable(mode),
            dag=dag,
        )

        if mode == "daily":
            def _gate(**ctx) -> bool:
                ti: TaskInstance = ctx["ti"]
                return bool(ti.xcom_pull(task_ids="etl.prepare_args", key="PROCEED"))

            gate = ShortCircuitOperator(task_id="gate", python_callable=_gate, dag=dag)
            prepare_args >> gate  # type: ignore
            upstream = gate
        else:
            upstream = prepare_args

        # Docker tasks, dynamically parameterized via XCom
        ingest = DockerOperator(
            task_id="ingest",
            image=IMG_INGEST,
            command=[
                "--raw-path",
                "{{ var.value.container_repo_root }}/data/raw/"
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='RAW_FILE_NAME') }}",
                "--site",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='SITE') }}",
                "--orientation",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='ORIENTATION') }}",
                "--range-start",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='WINDOW')['start'] }}",
                "--range-end",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='WINDOW')['end'] }}",
                "--timestamp-col",
                "date_et_heure_de_comptage",
                "--sub-dir",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='SUB_DIR') }}",
                "--interim-name",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='INTERIM_NAME') }}",
            ],
            environment=prepare_args.output,
            mounts=MOUNTS,
            docker_url="unix://var/run/docker.sock",
            api_version="auto",
            network_mode=DOCKER_NET,
            mount_tmp_dir=False,
            auto_remove=True,
            dag=dag,
        )

        features = DockerOperator(
            task_id="features",
            image=IMG_FEATS,
            command=[
                "--interim-path",
                "{{ var.value.container_repo_root }}/data/interim/"
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='SUB_DIR') }}/"
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='INTERIM_NAME') }}",
                "--sub-dir",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='SUB_DIR') }}",
                "--processed-name",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='PROCESSED_NAME') }}",
                "--timestamp-col",
                "date_et_heure_de_comptage",
            ],
            environment=prepare_args.output,
            mounts=MOUNTS,
            docker_url="unix://var/run/docker.sock",
            api_version="auto",
            network_mode=DOCKER_NET,
            mount_tmp_dir=False,
            auto_remove=True,
            dag=dag,
        )

        models = DockerOperator(
            task_id="models",
            image=IMG_MODELS,
            command=[
                "--processed-path",
                "{{ var.value.container_repo_root }}/data/processed/"
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='SUB_DIR') }}/"
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='PROCESSED_NAME') }}",
                "--sub-dir",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='SUB_DIR') }}",
                "--target-col",
                "comptage_horaire",
                "--ts-col-utc",
                "date_et_heure_de_comptage_utc",
                "--ts-col-local",
                "date_et_heure_de_comptage_local",
                "--ar",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='AR') }}",
                "--mm",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='MM') }}",
                "--roll",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='ROLL') }}",
                "--test-ratio",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='TEST_RATIO') }}",
                "--grid-iter",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='GRID_ITER') }}",
            ],
            environment=prepare_args.output,
            mounts=MOUNTS,
            docker_url="unix://var/run/docker.sock",
            api_version="auto",
            network_mode=DOCKER_NET,
            mount_tmp_dir=False,
            auto_remove=True,
            dag=dag,
        )

        read_manifests = PythonOperator(
            task_id="read_manifests",
            python_callable=_read_manifests,
            dag=dag,
        )

        upstream >> ingest >> features >> models >> read_manifests  # type: ignore

    return etl


# --------------------------------------------------------------------------- #
# Counter ID extraction and flags
# --------------------------------------------------------------------------- #
def _extract_counter_id(ctx: Mapping[str, Any]) -> Optional[str]:
    """Extract counter_id from DAG params or dag_run.conf."""
    params_obj = ctx.get("params")
    if isinstance(params_obj, Mapping):
        val = params_obj.get("counter_id")
        if isinstance(val, str) and val:
            return val
    dag_run = ctx.get("dag_run")
    conf_obj = getattr(dag_run, "conf", None)
    if isinstance(conf_obj, Mapping):
        val = conf_obj.get("counter_id")
        if isinstance(val, str) and val:
            return val
    return None


def _init_gate_callable(**ctx) -> bool:
    """Short-circuit init DAG if init already done for this counter."""
    counter_id = _extract_counter_id(ctx) or "UNKNOWN"
    key = f"bike_init_done__{counter_id}"
    return Variable.get(key, default_var="0") != "1"


def _mark_init_done_callable(**ctx) -> None:
    """Mark init done for this counter by setting Airflow variable."""
    counter_id = _extract_counter_id(ctx) or "UNKNOWN"
    key = f"bike_init_done__{counter_id}"
    Variable.set(key, "1")


# --------------------------------------------------------------------------- #
# DAGs: Init and Daily
# --------------------------------------------------------------------------- #
with DAG(
    dag_id="bike_traffic_init",
    description="One-shot historical bootstrap (with short-circuit).",
    start_date=datetime(2025, 9, 1),
    schedule=None,  # Triggered by orchestrator only (or manually)
    catchup=False,
    tags=["mlops", "init", "bike"],
    params=DAG_PARAMS,
) as init_dag:
    init_gate = ShortCircuitOperator(task_id="init_gate", python_callable=_init_gate_callable)
    etl = build_etl_group(init_dag, mode="init")
    mark_done = PythonOperator(task_id="mark_init_done", python_callable=_mark_init_done_callable)
    api_refresh = SimpleHttpOperator(
        task_id="api_refresh",
        http_conn_id="api_dev",
        endpoint="/admin/refresh",
        method="POST",
        response_check=lambda r: r.status_code == 200,
        log_response=True,
    )
    init_gate >> etl >> mark_done >> api_refresh  # type: ignore


with DAG(
    dag_id="bike_traffic_daily",
    description="Daily sliding window with DockerOperator (XCom-driven).",
    start_date=datetime(2025, 9, 1),
    schedule=None,  # Triggered by orchestrator only (or manually)
    catchup=False,
    tags=["mlops", "daily", "bike"],
    params=DAG_PARAMS,
) as daily_dag:
    etl = build_etl_group(daily_dag, mode="daily")
    api_refresh = SimpleHttpOperator(
        task_id="api_refresh",
        http_conn_id="api_dev",
        endpoint="/admin/refresh",
        method="POST",
        response_check=lambda r: r.status_code == 200,
        log_response=True,
    )
    etl >> api_refresh  # type: ignore
