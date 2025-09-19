# ruff: noqa
# mypy: ignore-errors
"""
Bike traffic pipeline — DockerOperator + XCom args (final, robust).

- prepare_args : calcule RUN_ID, fenêtre (init: 0–75 ; daily: glissante), pousse:
    * RUN_ID, WINDOW, PROCEED (bool), et tous les flags requis
    * retourne un dict ENV (injecté dans les 3 conteneurs via environment=prepare_args.output)
- gate (daily only) : lit PROCEED depuis XCom et court-circuite si False
- commandes Docker : Jinja pur, sans f-strings
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from airflow import DAG
from airflow.models import Variable, TaskInstance
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.task_group import TaskGroup
from docker.types import Mount

# --------------------------- Variables Airflow ------------------------------- #

SUB_DIR = Variable.get("bike_subdir", default_var="Sebastopol_N-S_mlops")

IMG_DATA = Variable.get("docker_image_data", default_var="ml-data:dev")
IMG_FEATS = Variable.get("docker_image_features", default_var="ml-features:dev")
IMG_MODELS = Variable.get("docker_image_models", default_var="ml-models:dev")

DOCKER_NET = Variable.get("docker_network", default_var="mlops_net")

HOST_REPO = Variable.get("host_repo_root", default_var="/")
CONT_REPO = Variable.get("container_repo_root", default_var="/app")

INC_PCT = float(Variable.get("daily_increment_pct", default_var="1.0"))
ANCHOR = Variable.get("initial_anchor_date", default_var=str(datetime.today().date()))

API_CONN_ID = Variable.get("api_conn_id", default_var="bike_api")
FINAL_BASENAME = Variable.get("final_basename", default_var="y_full.csv")

# Ingestion
RAW_FILE_NAME = Variable.get(
    "raw_file_name",
    default_var="comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv"
)
SITE = Variable.get("site", default_var="Totem 73 boulevard de Sébastopol")
ORIENTATION = Variable.get("orientation", default_var="N-S")
INTERIM_NAME = Variable.get("interim_name", default_var="initial.csv")

# Features
PROCESSED_NAME = Variable.get("processed_name", default_var="initial_with_feats.csv")

# Modeling
AR = Variable.get("ar", default_var="7")
MM = Variable.get("mm", default_var="1")
ROLL = Variable.get("roll", default_var="24")
TEST_RATIO = Variable.get("test_ratio", default_var="0.25")
GRID_ITER = Variable.get("grid_iter", default_var="0")

# MLflow / MinIO
MLFLOW_TRACKING_URI = Variable.get(
    "mlflow_tracking_uri",
    default_var="http://mlflow-server:5000"
)
MLFLOW_S3_ENDPOINT_URL = Variable.get(
    "mlflow_s3_endpoint_url",
    default_var="http://mlflow-minio:9000"
)
AWS_ACCESS_KEY_ID = Variable.get("aws_access_key_id", default_var="minio")
AWS_SECRET_ACCESS_KEY = Variable.get("aws_secret_access_key", default_var="minio123")
AWS_DEFAULT_REGION = Variable.get("aws_default_region", default_var="us-east-1")

# Divers
TZ = Variable.get("tz", default_var="Europe/Paris")
AIRFLOW_UID = Variable.get("airflow_uid", default_var="0")
AIRFLOW_GID = Variable.get("airflow_gid", default_var="0")

MOUNTS = [
    Mount(source=f"{HOST_REPO}/data", target=f"{CONT_REPO}/data", type="bind"),
    Mount(source=f"{HOST_REPO}/models", target=f"{CONT_REPO}/models", type="bind"),
]


# ------------------------------ Helpers ------------------------------------- #
def _make_env(run_id: str, window: Dict[str, float]) -> Dict[str, str]:
    artifact_root = f"{CONT_REPO}/data/runs/{SUB_DIR}/{run_id}"
    return {
        "RUN_ID": run_id,
        "SUB_DIR": SUB_DIR,
        "RANGE_START": str(window["start"]),
        "RANGE_END": str(window["end"]),
        "ARTIFACT_RUN_ROOT": artifact_root,
        "MANIFEST_INGEST": f"{artifact_root}/ingest/manifest.json",
        "MANIFEST_FEATS": f"{artifact_root}/features/manifest.json",
        "MANIFEST_MODELS": f"{artifact_root}/models/manifest.json",
        "TZ": TZ,
        # MLflow / MinIO
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "MLFLOW_S3_ENDPOINT_URL": MLFLOW_S3_ENDPOINT_URL,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
        # (optionnel) UID/GID si vos images en ont besoin
        "AIRFLOW_UID": str(AIRFLOW_UID),
        "AIRFLOW_GID": str(AIRFLOW_GID),
    }


def _host_paths() -> Dict[str, str]:
    final_root = Path(HOST_REPO) / "data" / "final" / SUB_DIR
    versions_root = final_root / "_versions"
    current_link = final_root / "current"
    return {
        "final_root": str(final_root),
        "versions_root": str(versions_root),
        "current_link": str(current_link),
    }


def _promote_artifacts(**ctx) -> None:
    ti: TaskInstance = ctx["ti"]
    run_id: str = ti.xcom_pull(key="RUN_ID", task_ids="etl.prepare_args")
    models_manifest: Dict[str, Any] = ti.xcom_pull(
        key="MODELS_MANIFEST", task_ids="etl.read_manifests"
    )
    paths = _host_paths()
    versions_root = Path(paths["versions_root"])
    current_link = Path(paths["current_link"])
    final_root = Path(paths["final_root"])

    versions_root.mkdir(parents=True, exist_ok=True)
    target_version_dir = versions_root / run_id
    target_version_dir.mkdir(parents=True, exist_ok=True)

    cont_final = models_manifest["outputs"]["table"]
    host_final = Path(str(cont_final).replace(CONT_REPO, HOST_REPO))

    dst_final = target_version_dir / FINAL_BASENAME
    dst_final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(host_final, dst_final)

    tmp_link = final_root / ".current_tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(target_version_dir)
    tmp_link.replace(current_link)


def _read_manifests(**ctx) -> Dict[str, Any]:
    ti: TaskInstance = ctx["ti"]
    run_id: str = ti.xcom_pull(key="RUN_ID", task_ids="etl.prepare_args")
    artifact_root = Path(HOST_REPO) / "data" / "runs" / SUB_DIR / run_id

    def _try_load(p: Path):
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    ing = _try_load(artifact_root / "ingest" / "manifest.json")
    fea = _try_load(artifact_root / "features" / "manifest.json")
    mod = _try_load(artifact_root / "models" / "manifest.json")

    # Fallbacks “conventionnels” si absent
    if ing is None:
        ing = {"outputs": {"interim_path": f"{CONT_REPO}/data/interim/{SUB_DIR}/{INTERIM_NAME}"}}
    if fea is None:
        fea = {
            "inputs": {"interim_path": ing["outputs"]["interim_path"]},
            "outputs": {"processed_path": f"{CONT_REPO}/data/processed/{SUB_DIR}/{PROCESSED_NAME}"},
        }
    if mod is None:
        mod = {
            "inputs": {"processed_path": fea["outputs"]["processed_path"]},
            "outputs": {"table": f"{CONT_REPO}/data/final/{SUB_DIR}/{FINAL_BASENAME}"},
        }

    ti.xcom_push(key="INGEST_MANIFEST", value=ing)
    ti.xcom_push(key="FEATURES_MANIFEST", value=fea)
    ti.xcom_push(key="MODELS_MANIFEST", value=mod)
    return {"ingest": ing, "features": fea, "models": mod}


# --------------------------- Prepare args ------------------------------------ #
def _prepare_args_common(ti: TaskInstance, exec_date: datetime, mode: str) -> Dict[str, Any]:
    """Calcule RUN_ID, fenêtre, pousse XCom, et retourne ENV (dict)."""
    run_id = f"{exec_date.strftime('%Y%m%d')}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    if mode == "init":
        window = {"start": 0.0, "end": 75.0}
        proceed = True
    else:
        delta_days = (exec_date.date() - datetime.fromisoformat(ANCHOR).date()).days
        x = max(0.0, delta_days * INC_PCT)
        start = max(0.0, min(100.0, 0.0 + x))
        end = max(0.0, min(100.0, 75.0 + x))
        window = {"start": round(start, 2), "end": round(end, 2)}
        proceed = end > 75.0

    # XComs
    ti.xcom_push(key="RUN_ID", value=run_id)
    ti.xcom_push(key="WINDOW", value=window)
    ti.xcom_push(key="PROCEED", value=proceed)

    # Flags pour conteneurs
    for k, v in {
        "RAW_FILE_NAME": RAW_FILE_NAME,
        "SITE": SITE,
        "ORIENTATION": ORIENTATION,
        "INTERIM_NAME": INTERIM_NAME,
        "PROCESSED_NAME": PROCESSED_NAME,
        "AR": str(AR),
        "MM": str(MM),
        "ROLL": str(ROLL),
        "TEST_RATIO": str(TEST_RATIO),
        "GRID_ITER": str(GRID_ITER),
    }.items():
        ti.xcom_push(key=k, value=v)

    env = _make_env(run_id, window)
    return env


def _prepare_args_callable(mode: str):
    def _callable(**ctx) -> Dict[str, Any]:
        ti: TaskInstance = ctx["ti"]
        exec_date = ctx["data_interval_end"]
        return _prepare_args_common(ti, exec_date, mode)
    return _callable


# --------------------------- Build ETL group --------------------------------- #
default_args = {"owner": "mlops", "retries": 1}


def build_etl_group(dag: DAG, mode: str) -> TaskGroup:
    with TaskGroup(group_id="etl", dag=dag) as etl:
        # 1) Préparer les args + ENV (retourné -> XComArg dict)
        prepare_args = PythonOperator(
            task_id="prepare_args",
            python_callable=_prepare_args_callable(mode),
            dag=dag,
        )

        # 2) Daily: barrière de poursuite selon PROCEED
        if mode == "daily":
            def _gate(**ctx) -> bool:
                ti: TaskInstance = ctx["ti"]
                return bool(ti.xcom_pull(task_ids="etl.prepare_args", key="PROCEED"))
            gate = ShortCircuitOperator(task_id="gate", python_callable=_gate, dag=dag)
            prepare_args >> gate  # type: ignore
            upstream = gate
        else:
            upstream = prepare_args

        # 3) Docker tasks — commandes en Jinja pur, env via dict (XComArg)
        ingest = DockerOperator(
            task_id="ingest",
            image=IMG_DATA,
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
                "{{ var.value.bike_subdir }}",
                "--interim-name",
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='INTERIM_NAME') }}",
            ],
            environment=prepare_args.output,  # <-- dict via XComArg
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
                "{{ var.value.container_repo_root }}/data/interim/{{ var.value.bike_subdir }}/"
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='INTERIM_NAME') }}",
                "--sub-dir",
                "{{ var.value.bike_subdir }}",
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
                "{{ var.value.container_repo_root }}/data/processed/{{ var.value.bike_subdir }}/"
                "{{ ti.xcom_pull(task_ids='etl.prepare_args', key='PROCESSED_NAME') }}",
                "--sub-dir",
                "{{ var.value.bike_subdir }}",
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


# ------------------------ DAG 1: initial load & deploy ----------------------- #
with DAG(
    dag_id="init_load_and_deploy_docker",
    description="Initial production [0 ; 75] with DockerOperator (XCom-driven).",
    start_date=datetime(2025, 9, 1),
    schedule="@once",
    catchup=False,
    default_args=default_args,
    tags=["mlops", "init", "bike", "docker"],
) as init_dag:
    etl = build_etl_group(init_dag, mode="init")

    promote = PythonOperator(task_id="promote_artifacts", python_callable=_promote_artifacts)
    api_refresh = SimpleHttpOperator(
        task_id="api_refresh",
        http_conn_id=API_CONN_ID,
        endpoint="/admin/refresh",
        method="POST",
        response_check=lambda r: r.status_code == 200,
        log_response=True,
    )
    etl >> promote >> api_refresh  # type: ignore

# ------------------------- DAG 2: daily window refresh ----------------------- #

with DAG(
    dag_id="daily_window_refresh_docker",
    description="Daily sliding window with DockerOperator (XCom-driven).",
    start_date=datetime(2025, 9, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args=default_args,
    tags=["mlops", "daily", "bike", "docker"],
) as daily_dag:
    etl = build_etl_group(daily_dag, mode="daily")

    promote = PythonOperator(task_id="promote_artifacts", python_callable=_promote_artifacts)
    api_refresh = SimpleHttpOperator(
        task_id="api_refresh",
        http_conn_id=API_CONN_ID,
        endpoint="/admin/refresh",
        method="POST",
        response_check=lambda r: r.status_code == 200,
        log_response=True,
    )
    etl >> promote >> api_refresh  # type: ignore
