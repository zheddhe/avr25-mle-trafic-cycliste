# src/ml/models/train_and_predict.py
from __future__ import annotations

import logging
import os
import sys

import click
import numpy as np
import pandas as pd
import pytz

from src.metrics.pipeline_metrics import track_pipeline_step
from src.ml.models.artifact_manifest_emission import (
    emit_prediction_artifact_manifest,
)
from src.ml.models.mlflow_tracking import (
    configure_mlflow_from_env,
    is_model_registry_enabled,
    log_local_artifacts,
    log_model_with_signature,
    log_report_content,
    start_run,
)
from src.ml.models.models_utils import (
    push_business_metrics,
    save_artefacts,
    train_timeseries_model,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _extract_site_orientation(sub_dir: str) -> tuple[str, str]:
    """
    Extract site and orientation from a pipeline sub-directory name.
    """

    parts = sub_dir.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], "NA"


# -------------------------------------------------------------------
# Logs management
# -------------------------------------------------------------------
log_dir = os.path.join("logs", "ml")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "train_and_predict.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------
@click.command()
@click.option(
    "--processed-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to processed CSV with *_utc and *_local columns.",
)
@click.option(
    "--sub-dir",
    type=str,
    default=None,
    help=(
        "ASCII target <sub-dir> under data/processed. If not set, write to "
        "data/final/<sub-dir-from-processed-path>."
    ),
)
@click.option(
    "--target-col",
    type=str,
    default="comptage_horaire",
    show_default=True,
    help="Target column for regression.",
)
@click.option(
    "--ts-col-utc",
    type=str,
    default="date_et_heure_de_comptage_utc",
    show_default=True,
    help="UTC timestamp column name.",
)
@click.option(
    "--ts-col-local",
    type=str,
    default="date_et_heure_de_comptage_local",
    show_default=True,
    help="Local timestamp column name.",
)
@click.option(
    "--ar",
    type=int,
    default=7,
    show_default=True,
    help="Number of AR lags.",
)
@click.option(
    "--mm",
    type=int,
    default=1,
    show_default=True,
    help="Number of moving averages.",
)
@click.option(
    "--roll",
    type=int,
    default=24,
    show_default=True,
    help="Base window in hours for moving averages.",
)
@click.option(
    "--test-ratio",
    type=float,
    default=0.25,
    show_default=True,
    help="Fraction used for chronological test split.",
)
@click.option(
    "--grid-iter",
    type=int,
    default=0,
    show_default=True,
    help="Bayesian search iterations. Zero disables search.",
)
@click.option(
    "--mlflow-uri",
    type=str,
    default=None,
    help="Optional MLflow tracking URI. Overrides MLFLOW_TRACKING_URI.",
)
@click.option(
    "--artifact-manifest-root",
    type=click.Path(file_okay=False),
    default=None,
    envvar="ARTIFACT_MANIFEST_ROOT",
    help="Optional directory used to write prediction artifact manifests.",
)
@click.option(
    "--artifact-repository-root",
    type=click.Path(file_okay=False),
    default=".",
    envvar="ARTIFACT_REPOSITORY_ROOT",
    show_default=True,
    help="Repository root used to resolve manifest local artifact paths.",
)
@click.option(
    "--artifact-object-uri",
    type=str,
    default=None,
    envvar="ARTIFACT_OBJECT_URI",
    help="Optional s3:// URI for the prediction artifact.",
)
def main(
    processed_path: str,
    sub_dir: str | None,
    target_col: str,
    ts_col_utc: str,
    ts_col_local: str,
    ar: int,
    mm: int,
    roll: int,
    test_ratio: float,
    grid_iter: int,
    mlflow_uri: str | None,
    artifact_manifest_root: str | None,
    artifact_repository_root: str,
    artifact_object_uri: str | None,
) -> None:
    """
    Train a time-series model, persist predictions, and emit optional manifests.
    """

    labels = {
        "dag": os.getenv("AIRFLOW_CTX_DAG_ID", "unknown_dag"),
        "task": os.getenv("AIRFLOW_CTX_TASK_ID", "etl.models"),
        "run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "local"),
        "site": os.getenv("SITE", "NA"),
        "site_short": os.getenv("SITE_SHORT", "NA"),
        "orientation": os.getenv("ORIENTATION", "NA"),
    }

    with track_pipeline_step("models", labels) as metrics_payload:
        if not (0.0 < test_ratio < 0.95):
            raise click.BadParameter(
                f"test-ratio must be in (0, 0.95). Got {test_ratio}.",
            )
        if grid_iter < 0:
            raise click.BadParameter(f"grid-iter must be >= 0. Got {grid_iter}.")

        configure_mlflow_from_env(explicit_uri=mlflow_uri)
        registry_enabled = is_model_registry_enabled(explicit_uri=mlflow_uri)

        if sub_dir is None:
            sub_dir = os.path.basename(os.path.dirname(processed_path))

        try:
            logger.info(f"Loading processed CSV [{processed_path}] ...")
            df = pd.read_csv(processed_path, index_col=0)
        except Exception as exc:
            logger.exception(f"Failed to load processed CSV: {exc}")
            raise click.ClickException(
                f"Failed to load processed CSV: {exc}",
            ) from exc

        for col in [ts_col_utc, ts_col_local]:
            if col not in df.columns:
                raise click.ClickException(f"Missing required column: {col}")

        try:
            df[ts_col_utc] = pd.to_datetime(
                df[ts_col_utc],
                format="%Y-%m-%d %H:%M:%S%z",
                utc=True,
            )
        except Exception:
            df[ts_col_utc] = pd.to_datetime(df[ts_col_utc], utc=True)

        try:
            df[ts_col_local] = pd.to_datetime(
                df[ts_col_local],
                format="%Y-%m-%d %H:%M:%S%z",
                utc=True,
            ).dt.tz_convert(pytz.timezone("Europe/Paris"))
        except Exception:
            df[ts_col_local] = pd.to_datetime(
                df[ts_col_local],
                utc=True,
            ).dt.tz_convert(pytz.timezone("Europe/Paris"))

        ts_cols = [ts_col_local, ts_col_utc]
        report = train_timeseries_model(
            df,
            target_col=target_col,
            timestamp_cols=ts_cols,
            temp_feats=[ar, mm, roll],
            test_ratio=test_ratio,
            iter_grid_search=grid_iter,
        )

        site_short = os.getenv("SITE_SHORT")
        if not site_short:
            site_short = sub_dir
            log_message = (
                f"SITE_SHORT not set, fallback to sub_dir=[{sub_dir}] "
                "to construct registration model name"
            )
            if registry_enabled:
                logger.warning(log_message)
            else:
                logger.info(log_message)

        tags = {
            "counter.subdir": sub_dir,
            "site.short": site_short,
            "model.family": "XGBRegressor",
        }
        model_version = os.getenv("MODEL_VERSION")
        with start_run(
            experiment_name=site_short,
            run_name=os.path.basename(processed_path),
            tags=tags,
        ) as mlflow_run:
            if mlflow_run is not None:
                model_version = mlflow_run.info.run_id

            log_report_content(report, target_col=target_col)
            y_full_path = save_artefacts(report, sub_dir)

            x_sample = report["X_train"].head(1)
            registered_model_name = (
                f"{site_short}-model" if registry_enabled else None
            )
            log_model_with_signature(
                pipe_model=report["pipe_model"],
                sample_input_df=x_sample,
                model_artifact_name="model_pipeline",
                registered_name=registered_model_name,
                registry_enabled=registry_enabled,
            )
            log_local_artifacts(sub_dir)

        _emit_manifest_or_raise(
            artifact_manifest_root=artifact_manifest_root,
            artifact_repository_root=artifact_repository_root,
            artifact_object_uri=artifact_object_uri,
            prediction_path=y_full_path,
            processed_path=processed_path,
            sub_dir=sub_dir,
            model_version=model_version,
        )

        _promote_latest_model_alias(
            site_short=site_short,
            registry_enabled=registry_enabled,
        )
        _push_business_metrics(
            report=report,
            df=df,
            ts_col_utc=ts_col_utc,
            sub_dir=sub_dir,
        )

        metrics_payload["records"] = int(len(df))

    logger.info("Training and forecasting ended successfully.")
    sys.exit(0)


def _emit_manifest_or_raise(
    *,
    artifact_manifest_root: str | None,
    artifact_repository_root: str,
    artifact_object_uri: str | None,
    prediction_path: str,
    processed_path: str,
    sub_dir: str,
    model_version: str | None,
) -> None:
    try:
        emitted_manifest = emit_prediction_artifact_manifest(
            manifest_root=artifact_manifest_root,
            prediction_path=prediction_path,
            processed_path=processed_path,
            sub_dir=sub_dir,
            repository_root=artifact_repository_root,
            run_id=os.getenv("RUN_ID") or os.getenv("AIRFLOW_CTX_DAG_RUN_ID"),
            counter_id=os.getenv("COUNTER_ID") or sub_dir,
            dataset_version=os.getenv("DATASET_VERSION"),
            model_version=model_version,
            producer_service=os.getenv(
                "ARTIFACT_PRODUCER_SERVICE",
                "ml-models",
            ),
            producer_image=os.getenv("ARTIFACT_PRODUCER_IMAGE"),
            producer_version=os.getenv("ARTIFACT_PRODUCER_VERSION"),
            object_uri=artifact_object_uri,
            promote=True,
        )
    except Exception as exc:
        logger.exception("Failed to emit prediction artifact manifest")
        raise click.ClickException(
            f"Failed to emit prediction artifact manifest: {exc}",
        ) from exc

    if emitted_manifest is not None:
        logger.info(
            f"Prediction artifact manifest emitted for "
            f"run_id=[{emitted_manifest.run_id}]."
        )


def _promote_latest_model_alias(
    *,
    site_short: str,
    registry_enabled: bool,
) -> None:
    if not registry_enabled:
        logger.info(
            "Model registry promotion skipped because no MLflow tracking URI "
            "is configured."
        )
        return

    try:
        from mlflow.tracking import MlflowClient

        model_name = f"{site_short}-model"
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.warning(f"No version found for model [{model_name}].")
            return

        latest = max(versions, key=lambda version: int(version.version))
        try:
            client.set_registered_model_alias(
                model_name,
                "prod",
                latest.version,
            )
        except Exception as exc:
            logger.warning(f"Alias 'prod' undefined: {exc}")

        logger.info(
            f"Model [{model_name}] version {latest.version} "
            f"promoted to production."
        )
    except Exception as exc:
        logger.warning(f"Model promotion failed: {exc}")


def _push_business_metrics(
    *,
    report: dict,
    df: pd.DataFrame,
    ts_col_utc: str,
    sub_dir: str,
) -> None:
    try:
        y_train_pred = np.asarray(report["y_train_pred"]).reshape(-1)
        y_test = np.asarray(report["y_test"]).reshape(-1)
        y_test_pred = np.asarray(report["y_test_pred"]).reshape(-1)

        rmse = float(np.sqrt(np.mean((y_test_pred - y_test) ** 2)))
        mape = float(
            np.mean(
                np.abs(
                    (y_test - y_test_pred)
                    / np.clip(np.abs(y_test), 1e-6, None),
                ),
            )
            * 100.0
        )
        drift = float(np.mean(y_test) - np.mean(y_train_pred))
        push_business_metrics(
            rmse=rmse,
            mape=mape,
            data_drift=drift,
            last_timestamp=df[ts_col_utc].max(),
            site=sub_dir,
        )
    except Exception as exc:
        logger.warning(f"Business metrics push skipped: {exc}")
