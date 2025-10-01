# src/ml/models/train_and_predict.py
from __future__ import annotations

import os
import json
import sys
from pathlib import Path
import logging
from typing import Optional

import click
import pandas as pd
import numpy as np
import pytz
from mlflow.tracking import MlflowClient

from src.ml.models.models_utils import (
    train_timeseries_model,
    save_artefacts,
    track_pipeline_step,
    push_business_metrics
)
from src.ml.models.mlflow_tracking import (
    configure_mlflow_from_env,
    start_run,
    log_report_content,
    log_model_with_signature,
    log_local_artifacts,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def write_manifest(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _extract_site_orientation(sub_dir: str) -> tuple[str, str]:
    """
    Déduit (site, orientation) à partir d'un nom de sous-dossier.
    On ne garde que les deux premiers segments séparés par '_',
    tout le reste est ignoré.

    Exemples :
        'Sebastopol_N-S_mlops' -> ('Sebastopol', 'N-S')
        'Sebastopol_N-S_extra_suffix' -> ('Sebastopol', 'N-S')
        'Sebastopol' -> ('Sebastopol', 'NA')
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
        "ASCII Target <sub-dir> under data/processed. If not set, will write to "
        "data/final/<sub-dir-from-processed-path>"
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
    help="Base window (hours) for moving averages.",
)
@click.option(
    "--test-ratio",
    type=float,
    default=0.25,
    show_default=True,
    help="Fraction used for test split (chronological).",
)
@click.option(
    "--grid-iter",
    type=int,
    default=0,
    show_default=True,
    help="Bayesian search iterations (0 disables search).",
)
@click.option(
    "--mlflow-uri",
    type=str,
    default=None,
    help=("Optional MLflow tracking URI (overrides env " "MLFLOW_TRACKING_URI)."),
)
def main(
    processed_path: str,
    sub_dir: Optional[str],
    target_col: str,
    ts_col_utc: str,
    ts_col_local: str,
    ar: int,
    mm: int,
    roll: int,
    test_ratio: float,
    grid_iter: int,
    mlflow_uri: Optional[str],
) -> None:
    """
    Entraîne un modèle, journalise dans MLflow, puis pousse des métriques métier
    (RMSE, MAPE, fraîcheur, dernière obs/pred) vers Pushgateway.
    """
    # Labels Prometheus (grouping_key)
    labels = {
        "dag": os.getenv("AIRFLOW_CTX_DAG_ID", "unknown_dag"),
        "task": os.getenv("AIRFLOW_CTX_TASK_ID", "etl.models"),
        "run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "local"),
        "site": os.getenv("SITE", "NA"),
        "orientation": os.getenv("ORIENTATION", "NA"),
    }

    with track_pipeline_step("models", labels) as m:
        if not (0.0 < test_ratio < 0.95):
            raise click.BadParameter(f"test-ratio must be in (0, 0.95). Got {test_ratio}.")
        if grid_iter < 0:
            raise click.BadParameter(f"grid-iter must be >= 0. Got {grid_iter}.")

        # Configure MLflow (mlflow_uri CLI > env)
        configure_mlflow_from_env(explicit_uri=mlflow_uri)

        if sub_dir is None:
            sub_dir = os.path.basename(os.path.dirname(processed_path))

        # Données
        try:
            logger.info(f"Loading processed CSV [{processed_path}] ...")
            df = pd.read_csv(processed_path, index_col=0)
        except Exception as exc:
            logger.exception(f"Failed to load processed CSV: {exc}")
            raise click.ClickException(f"Failed to load processed CSV: {exc}")

        for col in [ts_col_utc, ts_col_local]:
            if col not in df.columns:
                raise click.ClickException(f"Missing required column: {col}")

        # Datetimes tz-aware
        try:
            df[ts_col_utc] = pd.to_datetime(
                df[ts_col_utc], format="%Y-%m-%d %H:%M:%S%z", utc=True
            )
        except Exception:
            df[ts_col_utc] = pd.to_datetime(df[ts_col_utc], utc=True)

        try:
            df[ts_col_local] = pd.to_datetime(
                df[ts_col_local], format="%Y-%m-%d %H:%M:%S%z", utc=True
            ).dt.tz_convert(pytz.timezone("Europe/Paris"))
        except Exception:
            df[ts_col_local] = pd.to_datetime(df[ts_col_local], utc=True).dt.tz_convert(
                pytz.timezone("Europe/Paris")
            )

        # Entraînement + forecast récursif
        ts_cols = [ts_col_local, ts_col_utc]
        report = train_timeseries_model(
            df,
            target_col=target_col,
            timestamp_cols=ts_cols,
            temp_feats=[ar, mm, roll],
            test_ratio=test_ratio,
            iter_grid_search=grid_iter,
        )

        # MLflow
        tags = {
            "counter.subdir": sub_dir,
            "model.family": "XGBRegressor",
        }
        with start_run(
            experiment_name=sub_dir,
            run_name=os.path.basename(processed_path),
            tags=tags,
        ):
            log_report_content(report, target_col=target_col)
            y_full_path = save_artefacts(report, sub_dir)

            # Log du modèle + signature
            x_sample = report["X_train"].head(1)
            log_model_with_signature(
                pipe_model=report["pipe_model"],
                sample_input_df=x_sample,
                artifact_path="model_pipeline",
                registered_name=f"{sub_dir}-model",
            )
            log_local_artifacts(sub_dir)

        # Métriques métier → Pushgateway
        try:
            # y_test / y_test_pred attendus dans report
            y_true = np.asarray(report["y_test"]).reshape(-1)
            y_pred = np.asarray(report["y_test_pred"]).reshape(-1)

            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            mape = float(
                np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
            )
            r2 = float(report["metrics"]["test"].get("R2", np.nan))

            n_true = int(y_true.size)
            n_pred = int(y_pred.size)
            day_offset = int(os.getenv("DAY_OFFSET", "0"))

            # Dernier timestamp connu côté données
            last_ts = pd.to_datetime(df[ts_col_utc].max(), utc=True).to_pydatetime()

            site_env = os.getenv("SITE")
            ori_env = os.getenv("ORIENTATION")
            if site_env and ori_env:
                site, orientation = site_env, ori_env
            else:
                site, orientation = _extract_site_orientation(sub_dir)

            push_business_metrics(
                site=site,
                orientation=orientation,
                rmse=rmse,
                mape=mape,
                r2=r2,
                n_obs_true=n_true,
                n_obs_pred=n_pred,
                last_ts=last_ts,           # fraîcheur “réelle”
                day_offset=day_offset,     # fraîcheur “simulée” (dayX)
            )
            logger.info(
                f"Pushed business metrics to Pushgateway: site={site}, ori={orientation}, "
                f"RMSE={rmse:.3f}, MAPE={mape:.2f}, R²={r2}, "
                f"last_ts={last_ts}, day_offset={day_offset}"
            )
        except Exception as exc:
            logger.warning(f"Failed to push business metrics: {exc}")

        # Manifest optionnel
        man = os.getenv("MANIFEST_MODELS")
        if man:
            try:
                write_manifest(
                    man,
                    {
                        "inputs": {
                            "processed_path": processed_path,
                            "target_col": target_col,
                            "ts_col_utc": ts_col_utc,
                            "ts_col_local": ts_col_local,
                            "ar": ar,
                            "mm": mm,
                            "roll": roll,
                            "test_ratio": test_ratio,
                            "grid_iter": grid_iter,
                        },
                        "outputs": {"table": str(y_full_path)},
                        "run": {"run_id": os.getenv("RUN_ID"), "sub_dir": sub_dir},
                    },
                )
                logger.info(f"Models manifest written to [{man}]")
            except Exception as exc:
                logger.warning(f"Failed to write models manifest [{man}]: {exc}")

        # Automatic Model Registry Promotion (safe initialization if needed)
        try:
            model_name = f"{sub_dir}-model"
            client = MlflowClient()

            # Ensure the registered model exists
            try:
                client.get_registered_model(model_name)
            except Exception:
                try:
                    client.create_registered_model(model_name)
                    logger.info(f"Created new registered model [{model_name}].")
                except Exception as exc:
                    logger.warning(
                        f"Failed to create registered model [{model_name}]: {exc}"
                    )
                    raise  # No point continuing without registry entry

            # Retrieve versions
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.warning(f"No version found for model [{model_name}].")
            else:
                latest = max(versions, key=lambda v: int(v.version))
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                try:
                    client.set_registered_model_alias(
                        model_name, "prod", latest.version
                    )
                except Exception as exc:
                    logger.warning(f"Alias 'prod' undefined: {exc}")
                logger.info(
                    f"Model [{model_name}] version {latest.version} "
                    f"promoted to production."
                )
        except Exception as exc:
            logger.warning(f"Model promotion failed: {exc}")

        # Volume traité pour la métrique: lignes de df
        m["records"] = int(len(df))

    logger.info("Training and forecasting ended successfully.")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except click.ClickException as e:
        logger.error(str(e))
        sys.exit(1)
