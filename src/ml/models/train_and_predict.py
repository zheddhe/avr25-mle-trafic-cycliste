# src/ml/models/train_and_predict.py
from __future__ import annotations

import os
import logging
from typing import Optional
import click
import pandas as pd
import pytz

from src.ml.models.models_utils import (
    train_timeseries_model,
    save_artefacts,
)
from src.ml.models.mlflow_tracking import (
    configure_mlflow_from_env,
    start_run,
    log_report_content,
    log_model_with_signature,
    log_local_artifacts,
)


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
    help=("ASCII Target <sub-dir> under data/processed. If not set, will write to "
          "data/final/<sub-dir-from-processed-path>"),
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
    help=("Optional MLflow tracking URI (overrides env "
          "MLFLOW_TRACKING_URI)."),
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
    Train a time-series model and produce recursive forecasts with MLflow logs.
    """
    # Configure MLflow URI from env, optionally override by CLI
    configure_mlflow_from_env(explicit_uri=mlflow_uri)

    if sub_dir is None:
        sub_dir = os.path.basename(os.path.dirname(processed_path))

    try:
        logger.info(f"Loading processed CSV [{processed_path}] ...")
        df = pd.read_csv(processed_path, index_col=0)
    except Exception as exc:
        logger.exception(f"Failed to load processed CSV: {exc}")
        exit(1)

    for col in [ts_col_utc, ts_col_local]:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            exit(1)

    # Ensure tz-aware datetimes
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
        df[ts_col_local] = pd.to_datetime(
            df[ts_col_local], utc=True
        ).dt.tz_convert(pytz.timezone("Europe/Paris"))

    # Train + recursive forecast
    ts_cols = [ts_col_local, ts_col_utc]
    report = train_timeseries_model(
        df,
        target_col=target_col,
        timestamp_cols=ts_cols,
        temp_feats=[ar, mm, roll],
        test_ratio=test_ratio,
        iter_grid_search=grid_iter,
    )

    # MLflow logging + local artifacts
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
        save_artefacts(report, sub_dir)

        # Log model with signature using one train row
        x_sample = report["X_train"].head(1)
        log_model_with_signature(
            pipe_model=report["pipe_model"],
            sample_input_df=x_sample,
            artifact_path="model_pipeline",
            registered_name=f"{sub_dir}-model",
        )
        log_local_artifacts(sub_dir)

    logger.info("Training and forecasting ended successfully.")
    exit(0)


if __name__ == "__main__":
    main()
