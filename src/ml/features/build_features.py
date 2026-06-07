# Build feature datasets.
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
import pandas as pd

from src.metrics.pipeline_metrics import track_pipeline_step
from src.ml.features.artifact_manifest_emission import (
    emit_feature_dataset_artifact_manifest,
)
from src.ml.features.features_utils import DatetimePeriodicsTransformer


log_dir = os.path.join("logs", "ml")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "build_features.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

COLUMNS_TO_DROP = [
    "nom_du_site_de_comptage",
    "orientation_compteur",
    "weather_code_wmo_code",
    "temperature_2m_c",
    "rain_mm",
    "snowfall_cm",
    "weather_code_wmo_code_category",
    "latitude",
    "longitude",
    "arrondissement",
    "elevation",
    "date_et_heure_de_comptage_hour",
    "date_et_heure_de_comptage_day",
    "date_et_heure_de_comptage_day_of_year",
    "date_et_heure_de_comptage_day_of_week",
    "date_et_heure_de_comptage_week",
    "date_et_heure_de_comptage_month",
    "date_et_heure_de_comptage_year",
    "date_et_heure_de_comptage_sin_week",
    "date_et_heure_de_comptage_cos_week",
    "date_et_heure_de_comptage_cos_day_of_year",
    "date_et_heure_de_comptage_sin_day_of_year",
]


@click.command()
@click.option(
    "--interim-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the interim CSV produced by ingestion.",
)
@click.option(
    "--sub-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Target subdir under data/processed.",
)
@click.option(
    "--processed-name",
    type=str,
    default="initial_with_feats.csv",
    show_default=True,
    help="Processed CSV filename inside data/processed/<sub-dir>.",
)
@click.option(
    "--timestamp-col",
    type=str,
    default="date_et_heure_de_comptage",
    show_default=True,
    help="Original ISO8601 timestamp column name.",
)
@click.option(
    "--extra-drop",
    "extra_drop",
    type=str,
    multiple=True,
    help="Additional columns to drop. Can be used multiple times.",
)
@click.option(
    "--artifact-manifest-root",
    type=click.Path(file_okay=False),
    default=None,
    envvar="ARTIFACT_MANIFEST_ROOT",
    help="Optional directory used to write feature dataset manifests.",
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
    help="Optional s3:// URI for the feature dataset artifact.",
)
def main(
    interim_path: str,
    sub_dir: str | None,
    processed_name: str,
    timestamp_col: str,
    extra_drop: list[str],
    artifact_manifest_root: str | None,
    artifact_repository_root: str,
    artifact_object_uri: str | None,
) -> None:
    """Build periodic datetime features and write a processed dataset."""

    labels = {
        "dag": os.getenv("AIRFLOW_CTX_DAG_ID", "unknown_dag"),
        "task": os.getenv("AIRFLOW_CTX_TASK_ID", "etl.features"),
        "run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "local"),
        "site": os.getenv("SITE", "NA"),
        "site_short": os.getenv("SITE_SHORT", "NA"),
        "orientation": os.getenv("ORIENTATION", "NA"),
    }

    with track_pipeline_step("features", labels) as metrics_payload:
        try:
            logger.info(f"Loading interim CSV [{interim_path}] ...")
            df = pd.read_csv(interim_path, index_col=0)
        except Exception as exc:
            logger.exception(f"Failed to load interim CSV: {exc}")
            raise click.ClickException(f"Failed to load interim CSV: {exc}") from exc

        if timestamp_col not in df.columns:
            raise click.ClickException(
                f"Timestamp column [{timestamp_col}] not found in interim dataset."
            )

        transformer = DatetimePeriodicsTransformer(timestamp_col=timestamp_col)
        df = transformer.transform(df)

        to_drop = [
            column
            for column in list(COLUMNS_TO_DROP) + list(extra_drop)
            if column in df.columns
        ]
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} column(s): {to_drop}")
            df = df.drop(columns=to_drop)

        if sub_dir is None:
            sub_dir = os.path.basename(os.path.dirname(interim_path))
        out_dir = os.path.join("data", "processed", sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        processed_path = os.path.join(out_dir, processed_name)

        df.to_csv(processed_path, index=True)
        logger.info(
            f"Saved processed CSV to [{processed_path}]"
            f" ({len(df)} rows, {df.shape[1]} cols)."
        )

        emitted_manifest = emit_feature_dataset_artifact_manifest(
            manifest_root=artifact_manifest_root,
            payload_path=processed_path,
            source_file_name=Path(interim_path).name,
            sub_dir=sub_dir,
            repository_root=artifact_repository_root,
            run_id=os.getenv("RUN_ID") or os.getenv("AIRFLOW_CTX_DAG_RUN_ID"),
            counter_id=os.getenv("COUNTER_ID") or sub_dir,
            dataset_version=os.getenv("DATASET_VERSION"),
            producer_service=os.getenv("ARTIFACT_PRODUCER_SERVICE", "ml-features"),
            producer_image=os.getenv("ARTIFACT_PRODUCER_IMAGE"),
            producer_version=os.getenv("ARTIFACT_PRODUCER_VERSION"),
            object_uri=artifact_object_uri,
            promote=True,
        )
        if emitted_manifest is not None:
            logger.info(
                "Feature dataset artifact manifest emitted for "
                f"run_id=[{emitted_manifest.run_id}].",
            )

        metrics_payload["records"] = int(len(df))

    logger.info("Feature engineering ended successfully.")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except click.ClickException as error:
        logger.error(str(error))
        sys.exit(1)
