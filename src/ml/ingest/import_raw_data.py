# Ingest raw counter data.
from __future__ import annotations

import logging
import os
import re
import sys
import unicodedata
from pathlib import Path

import click
import pandas as pd

from src.metrics.pipeline_metrics import track_pipeline_step
from src.ml.ingest.artifact_manifest_emission import (
    emit_interim_dataset_artifact_manifest,
)
from src.ml.ingest.ingest_utils import apply_percent_range_selection


def slugify_ascii(text: str) -> str:
    """Build an ASCII-safe slug from a counter label."""

    normalized = unicodedata.normalize("NFKD", text)
    text_ascii = normalized.encode("ascii", "ignore").decode("ascii")
    text_ascii = re.sub(r"[^A-Za-z0-9\-_]+", "_", text_ascii)
    text_ascii = re.sub(r"_+", "_", text_ascii).strip("_")
    return text_ascii[:64] or "counter"


LOG_DIR = os.path.join("logs", "ml")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "import_raw_data.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--raw-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the raw CSV file.",
)
@click.option(
    "--site",
    type=str,
    required=True,
    help="Exact value of column 'nom_du_site_de_comptage'.",
)
@click.option(
    "--orientation",
    type=str,
    required=True,
    help="Exact value of column 'orientation_compteur'.",
)
@click.option(
    "--range-start",
    type=float,
    default=0.0,
    show_default=True,
    help="Start percent [0..100] for chronological slice.",
)
@click.option(
    "--range-end",
    type=float,
    default=100.0,
    show_default=True,
    help="End percent [0..100] for chronological slice.",
)
@click.option(
    "--timestamp-col",
    type=str,
    default="date_et_heure_de_comptage",
    show_default=True,
    help="Timestamp column in raw CSV.",
)
@click.option(
    "--sub-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Target subdir under data/interim.",
)
@click.option(
    "--interim-name",
    type=str,
    default="initial.csv",
    show_default=True,
    help="Output CSV filename inside data/interim/<sub-dir>/.",
)
@click.option(
    "--artifact-manifest-root",
    type=click.Path(file_okay=False),
    default=None,
    envvar="ARTIFACT_MANIFEST_ROOT",
    help="Optional directory used to write interim dataset manifests.",
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
    help="Optional s3:// URI for the interim dataset artifact.",
)
def main(
    raw_path: str,
    site: str,
    orientation: str,
    range_start: float,
    range_end: float,
    timestamp_col: str,
    sub_dir: str | None,
    interim_name: str,
    artifact_manifest_root: str | None,
    artifact_repository_root: str,
    artifact_object_uri: str | None,
) -> None:
    """Extract a single counter and write an interim dataset slice."""

    labels = {
        "dag": os.getenv("AIRFLOW_CTX_DAG_ID", "unknown_dag"),
        "task": os.getenv("AIRFLOW_CTX_TASK_ID", "etl.ingest"),
        "run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "local"),
        "site": os.getenv("SITE", "NA"),
        "site_short": os.getenv("SITE_SHORT", "NA"),
        "orientation": os.getenv("ORIENTATION", "NA"),
    }

    with track_pipeline_step("ingest", labels) as metrics_payload:
        if not (0.0 <= range_start <= 100.0 and 0.0 <= range_end <= 100.0):
            raise click.BadParameter(
                "range-start/range-end must be within [0, 100]. "
                f"Got ({range_start}, {range_end}).",
            )
        if range_start > range_end:
            raise click.BadParameter(
                "range-start must be <= range-end. "
                f"Got ({range_start}, {range_end}).",
            )

        try:
            LOGGER.info(f"Loading raw CSV [{raw_path}] ...")
            df = pd.read_csv(raw_path, index_col=0)
        except Exception as exc:
            LOGGER.exception("Failed to load raw CSV")
            raise click.ClickException(f"Failed to load raw CSV: {exc}") from exc

        key_cols = ["nom_du_site_de_comptage", "orientation_compteur"]
        missing = [col for col in key_cols + [timestamp_col] if col not in df.columns]
        if missing:
            raise click.ClickException(f"Missing required columns: {missing}")

        grouped = df.groupby(key_cols, dropna=False)
        key = (site, orientation)
        if key not in grouped.groups:
            raise click.ClickException(f"Counter not found: {key}")

        df_counter = grouped.get_group(key).copy()
        LOGGER.info(f"Counter [{site} | {orientation}] rows: {len(df_counter)}")

        try:
            df_counter[timestamp_col] = pd.to_datetime(
                df_counter[timestamp_col],
                format="%Y-%m-%dT%H:%M:%S%z",
                utc=True,
            )
        except Exception:
            df_counter[timestamp_col] = pd.to_datetime(
                df_counter[timestamp_col],
                utc=True,
            )

        df_counter = df_counter.sort_values(timestamp_col).reset_index(drop=True)
        df_counter = apply_percent_range_selection(
            df_counter,
            (range_start, range_end),
        )
        if df_counter.empty:
            raise click.ClickException("Slice produced an empty DataFrame.")

        if sub_dir is None:
            sub_dir = slugify_ascii(f"{site}_{orientation}")
        out_dir = os.path.join("data", "interim", sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, interim_name)

        df_counter.to_csv(out_path, index=True)
        LOGGER.info(f"Saved interim slice -> [{out_path}] ({len(df_counter)} rows)")

        emitted_manifest = emit_interim_dataset_artifact_manifest(
            manifest_root=artifact_manifest_root,
            payload_path=out_path,
            source_file_name=Path(raw_path).name,
            sub_dir=sub_dir,
            repository_root=artifact_repository_root,
            run_id=os.getenv("RUN_ID") or os.getenv("AIRFLOW_CTX_DAG_RUN_ID"),
            counter_id=os.getenv("COUNTER_ID") or sub_dir,
            producer_service=os.getenv("ARTIFACT_PRODUCER_SERVICE", "ml-ingest"),
            producer_image=os.getenv("ARTIFACT_PRODUCER_IMAGE"),
            producer_version=os.getenv("ARTIFACT_PRODUCER_VERSION"),
            object_uri=artifact_object_uri,
            promote=True,
        )
        if emitted_manifest is not None:
            LOGGER.info(
                "Interim dataset artifact manifest emitted for "
                f"run_id=[{emitted_manifest.run_id}].",
            )

        metrics_payload["records"] = int(len(df_counter))

    LOGGER.info("Data ingestion ended successfully.")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except click.ClickException as error:
        LOGGER.error(str(error))
        sys.exit(1)
