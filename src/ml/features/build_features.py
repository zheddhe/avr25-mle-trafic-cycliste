# src/ml/features/build_features.py
from __future__ import annotations

import os
import json
import sys
from pathlib import Path
import logging
from typing import Optional, List

import click
import pandas as pd

from src.ml.features.features_utils import (
    DatetimePeriodicsTransformer,
    track_pipeline_step,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def write_manifest(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -------------------------------------------------------------------
# Logs Management
# -------------------------------------------------------------------
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
    # Optionally redundant engineered columns (keep if you want)
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


# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------
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
    help=("ASCII Target <sub-dir> under data/processed. If not set, will write to "
          "data/processed/<sub-dir-from-interim-path>"),
)
@click.option(
    "--processed-name",
    type=str,
    default="initial_with_feats.csv",
    show_default=True,
    help="processed CSV filename inside data/processed/<sub-dir>.",
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
    help="Additional columns to drop (can be used multiple times).",
)
def main(
    interim_path: str,
    sub_dir: Optional[str],
    processed_name: str,
    timestamp_col: str,
    extra_drop: List[str],
) -> None:
    """
    Build periodic datetime features and save a processed CSV.
    Push batch metrics (duration, volume) to pushgateway.
    """
    labels = {
        "dag": os.getenv("AIRFLOW_CTX_DAG_ID", "unknown_dag"),
        "task": os.getenv("AIRFLOW_CTX_TASK_ID", "etl.features"),
        "run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "local"),
        "site": os.getenv("SITE", "NA"),
        "orientation": os.getenv("ORIENTATION", "NA"),
    }

    with track_pipeline_step("features", labels) as m:
        # Lecture
        try:
            logger.info(f"Loading interim CSV [{interim_path}] ...")
            df = pd.read_csv(interim_path, index_col=0)
        except Exception as exc:
            logger.exception(f"Failed to load interim CSV: {exc}")
            raise click.ClickException(f"Failed to load interim CSV: {exc}")

        if timestamp_col not in df.columns:
            raise click.ClickException(
                f"Timestamp column [{timestamp_col}] not found in interim dataset."
            )

        # Enrichissement des features temporelles
        tr_date = DatetimePeriodicsTransformer(timestamp_col=timestamp_col)
        df = tr_date.transform(df)

        # Filtrage des colonnes inutiles
        to_drop = [c for c in list(COLUMNS_TO_DROP) + list(extra_drop) if c in df.columns]
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} column(s): {to_drop}")
            df = df.drop(columns=to_drop)

        # Résolution du chemin de sortie
        if sub_dir is None:
            sub_dir = os.path.basename(os.path.dirname(interim_path))
        out_dir = os.path.join("data", "processed", sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        processed_path = os.path.join(out_dir, processed_name)

        # Ecriture
        df.to_csv(processed_path, index=True)
        logger.info(
            f"Saved processed CSV to [{processed_path}]"
            f" ({len(df)} rows, {df.shape[1]} cols)."
        )

        # Manifest optionnel
        man = os.getenv("MANIFEST_FEATS")
        if man:
            try:
                write_manifest(
                    man,
                    {
                        "inputs": {
                            "interim_path": str(interim_path),
                            "timestamp_col": timestamp_col,
                            "extra_drop": list(extra_drop) if extra_drop else [],
                        },
                        "outputs": {
                            "processed_path": str(processed_path)
                        },
                        "run": {
                            "run_id": os.getenv("RUN_ID"),
                            "sub_dir": sub_dir
                        }
                    }
                )
                logger.info(f"Features manifest written to [{man}].")
            except Exception as exc:
                logger.warning(f"Failed to write manifest [{man}]: {exc}")

        # Volume traité pour la métrique
        m["records"] = int(len(df))

    logger.info("Feature engineering ended successfully.")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except click.ClickException as e:
        logger.error(str(e))
        sys.exit(1)
