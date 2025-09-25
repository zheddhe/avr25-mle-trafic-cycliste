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

from src.ml.features.features_utils import DatetimePeriodicsTransformer
from src.monitoring.metrics_push import track_pipeline_step  # <-- monitoring batch

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def write_manifest(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------------------
# Logs
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
    # Colonnes éventuellement redondantes (à garder si besoin)
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
# CLI
# -------------------------------------------------------------------
@click.command()
@click.option(
    "--interim-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Chemin du CSV intermédiaire produit par l’ingest.",
)
@click.option(
    "--sub-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Sous-dossier cible ASCII sous data/processed. Défaut: dérivé de interim-path.",
)
@click.option(
    "--processed-name",
    type=str,
    default="initial_with_feats.csv",
    show_default=True,
    help="Nom du CSV en sortie dans data/processed/<sub-dir>.",
)
@click.option(
    "--timestamp-col",
    type=str,
    default="date_et_heure_de_comptage",
    show_default=True,
    help="Nom de la colonne timestamp ISO8601 dans le CSV intermédiaire.",
)
@click.option(
    "--extra-drop",
    "extra_drop",
    type=str,
    multiple=True,
    help="Colonnes additionnelles à supprimer (option répétable).",
)
def main(
    interim_path: str,
    sub_dir: Optional[str],
    processed_name: str,
    timestamp_col: str,
    extra_drop: List[str],
) -> None:
    """
    Construit des features périodiques et écrit un CSV processed.
    Pousse des métriques batch (durée, volume) vers Pushgateway.
    """
    labels = {
        "dag": os.getenv("AIRFLOW_CTX_DAG_ID", "unknown_dag"),
        "task": os.getenv("AIRFLOW_CTX_TASK_ID", "etl.features"),
        "run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "local"),
    }

    with track_pipeline_step("features", labels) as m:
        # Lecture
        try:
            logger.info("Chargement du CSV intermédiaire [%s] ...", interim_path)
            df = pd.read_csv(interim_path, index_col=0)
        except Exception as exc:
            logger.exception("Lecture CSV échouée: %s", exc)
            raise click.ClickException(f"Echec lecture CSV: {exc}")

        if timestamp_col not in df.columns:
            raise click.ClickException(
                f"Colonne timestamp manquante dans le dataset: [{timestamp_col}]"
            )

        # Enrichissement des features temporelles
        tr_date = DatetimePeriodicsTransformer(timestamp_col=timestamp_col)
        df = tr_date.transform(df)

        # Filtrage des colonnes inutiles
        to_drop = [c for c in list(COLUMNS_TO_DROP) + list(extra_drop) if c in df.columns]
        if to_drop:
            logger.info("Suppression de %d colonne(s): %s", len(to_drop), to_drop)
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
            "CSV processed écrit -> [%s] (%d lignes, %d colonnes).",
            processed_path,
            len(df),
            df.shape[1],
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
                        "outputs": {"processed_path": str(processed_path)},
                        "run": {"run_id": os.getenv("RUN_ID"), "sub_dir": sub_dir},
                    },
                )
                logger.info("Manifest features écrit -> [%s].", man)
            except Exception as exc:
                logger.warning("Echec écriture manifest [%s]: %s", man, exc)

        # Volume traité pour la métrique
        m["records"] = int(len(df))

    logger.info("Feature engineering terminé avec succès.")

if __name__ == "__main__":
    try:
        main()
    except click.ClickException as e:
        logger.error(str(e))
        sys.exit(1)
