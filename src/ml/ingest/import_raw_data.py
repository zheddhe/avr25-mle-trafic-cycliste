# src/ml/ingest/import_raw_data.py
from __future__ import annotations

import os
import json
import sys
from pathlib import Path
import re
import logging
import unicodedata
from typing import Optional

import click
import pandas as pd

from src.ml.ingest.data_utils import apply_percent_range_selection
from src.monitoring.metrics_push import track_pipeline_step  # <-- monitoring batch

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def write_manifest(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def slugify_ascii(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    text_ascii = norm.encode("ascii", "ignore").decode("ascii")
    text_ascii = re.sub(r"[^A-Za-z0-9\-_]+", "_", text_ascii)
    text_ascii = re.sub(r"_+", "_", text_ascii).strip("_")
    return text_ascii[:64] or "counter"


# -------------------------------------------------------------------
# Logs
# -------------------------------------------------------------------
log_dir = os.path.join("logs", "ml")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "import_raw_data.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
@click.command()
@click.option(
    "--raw-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Chemin du CSV brut.",
)
@click.option(
    "--site",
    type=str,
    required=True,
    help="Valeur exacte de 'nom_du_site_de_comptage'.",
)
@click.option(
    "--orientation",
    type=str,
    required=True,
    help="Valeur exacte de 'orientation_compteur' (ex. 'N-S').",
)
@click.option(
    "--range-start",
    type=float,
    default=0.0,
    show_default=True,
    help="Début de la tranche temporelle en pourcentage [0..100].",
)
@click.option(
    "--range-end",
    type=float,
    default=100.0,
    show_default=True,
    help="Fin de la tranche temporelle en pourcentage [0..100].",
)
@click.option(
    "--timestamp-col",
    type=str,
    default="date_et_heure_de_comptage",
    show_default=True,
    help="Colonne timestamp ISO8601 dans le CSV brut.",
)
@click.option(
    "--sub-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Sous-dossier cible ASCII sous data/interim. Par défaut dérivé de site+orientation.",
)
@click.option(
    "--interim-name",
    type=str,
    default="initial.csv",
    show_default=True,
    help="Nom du fichier intermédiaire dans data/interim/<sub-dir>.",
)
def main(
    raw_path: str,
    site: str,
    orientation: str,
    range_start: float,
    range_end: float,
    timestamp_col: str,
    sub_dir: Optional[str],
    interim_name: str,
) -> None:
    """
    Extrait un compteur du CSV brut et écrit un slice intermédiaire.
    Pousse des métriques batch (durée, volume) vers Pushgateway.
    """
    # Labels de grouping pour Prometheus/Pushgateway
    labels = {
        "dag": os.getenv("AIRFLOW_CTX_DAG_ID", "unknown_dag"),
        "task": os.getenv("AIRFLOW_CTX_TASK_ID", "etl.ingest"),
        "run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "local"),
        "site": slugify_ascii(site),
        "orientation": slugify_ascii(orientation),
    }

    # Encapsule tout dans le context manager -> pousse metrics en fin (success/error)
    with track_pipeline_step("ingest", labels) as m:
        # Validation paramètres
        if not (0.0 <= range_start <= 100.0 and 0.0 <= range_end <= 100.0):
            raise click.BadParameter(
                f"range-start/end doivent être dans [0,100]. Reçu ({range_start}, {range_end})."
            )
        if range_start > range_end:
            raise click.BadParameter(
                f"range-start doit être <= range-end. Reçu ({range_start}, {range_end})."
            )

        # Lecture
        try:
            logger.info("Chargement du CSV brut [%s] ...", raw_path)
            df = pd.read_csv(raw_path, index_col=0)
        except Exception as exc:
            logger.exception("Lecture CSV échouée: %s", exc)
            # Le context manager marquera 'error' et poussera les métriques
            raise click.ClickException(f"Echec lecture CSV: {exc}")

        # Colonnes requises
        key_cols = ["nom_du_site_de_comptage", "orientation_compteur"]
        missing = [c for c in key_cols + [timestamp_col] if c not in df.columns]
        if missing:
            raise click.ClickException(f"Colonnes manquantes: {missing}")

        # Filtre compteur
        grp = df.groupby(key_cols)
        key = (site, orientation)
        if key not in grp.groups:
            raise click.ClickException(f"Compteur introuvable: {key}")

        df_counter = grp.get_group(key).copy()
        logger.info("Compteur [%s | %s] -> %d lignes.", site, orientation, len(df_counter))

        # Tri chrono
        try:
            df_counter[timestamp_col] = pd.to_datetime(
                df_counter[timestamp_col], format="%Y-%m-%dT%H:%M:%S%z", utc=True
            )
        except Exception:
            df_counter[timestamp_col] = pd.to_datetime(df_counter[timestamp_col], utc=True)

        df_counter = df_counter.sort_values(timestamp_col).reset_index(drop=True)

        # Slice en pourcentage
        df_counter = apply_percent_range_selection(df_counter, (range_start, range_end))
        if df_counter.empty:
            raise click.ClickException("La tranche sélectionnée est vide.")

        # Sortie
        if sub_dir is None:
            sub_dir = slugify_ascii(f"{site}_{orientation}")
        out_dir = os.path.join("data", "interim", sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, interim_name)

        df_counter.to_csv(out_path, index=True)
        logger.info("Slice écrit -> [%s] (%d lignes).", out_path, len(df_counter))

        # Manifest optionnel
        man = os.getenv("MANIFEST_INGEST")
        if man:
            try:
                write_manifest(
                    man,
                    {
                        "inputs": {
                            "raw_path": str(Path(raw_path).resolve()),
                            "site": site,
                            "orientation": orientation,
                            "range": [range_start, range_end],
                            "timestamp_col": timestamp_col,
                        },
                        "outputs": {"interim_path": str(out_path)},
                        "run": {"run_id": os.getenv("RUN_ID"), "sub_dir": sub_dir},
                    },
                )
                logger.info("Manifest d'ingest écrit -> [%s].", man)
            except Exception as exc:
                logger.warning("Ecriture du manifest échouée [%s]: %s", man, exc)

        # Volume traité pour la métrique
        m["records"] = int(len(df_counter))

    logger.info("Ingestion terminée avec succès.")


if __name__ == "__main__":
    try:
        main()
    except click.ClickException as e:
        # click gère le message; on force un code retour 1
        logger.error(str(e))
        sys.exit(1)

