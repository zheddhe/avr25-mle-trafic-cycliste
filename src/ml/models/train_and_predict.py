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
import pytz
import mlflow
from mlflow.tracking import MlflowClient

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
# CLI
# -------------------------------------------------------------------
@click.command()
@click.option(
    "--processed-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Chemin du CSV processed contenant *_utc et *_local.",
)
@click.option(
    "--sub-dir",
    type=str,
    default=None,
    help=("Sous-dossier cible sous data/final. Si absent, utilise "
          "data/final/<sub-dir-dérivé-du-processed-path>."),
)
@click.option(
    "--target-col",
    type=str,
    default="comptage_horaire",
    show_default=True,
    help="Colonne cible pour la régression.",
)
@click.option(
    "--ts-col-utc",
    type=str,
    default="date_et_heure_de_comptage_utc",
    show_default=True,
    help="Nom de la colonne timestamp UTC.",
)
@click.option(
    "--ts-col-local",
    type=str,
    default="date_et_heure_de_comptage_local",
    show_default=True,
    help="Nom de la colonne timestamp local.",
)
@click.option(
    "--ar",
    type=int,
    default=7,
    show_default=True,
    help="Nombre de retards AR.",
)
@click.option(
    "--mm",
    type=int,
    default=1,
    show_default=True,
    help="Nombre de moyennes mobiles.",
)
@click.option(
    "--roll",
    type=int,
    default=24,
    show_default=True,
    help="Fenêtre de base (heures) pour moyennes mobiles.",
)
@click.option(
    "--test-ratio",
    type=float,
    default=0.25,
    show_default=True,
    help="Part chronologique pour le split test.",
)
@click.option(
    "--grid-iter",
    type=int,
    default=0,
    show_default=True,
    help="Itérations de recherche bayésienne (0 pour désactiver).",
)
@click.option(
    "--mlflow-uri",
    type=str,
    default=None,
    help="MLflow tracking URI optionnelle (prioritaire sur MLFLOW_TRACKING_URI).",
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
    Entraîne un modèle séries temporelles, génère les prédictions complètes, journalise sous MLflow.
    Pousse des métriques batch (durée, volume) vers Pushgateway.
    """
    # Labels Prometheus (grouping_key)
    labels = {
        "dag": os.getenv("AIRFLOW_CTX_DAG_ID", "unknown_dag"),
        "task": os.getenv("AIRFLOW_CTX_TASK_ID", "etl.models"),
        "run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "local"),
    }

    with track_pipeline_step("models", labels) as m:
        if not (0.0 < test_ratio < 0.95):
            raise click.BadParameter(f"test-ratio doit être dans (0, 0.95). Reçu: {test_ratio}")
        if grid_iter < 0:
            raise click.BadParameter(f"grid-iter doit être >= 0. Reçu: {grid_iter}")

        # Configure MLflow (mlflow_uri CLI > env)
        configure_mlflow_from_env(explicit_uri=mlflow_uri)

        if sub_dir is None:
            sub_dir = os.path.basename(os.path.dirname(processed_path))

        # Données
        try:
            logger.info("Chargement du CSV processed [%s] ...", processed_path)
            df = pd.read_csv(processed_path, index_col=0)
        except Exception as exc:
            logger.exception("Échec de lecture du CSV processed: %s", exc)
            raise click.ClickException(f"Lecture CSV échouée: {exc}")

        for col in [ts_col_utc, ts_col_local]:
            if col not in df.columns:
                raise click.ClickException(f"Colonne requise manquante: {col}")

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
            df[ts_col_local] = pd.to_datetime(
                df[ts_col_local], utc=True
            ).dt.tz_convert(pytz.timezone("Europe/Paris"))

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
                logger.info("Manifest models écrit: %s", man)
            except Exception as exc:
                logger.warning("Échec d'écriture du manifest models [%s]: %s", man, exc)

        # Promotion automatique Model Registry
        try:
            model_name = f"{sub_dir}-model"
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.warning("Aucune version trouvée pour le modèle [%s].", model_name)
            else:
                latest = max(versions, key=lambda v: int(v.version))
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                try:
                    client.set_registered_model_alias(model_name, "prod", latest.version)
                except Exception as e:
                    logger.warning("Alias 'prod' non défini: %s", e)
                logger.info(
                    "Modèle [%s] version %s promu en Production.",
                    model_name,
                    latest.version,
                )
        except Exception as e:
            logger.warning("Échec de promotion du modèle: %s", e)

        # Volume traité pour la métrique: lignes de y_full
        records = 0
        try:
            if y_full_path and os.path.isfile(y_full_path):
                df_full = pd.read_csv(y_full_path, index_col=0)
                records = int(len(df_full))
            elif "y_test_pred" in report:
                records = int(len(report["y_test_pred"]))
        except Exception as e:
            logger.warning("Impossible de compter les enregistrements: %s", e)
        m["records"] = records  # <-- pousse via Pushgateway à la sortie du contexte

    logger.info("Training et forecasting terminés avec succès.")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except click.ClickException as e:
        logger.error(str(e))
        sys.exit(1)
