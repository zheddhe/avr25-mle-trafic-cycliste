# src/ml/models/mlflow_tracking.py
from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


def configure_mlflow_from_env(
    explicit_uri: Optional[str] = None,
) -> None:
    """
    Configure MLflow tracking URI from arg or env.
    Respects MLFLOW_TRACKING_URI if set.
    """
    uri = explicit_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow tracking URI set to [{uri}]")
    else:
        logger.info("MLflow tracking URI not set (using default/local).")


def start_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """
    Set experiment and start a run with optional tags.
    """
    mlflow.set_experiment(experiment_name)
    ctx = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return ctx


def _log_params_flat(params: Optional[Dict[str, Any]]) -> None:
    """
    Log dict of hyperparams (if present), flattening as needed.
    """
    if not params:
        return
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    for k, v in flat.items():
        mlflow.log_param(k, v)


def log_report_content(
    report: Dict[str, Any],
    target_col: str,
) -> None:
    """
    Log metrics (train/test), basic shapes, table of prediction (forecasted or not).
    """
    # metrics
    metrics = report.get("metrics", {})
    for split, m in metrics.items():
        for k, v in m.items():
            mlflow.log_metric(f"{split}.{k}", float(v))

    # shapes
    xtr, xte = report.get("X_train"), report.get("X_test")
    ytr, yte = report.get("y_train"), report.get("y_test")
    if xtr is not None:
        mlflow.log_param("shape.X_train", f"{xtr.shape}")
    if xte is not None:
        mlflow.log_param("shape.X_test", f"{xte.shape}")
    if ytr is not None:
        mlflow.log_param("shape.y_train", f"{ytr.shape}")
    if yte is not None:
        mlflow.log_param("shape.y_test", f"{yte.shape}")

    # hyperparameters (if any from search)
    _log_params_flat(report.get("params"))

    # model flavor tag
    mlflow.set_tag("model.flavor", "sklearn-pipeline+xgboost")
    mlflow.set_tag("target", target_col)


def log_model_with_signature(
    pipe_model,
    sample_input_df,
    artifact_path: str = "model",
    registered_name: Optional[str] = None,
) -> None:
    """
    Log the sklearn Pipeline with signature inferred from a sample row.

    If the current MLflow tracking backend does not support a Model Registry
    (e.g. file:// tracking URI), we gracefully fall back to logging without
    `registered_model_name`.
    """
    df = sample_input_df.copy()
    try:
        for col in sample_input_df.select_dtypes(include="int").columns:
            df[col] = df[col].astype("float64")
        signature = infer_signature(df, pipe_model.predict(df))
    except Exception as exc:  # pragma: no cover (safety)
        logger.warning("Signature inference failed: %s", exc)
        signature = None

    # Heuristic: only try to register if a registry URI is configured
    reg_enabled = bool(os.getenv("MLFLOW_REGISTRY_URI"))
    reg_name = registered_name if reg_enabled else None

    try:
        mlflow.sklearn.log_model(
            sk_model=pipe_model,
            artifact_path=artifact_path,
            signature=signature,  # type: ignore
            registered_model_name=reg_name,
        )
    except MlflowException as exc:
        # FileStore or registry not available → retry without registration
        logger.warning(
            "Model registry not available, fallback to artifact-only: %s",
            exc,
        )
        mlflow.sklearn.log_model(
            sk_model=pipe_model,
            artifact_path=artifact_path,
            signature=signature,  # type: ignore
            registered_model_name=None,
        )


def log_local_artifacts(
    save_subdir: str,
    final_rel_dir: str = os.path.join("data", "final"),
    models_rel_dir: str = os.path.join("models"),
    logs_rel_dir: str = os.path.join("logs", "ml"),
) -> None:
    """
    Log all produced files/directories as MLflow artifacts.
    """
    # Data (predictions, splits, etc.)
    final_dir = os.path.join(final_rel_dir, save_subdir)
    if os.path.isdir(final_dir):
        mlflow.log_artifacts(final_dir, artifact_path="data_final")

    # Model/transformer/params/metrics pickles & json
    models_dir = os.path.join(models_rel_dir, save_subdir)
    if os.path.isdir(models_dir):
        mlflow.log_artifacts(models_dir, artifact_path="models_dir")

    # Logs (helpful for debugging)
    if os.path.isdir(logs_rel_dir):
        mlflow.log_artifacts(logs_rel_dir, artifact_path="logs_ml")
