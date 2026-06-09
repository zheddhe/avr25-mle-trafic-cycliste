# src/ml/models/mlflow_tracking.py
from __future__ import annotations

import importlib
import logging
import os
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature

logger = logging.getLogger(__name__)
mlflow_sklearn: Any = importlib.import_module("mlflow.sklearn")


def configure_mlflow_from_env(
    explicit_uri: str | None = None,
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


def is_model_registry_enabled(explicit_uri: str | None = None) -> bool:
    """
    Return whether model registry operations should be attempted.

    Local file-based MLflow runs remain artifact-only by default. Registry
    registration and alias promotion are enabled only when a tracking URI is
    explicitly provided through CLI or environment configuration.
    """

    return bool(explicit_uri or os.getenv("MLFLOW_TRACKING_URI"))


def start_run(
    experiment_name: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
):
    """
    Set experiment and start a run with optional tags.
    """
    mlflow.set_experiment(experiment_name)
    ctx = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return ctx


def _log_params_flat(params: dict[str, Any] | None) -> None:
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
    report: dict[str, Any],
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
    model_artifact_name: str = "model",
    registered_name: str | None = None,
    registry_enabled: bool | None = None,
) -> None:
    """
    Log the sklearn Pipeline with a signature inferred from a sample row.

    MLflow 3.x uses ``name`` for model artifact logging. Registry registration
    remains explicit and is disabled for local artifact-only runs by default.
    """
    df = sample_input_df.copy()
    try:
        for col in sample_input_df.select_dtypes(include="int").columns:
            df[col] = df[col].astype("float64")
        signature = infer_signature(df, pipe_model.predict(df))
    except Exception as exc:  # pragma: no cover (safety)
        logger.warning(f"Signature inference failed: {exc}")
        signature = None

    effective_registry_enabled = (
        is_model_registry_enabled()
        if registry_enabled is None
        else registry_enabled
    )
    reg_name = registered_name if effective_registry_enabled else None

    try:
        mlflow_sklearn.log_model(
            sk_model=pipe_model,
            name=model_artifact_name,
            signature=signature,  # type: ignore[arg-type]
            registered_model_name=reg_name,
        )
    except MlflowException as exc:
        logger.warning(
            f"Model registry not available, fallback to artifact-only: {exc}"
        )
        mlflow_sklearn.log_model(
            sk_model=pipe_model,
            name=model_artifact_name,
            signature=signature,  # type: ignore[arg-type]
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
