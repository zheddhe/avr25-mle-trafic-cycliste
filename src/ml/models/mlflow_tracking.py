"""MLflow helpers for model training runs."""

from __future__ import annotations

import importlib
import os
from typing import Any

from src.common.env import get_env
from src.common.logger import get_logger

LOGGER = get_logger(__name__)


def _get_mlflow() -> Any:
    """Import MLflow lazily when tracking is used."""

    return importlib.import_module("mlflow")


def _get_mlflow_sklearn() -> Any:
    """Import the MLflow sklearn flavor lazily."""

    return importlib.import_module("mlflow.sklearn")


def configure_mlflow_from_env(explicit_uri: str | None = None) -> None:
    """Configure MLflow tracking from CLI or process environment."""

    uri = explicit_uri or get_env("MLFLOW_TRACKING_URI")
    if uri:
        _get_mlflow().set_tracking_uri(uri)
        LOGGER.info("MLflow tracking URI configured: %s", uri)
        return

    LOGGER.debug("MLflow tracking URI not set, using default local tracking.")


def is_model_registry_enabled(explicit_uri: str | None = None) -> bool:
    """Return whether MLflow registry operations should be attempted."""

    return bool(explicit_uri or get_env("MLFLOW_TRACKING_URI"))


def start_run(
    experiment_name: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
):
    """Start an MLflow run under the configured experiment."""

    mlflow = _get_mlflow()
    mlflow.set_experiment(experiment_name)
    ctx = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return ctx


def _log_params_flat(params: dict[str, Any] | None) -> None:
    """Log nested parameter dictionaries with flattened keys."""

    if not params:
        return

    mlflow = _get_mlflow()
    flat = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}.{nested_key}"] = nested_value
        else:
            flat[key] = value
    for key, value in flat.items():
        mlflow.log_param(key, value)


def log_report_content(
    report: dict[str, Any],
    target_col: str,
) -> None:
    """Log metrics, dataset shapes, hyperparameters, and model tags."""

    mlflow = _get_mlflow()
    metrics = report.get("metrics", {})
    for split, split_metrics in metrics.items():
        for key, value in split_metrics.items():
            mlflow.log_metric(f"{split}.{key}", float(value))

    x_train = report.get("X_train")
    x_test = report.get("X_test")
    y_train = report.get("y_train")
    y_test = report.get("y_test")
    if x_train is not None:
        mlflow.log_param("shape.X_train", f"{x_train.shape}")
    if x_test is not None:
        mlflow.log_param("shape.X_test", f"{x_test.shape}")
    if y_train is not None:
        mlflow.log_param("shape.y_train", f"{y_train.shape}")
    if y_test is not None:
        mlflow.log_param("shape.y_test", f"{y_test.shape}")

    _log_params_flat(report.get("params"))

    mlflow.set_tag("model.flavor", "sklearn-pipeline+xgboost")
    mlflow.set_tag("target", target_col)


def log_model_with_signature(
    pipe_model,
    sample_input_df,
    model_artifact_name: str = "model",
    registered_name: str | None = None,
    registry_enabled: bool | None = None,
) -> None:
    """Log the sklearn pipeline and optionally register it in MLflow."""

    from mlflow.exceptions import MlflowException
    from mlflow.models.signature import infer_signature

    mlflow_sklearn = _get_mlflow_sklearn()
    df = sample_input_df.copy()
    try:
        for col in sample_input_df.select_dtypes(include="int").columns:
            df[col] = df[col].astype("float64")
        signature = infer_signature(df, pipe_model.predict(df))
    except Exception:  # pragma: no cover - defensive MLflow safety net.
        LOGGER.exception("MLflow signature inference failed")
        signature = None

    effective_registry_enabled = (
        is_model_registry_enabled()
        if registry_enabled is None
        else registry_enabled
    )
    registered_model_name = registered_name if effective_registry_enabled else None

    try:
        mlflow_sklearn.log_model(
            sk_model=pipe_model,
            name=model_artifact_name,
            signature=signature,  # type: ignore[arg-type]
            registered_model_name=registered_model_name,
        )
    except MlflowException as exc:
        LOGGER.warning(
            "Model registry unavailable, falling back to artifact-only log: %s",
            exc,
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
    """Log local training outputs as MLflow artifacts when present."""

    mlflow = _get_mlflow()
    final_dir = os.path.join(final_rel_dir, save_subdir)
    if os.path.isdir(final_dir):
        mlflow.log_artifacts(final_dir, artifact_path="data_final")

    models_dir = os.path.join(models_rel_dir, save_subdir)
    if os.path.isdir(models_dir):
        mlflow.log_artifacts(models_dir, artifact_path="models_dir")

    if os.path.isdir(logs_rel_dir):
        mlflow.log_artifacts(logs_rel_dir, artifact_path="logs_ml")
