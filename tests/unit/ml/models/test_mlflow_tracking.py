"""Unit tests for MLflow tracking helpers."""

from __future__ import annotations

from src.ml.models.mlflow_tracking import is_model_registry_enabled


class TestMlflowTracking:
    """Unit tests for MLflow tracking configuration helpers."""

    def test_model_registry_is_enabled_with_explicit_uri(self) -> None:
        assert is_model_registry_enabled("http://mlflow:5000") is True

    def test_model_registry_is_disabled_without_tracking_uri(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        assert is_model_registry_enabled() is False
