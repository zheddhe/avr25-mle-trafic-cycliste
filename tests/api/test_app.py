# tests/api/test_app.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import ApiBusinessException, PredictionStore, create_app
from src.api.config import ApiSettings
from src.api.schemas import ArtifactSourceMetadata, CurrentArtifactMetadata
from src.api.serving import PredictionLoadResult, PredictionServingError
from src.artifacts.exceptions import ArtifactManifestNotFoundError


class TestApiPredictionStore:
    def test_refresh_replaces_predictions_and_artifacts(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        dataframe = pd.DataFrame({"y_pred": [1.0]})
        artifact = Mock(spec=CurrentArtifactMetadata)
        result = PredictionLoadResult(
            predictions={"counter-a": dataframe},
            artifacts={"counter-a": artifact},
        )
        store = PredictionStore(settings=settings)

        with patch("src.api.app.load_predictions_from_manifests") as loader:
            loader.return_value = result
            store.refresh()

        assert store.predictions["counter-a"] is dataframe
        assert store.artifacts["counter-a"] is artifact
        loader.assert_called_once_with(settings)

    def test_require_predictions_raises_when_store_is_empty(
        self,
        tmp_path: Path,
    ) -> None:
        store = PredictionStore(settings=_settings(tmp_path))

        with pytest.raises(ApiBusinessException) as exc_info:
            store.require_predictions()

        assert exc_info.value.type == "PredictionsNotLoaded"

    def test_require_artifacts_raises_when_store_is_empty(
        self,
        tmp_path: Path,
    ) -> None:
        store = PredictionStore(settings=_settings(tmp_path))

        with pytest.raises(ApiBusinessException) as exc_info:
            store.require_artifacts()

        assert exc_info.value.type == "ArtifactsNotLoaded"


class TestApiAppFactory:
    def test_startup_tolerates_missing_manifest_for_service_boot(
        self,
        tmp_path: Path,
    ) -> None:
        app = create_app(settings=_settings(tmp_path))

        with patch("src.api.app.load_predictions_from_manifests") as loader:
            loader.side_effect = ArtifactManifestNotFoundError("missing manifest")
            with TestClient(app) as client:
                response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_counters_returns_business_error_when_store_is_empty(
        self,
        tmp_path: Path,
    ) -> None:
        client = TestClient(create_app(settings=_settings(tmp_path)))

        response = client.get("/counters", auth=("user1", "user1"))

        assert response.status_code == 418
        assert response.json()["type"] == "PredictionsNotLoaded"

    def test_artifacts_returns_business_error_when_store_is_empty(
        self,
        tmp_path: Path,
    ) -> None:
        client = TestClient(create_app(settings=_settings(tmp_path)))

        response = client.get("/artifacts/current", auth=("user1", "user1"))

        assert response.status_code == 418
        assert response.json()["type"] == "ArtifactsNotLoaded"

    def test_predictions_returns_business_error_for_unavailable_counter(
        self,
        tmp_path: Path,
    ) -> None:
        app = create_app(settings=_settings(tmp_path))
        app.state.prediction_store.predictions = {
            "counter-a": _prediction_dataframe(),
        }
        client = TestClient(app)

        response = client.get(
            "/predictions/counter-b",
            auth=("user1", "user1"),
        )

        assert response.status_code == 418
        assert response.json()["type"] == "CounterUnavailable"
        assert "counter-a" in response.json()["message"]

    def test_current_artifact_returns_business_error_for_missing_counter(
        self,
        tmp_path: Path,
    ) -> None:
        app = create_app(settings=_settings(tmp_path))
        app.state.prediction_store.artifacts = {"counter-a": _artifact_metadata()}
        client = TestClient(app)

        response = client.get(
            "/artifacts/current/counter-b",
            auth=("user1", "user1"),
        )

        assert response.status_code == 418
        assert response.json()["type"] == "ArtifactUnavailable"
        assert "counter-a" in response.json()["message"]

    def test_admin_refresh_maps_loader_error_to_business_error(
        self,
        tmp_path: Path,
    ) -> None:
        app = create_app(settings=_settings(tmp_path))

        with patch("src.api.app.load_predictions_from_manifests") as loader:
            loader.side_effect = PredictionServingError("invalid payload")
            client = TestClient(app)
            response = client.post("/admin/refresh", auth=("admin1", "admin1"))

        assert response.status_code == 418
        assert response.json()["type"] == "PredictionServingError"
        assert response.json()["message"] == "invalid payload"


def _settings(tmp_path: Path) -> ApiSettings:
    return ApiSettings(
        manifest_root=tmp_path / "manifests",
        repository_root=tmp_path,
        counter_ids=(),
    )


def _prediction_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_et_heure_de_comptage_local": ["2026-01-01T01:00:00+01:00"],
            "date_et_heure_de_comptage_utc": ["2026-01-01T00:00:00+00:00"],
            "y_true": [10],
            "y_pred": [11.0],
            "forecast_mode": [False],
        }
    )


def _artifact_metadata() -> CurrentArtifactMetadata:
    return CurrentArtifactMetadata(
        counter_id="counter-a",
        run_id="run-a",
        artifact_type="predictions",
        status="promoted",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        producer_service="ml-models-prod",
        source=ArtifactSourceMetadata(dataset_version="test"),
        primary_backend="local",
        local_path="data/final/counter-a/y_full.csv",
        checksum_sha256="a" * 64,
    )
