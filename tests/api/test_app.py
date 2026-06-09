# tests/api/test_app.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import ApiBusinessException, PredictionStore, create_app
from src.api.config import ApiSettings
from src.api.schemas import CurrentArtifactMetadata
from src.api.serving import PredictionLoadResult
from src.artifacts.exceptions import ArtifactManifestNotFoundError


class TestApiPredictionStore:
    def test_refresh_replaces_predictions_and_artifacts(self, tmp_path: Path) -> None:
        settings = ApiSettings(
            manifest_root=tmp_path / "manifests",
            repository_root=tmp_path,
            counter_ids=(),
        )
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

        assert store.predictions == {"counter-a": dataframe}
        assert store.artifacts == {"counter-a": artifact}
        loader.assert_called_once_with(settings)

    def test_require_predictions_raises_when_store_is_empty(
        self,
        tmp_path: Path,
    ) -> None:
        settings = ApiSettings(
            manifest_root=tmp_path / "manifests",
            repository_root=tmp_path,
            counter_ids=(),
        )
        store = PredictionStore(settings=settings)

        with pytest.raises(ApiBusinessException) as exc_info:
            store.require_predictions()

        assert exc_info.value.type == "PredictionsNotLoaded"

    def test_require_artifacts_raises_when_store_is_empty(
        self,
        tmp_path: Path,
    ) -> None:
        settings = ApiSettings(
            manifest_root=tmp_path / "manifests",
            repository_root=tmp_path,
            counter_ids=(),
        )
        store = PredictionStore(settings=settings)

        with pytest.raises(ApiBusinessException) as exc_info:
            store.require_artifacts()

        assert exc_info.value.type == "ArtifactsNotLoaded"


class TestApiAppFactory:
    def test_startup_tolerates_missing_manifest_for_service_boot(
        self,
        tmp_path: Path,
    ) -> None:
        settings = ApiSettings(
            manifest_root=tmp_path / "manifests",
            repository_root=tmp_path,
            counter_ids=(),
        )
        app = create_app(settings=settings)

        with patch("src.api.app.load_predictions_from_manifests") as loader:
            loader.side_effect = ArtifactManifestNotFoundError("missing manifest")
            with TestClient(app) as client:
                response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
