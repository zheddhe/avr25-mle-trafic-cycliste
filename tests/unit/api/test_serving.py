# tests/api/test_serving.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from src.api.config import ApiSettings
from src.api.serving import (
    PredictionCsvError,
    PredictionServingError,
    UnsupportedPredictionBackendError,
    discover_current_prediction_counter_ids,
    load_prediction_dataframe_from_manifest,
    load_predictions_from_manifests,
    read_prediction_csv,
)
from src.artifacts.checksums import compute_sha256
from src.artifacts.exceptions import (
    ArtifactChecksumMismatchError,
    ArtifactManifestNotFoundError,
)
from src.artifacts.schemas import ArtifactManifest

COUNTER_ID = "Sebastopol_N-S"
RUN_ID = "api-serving-run"


def _write_prediction_payload(repository_root: Path) -> Path:
    payload_path = repository_root / "data" / "final" / COUNTER_ID / "y_full.csv"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(
        {
            "date_et_heure_de_comptage_local": ["2025-09-23 08:00:00"],
            "date_et_heure_de_comptage_utc": ["2025-09-23 06:00:00"],
            "y_true": [123],
            "y_pred": [120.5],
            "forecast_mode": [False],
        }
    )
    dataframe.to_csv(payload_path)
    return payload_path


def _manifest_payload(
    *,
    repository_root: Path,
    payload_path: Path,
    primary_backend: str = "local",
    checksum_sha256: str | None = None,
    local_path: str | None = None,
    object_uri: str | None = None,
) -> dict:
    resolved_local_path = local_path
    if resolved_local_path is None:
        resolved_local_path = payload_path.relative_to(repository_root).as_posix()

    storage = {
        "primary_backend": primary_backend,
        "local_path": resolved_local_path,
        "checksum_sha256": checksum_sha256 or compute_sha256(payload_path),
    }
    if object_uri is not None:
        storage["object_uri"] = object_uri

    return {
        "schema_version": "1.0",
        "artifact_type": "predictions",
        "status": "promoted",
        "run_id": RUN_ID,
        "counter_id": COUNTER_ID,
        "created_at": "2026-06-06T14:00:00+00:00",
        "producer": {"service": "ml-models-prod"},
        "source": {"dataset_version": "test-dataset"},
        "storage": storage,
    }


def _write_current_manifest(
    *,
    manifest_root: Path,
    manifest: dict,
) -> Path:
    counter_id = manifest["counter_id"]
    current_path = manifest_root / "predictions" / counter_id / "current.json"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps(manifest), encoding="utf-8")
    return current_path


def _settings(repository_root: Path, manifest_root: Path) -> ApiSettings:
    return ApiSettings(
        manifest_root=manifest_root,
        repository_root=repository_root,
        counter_ids=(),
    )


class TestApiServing:
    def test_discover_current_prediction_counter_ids(self, tmp_path: Path) -> None:
        manifest_root = tmp_path / "artifacts" / "manifests"
        current_path = manifest_root / "predictions" / COUNTER_ID / "current.json"
        current_path.parent.mkdir(parents=True)
        current_path.write_text("{}", encoding="utf-8")

        counter_ids = discover_current_prediction_counter_ids(manifest_root)

        assert counter_ids == (COUNTER_ID,)

    def test_discover_current_prediction_counter_ids_returns_empty_tuple(
        self,
        tmp_path: Path,
    ) -> None:
        counter_ids = discover_current_prediction_counter_ids(tmp_path / "missing")

        assert counter_ids == ()

    def test_load_predictions_from_promoted_current_manifest(
        self,
        tmp_path: Path,
    ) -> None:
        repository_root = tmp_path / "repository"
        manifest_root = repository_root / "artifacts" / "manifests"
        payload_path = _write_prediction_payload(repository_root)
        manifest = _manifest_payload(
            repository_root=repository_root,
            payload_path=payload_path,
        )
        _write_current_manifest(manifest_root=manifest_root, manifest=manifest)

        result = load_predictions_from_manifests(
            _settings(repository_root, manifest_root)
        )

        assert sorted(result.predictions.keys()) == [COUNTER_ID]
        assert result.predictions[COUNTER_ID].shape == (1, 5)
        assert result.artifacts[COUNTER_ID].counter_id == COUNTER_ID

    def test_load_predictions_raises_when_no_current_manifest(
        self,
        tmp_path: Path,
    ) -> None:
        repository_root = tmp_path / "repository"
        manifest_root = repository_root / "artifacts" / "manifests"

        with pytest.raises(ArtifactManifestNotFoundError):
            load_predictions_from_manifests(_settings(repository_root, manifest_root))

    def test_load_prediction_dataframe_rejects_unsupported_backend(
        self,
        tmp_path: Path,
    ) -> None:
        repository_root = tmp_path / "repository"
        payload_path = _write_prediction_payload(repository_root)
        manifest = ArtifactManifest.model_validate(
            _manifest_payload(
                repository_root=repository_root,
                payload_path=payload_path,
                primary_backend="object_storage",
                object_uri="s3://bucket/predictions/y_full.csv",
            )
        )

        with pytest.raises(UnsupportedPredictionBackendError):
            load_prediction_dataframe_from_manifest(
                manifest=manifest,
                settings=_settings(repository_root, tmp_path / "manifests"),
            )

    def test_load_prediction_dataframe_rejects_missing_local_path(
        self,
        tmp_path: Path,
    ) -> None:
        repository_root = tmp_path / "repository"
        payload_path = _write_prediction_payload(repository_root)
        raw_manifest = _manifest_payload(
            repository_root=repository_root,
            payload_path=payload_path,
            primary_backend="object_storage",
            local_path=None,
            object_uri="s3://bucket/predictions/y_full.csv",
        )
        raw_manifest["storage"]["primary_backend"] = "local"
        raw_manifest["storage"].pop("local_path")

        with pytest.raises(ValueError):
            ArtifactManifest.model_validate(raw_manifest)

    def test_load_prediction_dataframe_rejects_checksum_mismatch(
        self,
        tmp_path: Path,
    ) -> None:
        repository_root = tmp_path / "repository"
        payload_path = _write_prediction_payload(repository_root)
        manifest = ArtifactManifest.model_validate(
            _manifest_payload(
                repository_root=repository_root,
                payload_path=payload_path,
                checksum_sha256="a" * 64,
            )
        )

        with pytest.raises(ArtifactChecksumMismatchError):
            load_prediction_dataframe_from_manifest(
                manifest=manifest,
                settings=_settings(repository_root, tmp_path / "manifests"),
            )

    def test_read_prediction_csv_rejects_missing_required_column(
        self,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "bad_predictions.csv"
        pd.DataFrame(
            {
                "date_et_heure_de_comptage_local": ["2025-09-23 08:00:00"],
                "date_et_heure_de_comptage_utc": ["2025-09-23 06:00:00"],
                "y_pred": [120.5],
                "forecast_mode": [False],
            }
        ).to_csv(csv_path)

        with pytest.raises(PredictionCsvError) as exc_info:
            read_prediction_csv(csv_path)

        assert "y_true" in str(exc_info.value)

    def test_read_prediction_csv_rejects_empty_payload(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty_predictions.csv"
        pd.DataFrame(
            columns=[
                "date_et_heure_de_comptage_local",
                "date_et_heure_de_comptage_utc",
                "y_true",
                "y_pred",
                "forecast_mode",
            ]
        ).to_csv(csv_path)

        with pytest.raises(PredictionCsvError) as exc_info:
            read_prediction_csv(csv_path)

        assert "empty" in str(exc_info.value)

    def test_load_prediction_dataframe_rejects_wrong_artifact_type(
        self,
        tmp_path: Path,
    ) -> None:
        repository_root = tmp_path / "repository"
        payload_path = _write_prediction_payload(repository_root)
        raw_manifest = _manifest_payload(
            repository_root=repository_root,
            payload_path=payload_path,
        )
        raw_manifest["artifact_type"] = "metrics"
        manifest = ArtifactManifest.model_validate(raw_manifest)

        with pytest.raises(PredictionServingError):
            load_prediction_dataframe_from_manifest(
                manifest=manifest,
                settings=_settings(repository_root, tmp_path / "manifests"),
            )
