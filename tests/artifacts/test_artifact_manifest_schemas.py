"""Unit tests for artifact manifest schemas."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from artifacts.schemas import ArtifactManifest, StorageBackend

VALID_CHECKSUM = "a" * 64


class TestArtifactManifest:
    """Unit tests for ArtifactManifest."""

    @pytest.fixture
    def valid_local_manifest(self) -> dict:
        return {
            "schema_version": "1.0",
            "artifact_type": "predictions",
            "status": "promoted",
            "run_id": "2026-06-06T140000Z-sebastopol-ns",
            "counter_id": "Sebastopol_N-S_airflow",
            "created_at": "2026-06-06T14:00:00Z",
            "producer": {
                "service": "ml-models-prod",
                "image": "ml-models:prod",
            },
            "source": {
                "raw_file_name": "bike-counts.csv",
                "dataset_version": "local-dev-dvc",
                "model_version": "mlflow-run-20260606",
            },
            "storage": {
                "primary_backend": "local",
                "local_path": (
                    "docker/prod/runtime/artifacts/data/final/"
                    "Sebastopol_N-S_airflow/"
                    "2026-06-06T140000Z-sebastopol-ns/predictions.parquet"
                ),
                "checksum_sha256": VALID_CHECKSUM,
            },
        }

    def test_valid_local_manifest(self, valid_local_manifest):
        manifest = ArtifactManifest.model_validate(valid_local_manifest)

        assert manifest.storage.primary_backend == StorageBackend.LOCAL
        assert manifest.storage.local_path.endswith("predictions.parquet")
        assert manifest.storage.object_uri is None

    def test_valid_hybrid_manifest(self, valid_local_manifest):
        payload = deepcopy(valid_local_manifest)
        payload["storage"]["object_uri"] = (
            "s3://mlflow/artifacts/predictions/"
            "Sebastopol_N-S_airflow/"
            "2026-06-06T140000Z-sebastopol-ns/predictions.parquet"
        )

        manifest = ArtifactManifest.model_validate(payload)

        assert manifest.storage.primary_backend == StorageBackend.LOCAL
        assert manifest.storage.object_uri.startswith("s3://mlflow/")

    def test_missing_required_field_raises_validation_error(
        self,
        valid_local_manifest,
    ):
        payload = deepcopy(valid_local_manifest)
        payload.pop("run_id")

        with pytest.raises(ValidationError, match="run_id"):
            ArtifactManifest.model_validate(payload)

    def test_invalid_backend_raises_validation_error(self, valid_local_manifest):
        payload = deepcopy(valid_local_manifest)
        payload["storage"]["primary_backend"] = "filesystem"

        with pytest.raises(ValidationError, match="primary_backend"):
            ArtifactManifest.model_validate(payload)

    @pytest.mark.parametrize(
        ("field_name", "invalid_value", "message"),
        [
            (
                "local_path",
                "https://example.com/artifacts/predictions.parquet",
                "repository-relative",
            ),
            (
                "object_uri",
                "https://minio/artifacts/predictions.parquet",
                "s3://",
            ),
        ],
    )
    def test_invalid_uri_cases_raise_validation_error(
        self,
        valid_local_manifest,
        field_name,
        invalid_value,
        message,
    ):
        payload = deepcopy(valid_local_manifest)
        payload["storage"][field_name] = invalid_value

        with pytest.raises(ValidationError, match=message):
            ArtifactManifest.model_validate(payload)
