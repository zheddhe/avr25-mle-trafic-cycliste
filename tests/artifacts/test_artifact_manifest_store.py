"""Unit tests for artifact manifest store helpers."""

from __future__ import annotations

from copy import deepcopy

import pytest

from artifacts.checksums import compute_sha256
from artifacts.exceptions import (
    ArtifactChecksumMismatchError,
    ArtifactManifestNotFoundError,
    ArtifactManifestValidationError,
    ArtifactPayloadNotFoundError,
)
from artifacts.manifest_store import (
    promote_manifest,
    read_current_manifest,
    read_manifest,
    verify_local_payload,
    write_manifest,
)
from artifacts.schemas import ArtifactManifest, ArtifactStatus


class TestArtifactChecksums:
    """Unit tests for checksum helpers."""

    def test_compute_sha256_returns_expected_digest(self, tmp_path):
        payload_path = tmp_path / "payload.txt"
        payload_path.write_text("bike traffic\n", encoding="utf-8")

        checksum = compute_sha256(payload_path)

        assert checksum == (
            "1540cc2351515a9d24ab9e4c8cac0fae"
            "99a461dcfe8f9b4758ae3f39db30eed2"
        )

    def test_compute_sha256_missing_payload_raises_explicit_error(self, tmp_path):
        missing_path = tmp_path / "missing.csv"

        with pytest.raises(ArtifactPayloadNotFoundError, match="does not exist"):
            compute_sha256(missing_path)


class TestArtifactManifestStore:
    """Unit tests for filesystem-backed manifest store helpers."""

    @pytest.fixture
    def artifact_payload(self, tmp_path):
        payload_path = tmp_path / "docker/prod/runtime/artifacts/data/final"
        payload_path = payload_path / "Sebastopol_N-S_airflow" / "run-001"
        payload_path.mkdir(parents=True)
        artifact_file = payload_path / "predictions.csv"
        artifact_file.write_text("date,y_pred\n2026-06-06,42\n", encoding="utf-8")
        return artifact_file

    @pytest.fixture
    def valid_manifest(self, tmp_path, artifact_payload) -> dict:
        local_path = artifact_payload.relative_to(tmp_path).as_posix()
        return {
            "schema_version": "1.0",
            "artifact_type": "predictions",
            "status": "validated",
            "run_id": "run-001",
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
                "local_path": local_path,
                "checksum_sha256": compute_sha256(artifact_payload),
            },
        }

    def test_write_manifest_writes_run_scoped_manifest(
        self,
        tmp_path,
        valid_manifest,
    ):
        manifest_root = tmp_path / "docker/prod/runtime/artifacts/manifests"

        manifest_path = write_manifest(valid_manifest, manifest_root)

        assert manifest_path == (
            manifest_root / "runs" / "run-001" / "predictions-manifest.json"
        )
        assert manifest_path.is_file()
        assert read_manifest(manifest_path).run_id == "run-001"

    def test_write_manifest_rejects_invalid_manifest(self, tmp_path, valid_manifest):
        invalid_manifest = deepcopy(valid_manifest)
        invalid_manifest.pop("run_id")

        with pytest.raises(ArtifactManifestValidationError, match="run_id"):
            write_manifest(invalid_manifest, tmp_path / "manifests")

    def test_promote_manifest_writes_stable_current_manifest(
        self,
        tmp_path,
        valid_manifest,
    ):
        manifest_root = tmp_path / "docker/prod/runtime/artifacts/manifests"

        current_path = promote_manifest(
            valid_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )

        assert current_path == manifest_root / "current.json"
        assert current_path.is_file()
        current_manifest = read_current_manifest(manifest_root)
        assert isinstance(current_manifest, ArtifactManifest)
        assert current_manifest.status == ArtifactStatus.VALIDATED
        assert current_manifest.counter_id == "Sebastopol_N-S_airflow"

    def test_read_current_manifest_missing_file_raises_explicit_error(self, tmp_path):
        manifest_root = tmp_path / "docker/prod/runtime/artifacts/manifests"

        with pytest.raises(ArtifactManifestNotFoundError, match="does not exist"):
            read_current_manifest(manifest_root)

    def test_read_manifest_invalid_json_raises_validation_error(self, tmp_path):
        manifest_path = tmp_path / "current.json"
        manifest_path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(ArtifactManifestValidationError, match="not valid JSON"):
            read_manifest(manifest_path)

    def test_promote_manifest_rejects_missing_local_payload(
        self,
        tmp_path,
        valid_manifest,
    ):
        manifest = deepcopy(valid_manifest)
        manifest["storage"]["local_path"] = "docker/prod/runtime/missing.csv"

        with pytest.raises(ArtifactPayloadNotFoundError, match="does not exist"):
            promote_manifest(
                manifest,
                manifest_root=tmp_path / "manifests",
                repository_root=tmp_path,
            )

    def test_verify_local_payload_rejects_checksum_mismatch(
        self,
        tmp_path,
        valid_manifest,
    ):
        manifest = deepcopy(valid_manifest)
        manifest["storage"]["checksum_sha256"] = "b" * 64
        validated_manifest = ArtifactManifest.model_validate(manifest)

        with pytest.raises(ArtifactChecksumMismatchError, match="mismatch"):
            verify_local_payload(validated_manifest, repository_root=tmp_path)

    def test_promote_manifest_rejects_non_promotable_status(
        self,
        tmp_path,
        valid_manifest,
    ):
        manifest = deepcopy(valid_manifest)
        manifest["status"] = "served"

        with pytest.raises(ArtifactManifestValidationError, match="not eligible"):
            promote_manifest(
                manifest,
                manifest_root=tmp_path / "manifests",
                repository_root=tmp_path,
            )
