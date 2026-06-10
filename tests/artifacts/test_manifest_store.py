"""Unit tests for artifact manifest store helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.artifacts.checksums import compute_sha256
from src.artifacts.exceptions import (
    ArtifactChecksumMismatchError,
    ArtifactManifestNotFoundError,
    ArtifactManifestValidationError,
    ArtifactPayloadNotFoundError,
)
from src.artifacts.manifest_store import (
    promote_manifest,
    read_current_manifest,
    read_manifest,
    verify_local_payload,
    write_manifest,
)
from src.artifacts.schemas import ArtifactManifest, ArtifactStatus


class TestArtifactManifestStore:
    """Unit tests for filesystem-backed manifest store helpers."""

    @pytest.fixture
    def artifact_payload(self, tmp_path: Path) -> Path:
        artifact_file = tmp_path / "data/final/counter-1/predictions.csv"
        artifact_file.parent.mkdir(parents=True)
        artifact_file.write_text("date,y_pred\n2026-06-06,42\n", encoding="utf-8")
        return artifact_file

    @pytest.fixture
    def valid_manifest(self, tmp_path: Path, artifact_payload: Path) -> dict:
        local_path = artifact_payload.relative_to(tmp_path).as_posix()
        return {
            "schema_version": "1.0",
            "artifact_type": "predictions",
            "status": "validated",
            "run_id": "run-001",
            "counter_id": "counter-1",
            "created_at": "2026-06-06T14:00:00Z",
            "producer": {"service": "ml-models", "image": "ml-models:test"},
            "source": {"raw_file_name": "bike-counts.csv"},
            "storage": {
                "primary_backend": "local",
                "local_path": local_path,
                "checksum_sha256": compute_sha256(artifact_payload),
            },
        }

    def test_write_manifest_writes_counter_scoped_manifest(
        self,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest_root = tmp_path / "artifacts/manifests"

        manifest_path = write_manifest(valid_manifest, manifest_root)

        assert manifest_path == (
            manifest_root / "predictions" / "counter-1" / "run-001" / "manifest.json"
        )
        assert manifest_path.is_file()
        assert read_manifest(manifest_path).run_id == "run-001"

    def test_promote_manifest_writes_current_manifest(
        self,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest_root = tmp_path / "artifacts/manifests"

        current_path = promote_manifest(
            valid_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )

        assert current_path == manifest_root / "predictions" / "counter-1" / "current.json"
        current_manifest = read_current_manifest(
            manifest_root=manifest_root,
            artifact_type="predictions",
            counter_id="counter-1",
        )
        assert isinstance(current_manifest, ArtifactManifest)
        assert current_manifest.run_id == "run-001"
        assert current_manifest.status == ArtifactStatus.VALIDATED

    def test_write_manifest_rejects_invalid_manifest(
        self,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        invalid_manifest = deepcopy(valid_manifest)
        invalid_manifest.pop("run_id")

        with pytest.raises(ArtifactManifestValidationError, match="run_id"):
            write_manifest(invalid_manifest, tmp_path / "manifests")

    def test_read_current_manifest_missing_file_raises_explicit_error(
        self,
        tmp_path: Path,
    ) -> None:
        manifest_root = tmp_path / "artifacts/manifests"

        with pytest.raises(ArtifactManifestNotFoundError, match="does not exist"):
            read_current_manifest(
                manifest_root=manifest_root,
                artifact_type="predictions",
                counter_id="counter-1",
            )

    def test_read_manifest_invalid_json_raises_validation_error(
        self,
        tmp_path: Path,
    ) -> None:
        manifest_path = tmp_path / "current.json"
        manifest_path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(ArtifactManifestValidationError, match="not valid JSON"):
            read_manifest(manifest_path)

    def test_promote_manifest_rejects_missing_local_payload(
        self,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest = deepcopy(valid_manifest)
        manifest["storage"]["local_path"] = "data/final/counter-1/missing.csv"

        with pytest.raises(ArtifactPayloadNotFoundError, match="does not exist"):
            promote_manifest(
                manifest,
                manifest_root=tmp_path / "manifests",
                repository_root=tmp_path,
            )

    def test_verify_local_payload_rejects_checksum_mismatch(
        self,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest = deepcopy(valid_manifest)
        manifest["storage"]["checksum_sha256"] = "b" * 64
        validated_manifest = ArtifactManifest.model_validate(manifest)

        with pytest.raises(ArtifactChecksumMismatchError, match="mismatch"):
            verify_local_payload(validated_manifest, repository_root=tmp_path)

    def test_promote_manifest_rejects_non_promotable_status(
        self,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest = deepcopy(valid_manifest)
        manifest["status"] = "served"

        with pytest.raises(ArtifactManifestValidationError, match="not eligible"):
            promote_manifest(
                manifest,
                manifest_root=tmp_path / "manifests",
                repository_root=tmp_path,
            )
