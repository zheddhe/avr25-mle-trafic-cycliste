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
    CURRENT_MANIFEST_NAME,
    PROMOTION_LOCK_NAME,
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
            manifest_root
            / "predictions"
            / "counter-1"
            / "run-001"
            / "manifest.json"
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

        assert current_path == (
            manifest_root / "predictions" / "counter-1" / "current.json"
        )
        current_manifest = read_current_manifest(
            manifest_root=manifest_root,
            artifact_type="predictions",
            counter_id="counter-1",
        )
        assert isinstance(current_manifest, ArtifactManifest)
        assert current_manifest.run_id == "run-001"
        assert current_manifest.status == ArtifactStatus.VALIDATED

    def test_promote_manifest_replaces_current_manifest(
        self,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest_root = tmp_path / "artifacts/manifests"
        next_payload = _write_artifact_payload(
            tmp_path,
            "data/final/counter-1/predictions-next.csv",
            "date,y_pred\n2026-06-07,43\n",
        )
        next_manifest = _replace_manifest_payload(
            valid_manifest,
            tmp_path,
            next_payload,
            run_id="run-002",
        )

        promote_manifest(
            valid_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        promote_manifest(
            next_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        current_manifest = read_current_manifest(
            manifest_root=manifest_root,
            artifact_type="predictions",
            counter_id="counter-1",
        )

        assert current_manifest.run_id == "run-002"
        assert current_manifest.storage.local_path == (
            "data/final/counter-1/predictions-next.csv"
        )

    def test_repeated_promotion_is_idempotent(
        self,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest_root = tmp_path / "artifacts/manifests"

        first_path = promote_manifest(
            valid_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        second_path = promote_manifest(
            valid_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        current_manifest = read_current_manifest(
            manifest_root=manifest_root,
            artifact_type="predictions",
            counter_id="counter-1",
        )

        assert second_path == first_path
        assert current_manifest.run_id == "run-001"
        assert current_manifest.model_dump() == ArtifactManifest.model_validate(
            valid_manifest
        ).model_dump()

    def test_different_counters_use_independent_current_manifests(
        self,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest_root = tmp_path / "artifacts/manifests"
        counter_2_payload = _write_artifact_payload(
            tmp_path,
            "data/final/counter-2/predictions.csv",
            "date,y_pred\n2026-06-06,84\n",
        )
        counter_2_manifest = _replace_manifest_payload(
            valid_manifest,
            tmp_path,
            counter_2_payload,
            run_id="run-002",
            counter_id="counter-2",
        )

        promote_manifest(
            valid_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        promote_manifest(
            counter_2_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )

        first_current = read_current_manifest(
            manifest_root=manifest_root,
            artifact_type="predictions",
            counter_id="counter-1",
        )
        second_current = read_current_manifest(
            manifest_root=manifest_root,
            artifact_type="predictions",
            counter_id="counter-2",
        )
        assert first_current.run_id == "run-001"
        assert second_current.run_id == "run-002"

    def test_failed_current_write_preserves_previous_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest_root = tmp_path / "artifacts/manifests"
        next_payload = _write_artifact_payload(
            tmp_path,
            "data/final/counter-1/predictions-next.csv",
            "date,y_pred\n2026-06-07,43\n",
        )
        next_manifest = _replace_manifest_payload(
            valid_manifest,
            tmp_path,
            next_payload,
            run_id="run-002",
        )
        promote_manifest(
            valid_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )

        import src.artifacts.manifest_store as manifest_store

        real_write_json_atomic = manifest_store._write_json_atomic

        def fail_on_current_write(path: Path, content: str) -> None:
            if path.name == CURRENT_MANIFEST_NAME:
                raise OSError("simulated current write failure")
            real_write_json_atomic(path, content)

        monkeypatch.setattr(
            manifest_store,
            "_write_json_atomic",
            fail_on_current_write,
        )

        with pytest.raises(OSError, match="simulated current write failure"):
            promote_manifest(
                next_manifest,
                manifest_root=manifest_root,
                repository_root=tmp_path,
            )

        current_manifest = read_current_manifest(
            manifest_root=manifest_root,
            artifact_type="predictions",
            counter_id="counter-1",
        )
        assert current_manifest.run_id == "run-001"

    def test_same_counter_conflict_times_out_without_current_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        valid_manifest: dict,
    ) -> None:
        manifest_root = tmp_path / "artifacts/manifests"
        scope_path = manifest_root / "predictions" / "counter-1"
        scope_path.mkdir(parents=True)
        (scope_path / PROMOTION_LOCK_NAME).write_text("locked\n", encoding="utf-8")

        import src.artifacts.manifest_store as manifest_store

        monkeypatch.setattr(manifest_store, "DEFAULT_LOCK_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(manifest_store, "LOCK_POLL_INTERVAL_SECONDS", 0.001)

        with pytest.raises(ArtifactManifestValidationError, match="promotion lock"):
            promote_manifest(
                valid_manifest,
                manifest_root=manifest_root,
                repository_root=tmp_path,
            )

        assert not (scope_path / CURRENT_MANIFEST_NAME).exists()

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


def _write_artifact_payload(tmp_path: Path, relative_path: str, content: str) -> Path:
    payload_path = tmp_path / relative_path
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(content, encoding="utf-8")
    return payload_path


def _replace_manifest_payload(
    manifest: dict,
    repository_root: Path,
    payload_path: Path,
    *,
    run_id: str,
    counter_id: str | None = None,
) -> dict:
    updated_manifest = deepcopy(manifest)
    updated_manifest["run_id"] = run_id
    if counter_id is not None:
        updated_manifest["counter_id"] = counter_id
    updated_manifest["storage"]["local_path"] = (
        payload_path.relative_to(repository_root).as_posix()
    )
    updated_manifest["storage"]["checksum_sha256"] = compute_sha256(payload_path)
    return updated_manifest
