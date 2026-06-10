from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.artifacts.exceptions import ArtifactManifestValidationError
from src.artifacts.manifest_store import (
    CURRENT_MANIFEST_NAME,
    PROMOTION_LOCK_NAME,
    promote_manifest,
    read_current_manifest,
)
from src.artifacts.schemas import (
    SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactProducer,
    ArtifactSource,
    ArtifactStatus,
    ArtifactStorage,
    ArtifactType,
    StorageBackend,
)


class TestManifestStoreFailures:
    def test_failed_current_write_preserves_previous_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        manifest_root = tmp_path / "manifests"
        first_manifest = _build_manifest(
            tmp_path,
            _write_payload(tmp_path, "first.csv", "value\n1\n"),
            "counter-a",
            "run-a",
        )
        second_manifest = _build_manifest(
            tmp_path,
            _write_payload(tmp_path, "second.csv", "value\n2\n"),
            "counter-a",
            "run-b",
        )
        promote_manifest(
            first_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )

        import src.artifacts.manifest_store as manifest_store

        real_write_json_atomic = manifest_store._write_json_atomic

        def fail_on_current_write(path: Path, content: str) -> None:
            if path.name == CURRENT_MANIFEST_NAME:
                raise OSError("simulated current write failure")
            real_write_json_atomic(path, content)

        monkeypatch.setattr(manifest_store, "_write_json_atomic", fail_on_current_write)

        with pytest.raises(OSError, match="simulated current write failure"):
            promote_manifest(
                second_manifest,
                manifest_root=manifest_root,
                repository_root=tmp_path,
            )

        loaded_manifest = read_current_manifest(
            manifest_root,
            "predictions",
            "counter-a",
        )
        assert loaded_manifest.run_id == "run-a"

    def test_same_counter_conflict_times_out_without_current_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        manifest_root = tmp_path / "manifests"
        manifest = _build_manifest(
            tmp_path,
            _write_payload(tmp_path, "payload.csv", "value\n1\n"),
            "counter-a",
            "run-a",
        )
        scope_path = manifest_root / "predictions" / "counter-a"
        scope_path.mkdir(parents=True)
        (scope_path / PROMOTION_LOCK_NAME).write_text("locked\n", encoding="utf-8")

        import src.artifacts.manifest_store as manifest_store

        monkeypatch.setattr(manifest_store, "DEFAULT_LOCK_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(manifest_store, "LOCK_POLL_INTERVAL_SECONDS", 0.001)

        with pytest.raises(ArtifactManifestValidationError, match="promotion lock"):
            promote_manifest(
                manifest,
                manifest_root=manifest_root,
                repository_root=tmp_path,
            )

        assert not (scope_path / CURRENT_MANIFEST_NAME).exists()


def _write_payload(tmp_path: Path, name: str, content: str) -> Path:
    payload_path = tmp_path / name
    payload_path.write_text(content, encoding="utf-8")
    return payload_path


def _build_manifest(
    repository_root: Path,
    payload_path: Path,
    counter_id: str,
    run_id: str,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        artifact_type=ArtifactType.PREDICTIONS,
        status=ArtifactStatus.VALIDATED,
        run_id=run_id,
        counter_id=counter_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        producer=ArtifactProducer(service="unit-test"),
        source=ArtifactSource(raw_file_name="raw.csv"),
        storage=ArtifactStorage(
            primary_backend=StorageBackend.LOCAL,
            local_path=payload_path.relative_to(repository_root).as_posix(),
            checksum_sha256=hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        ),
    )
