from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from src.artifacts.manifest_store import (
    promote_manifest,
    read_current_manifest,
    read_manifest,
    write_manifest,
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


class TestManifestStore:
    def test_write_manifest_creates_run_scoped_manifest(
        self,
        tmp_path: Path,
    ) -> None:
        payload_path = _write_payload(tmp_path, "payload.csv", "value\n1\n")
        manifest = _build_manifest(tmp_path, payload_path, "counter-a", "run-a")

        manifest_path = write_manifest(manifest, tmp_path / "manifests")
        loaded_manifest = read_manifest(manifest_path)

        assert manifest_path == (
            tmp_path
            / "manifests"
            / "predictions"
            / "counter-a"
            / "run-a"
            / "manifest.json"
        )
        assert loaded_manifest.counter_id == "counter-a"
        assert loaded_manifest.run_id == "run-a"

    def test_promote_manifest_replaces_current_manifest(
        self,
        tmp_path: Path,
    ) -> None:
        manifest_root = tmp_path / "manifests"
        first_payload = _write_payload(tmp_path, "first.csv", "value\n1\n")
        second_payload = _write_payload(tmp_path, "second.csv", "value\n2\n")
        first_manifest = _build_manifest(tmp_path, first_payload, "counter-a", "run-a")
        second_manifest = _build_manifest(tmp_path, second_payload, "counter-a", "run-b")

        promote_manifest(
            first_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        promote_manifest(
            second_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        loaded_manifest = read_current_manifest(manifest_root, "predictions", "counter-a")

        assert loaded_manifest.run_id == "run-b"
        assert loaded_manifest.storage.local_path == "second.csv"

    def test_repeated_promotion_is_idempotent(self, tmp_path: Path) -> None:
        manifest_root = tmp_path / "manifests"
        payload_path = _write_payload(tmp_path, "payload.csv", "value\n1\n")
        manifest = _build_manifest(tmp_path, payload_path, "counter-a", "run-a")

        first_path = promote_manifest(
            manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        second_path = promote_manifest(
            manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        loaded_manifest = read_current_manifest(manifest_root, "predictions", "counter-a")

        assert second_path == first_path
        assert loaded_manifest.model_dump() == manifest.model_dump()

    def test_different_counters_use_independent_current_manifests(
        self,
        tmp_path: Path,
    ) -> None:
        manifest_root = tmp_path / "manifests"
        first_payload = _write_payload(tmp_path, "first.csv", "value\n1\n")
        second_payload = _write_payload(tmp_path, "second.csv", "value\n2\n")
        first_manifest = _build_manifest(tmp_path, first_payload, "counter-a", "run-a")
        second_manifest = _build_manifest(tmp_path, second_payload, "counter-b", "run-b")

        promote_manifest(
            first_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        promote_manifest(
            second_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )

        assert read_current_manifest(manifest_root, "predictions", "counter-a").run_id == "run-a"
        assert read_current_manifest(manifest_root, "predictions", "counter-b").run_id == "run-b"


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
            checksum_sha256=_sha256(payload_path),
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
