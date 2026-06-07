"""Ingest artifact manifest emission helpers."""

from __future__ import annotations

from pathlib import Path

from src.artifacts.manifest_emission import build_artifact_manifest, emit_artifact_manifest
from src.artifacts.schemas import ArtifactManifest, ArtifactType

DEFAULT_PRODUCER_SERVICE = "ml-ingest"


def emit_interim_dataset_artifact_manifest(
    *,
    manifest_root: str | Path | None,
    payload_path: str | Path,
    source_file_name: str | Path,
    sub_dir: str,
    repository_root: str | Path = ".",
    run_id: str | None = None,
    counter_id: str | None = None,
    dataset_version: str | None = None,
    producer_service: str | None = None,
    producer_image: str | None = None,
    producer_version: str | None = None,
    object_uri: str | None = None,
    promote: bool = True,
) -> ArtifactManifest | None:
    """Build and persist an interim dataset manifest when configured."""

    return emit_artifact_manifest(
        manifest_root=manifest_root,
        artifact_type=ArtifactType.INTERIM_DATASET,
        payload_path=payload_path,
        source_file_name=source_file_name,
        sub_dir=sub_dir,
        default_producer_service=DEFAULT_PRODUCER_SERVICE,
        repository_root=repository_root,
        run_id=run_id,
        counter_id=counter_id,
        dataset_version=dataset_version,
        producer_service=producer_service,
        producer_image=producer_image,
        producer_version=producer_version,
        object_uri=object_uri,
        promote=promote,
    )


def build_interim_dataset_artifact_manifest(
    *,
    payload_path: str | Path,
    source_file_name: str | Path,
    sub_dir: str,
    repository_root: str | Path = ".",
    run_id: str | None = None,
    counter_id: str | None = None,
    dataset_version: str | None = None,
    producer_service: str | None = None,
    producer_image: str | None = None,
    producer_version: str | None = None,
    object_uri: str | None = None,
) -> ArtifactManifest:
    """Create a validated interim dataset artifact manifest."""

    return build_artifact_manifest(
        artifact_type=ArtifactType.INTERIM_DATASET,
        payload_path=payload_path,
        source_file_name=source_file_name,
        sub_dir=sub_dir,
        default_producer_service=DEFAULT_PRODUCER_SERVICE,
        repository_root=repository_root,
        run_id=run_id,
        counter_id=counter_id,
        dataset_version=dataset_version,
        producer_service=producer_service,
        producer_image=producer_image,
        producer_version=producer_version,
        object_uri=object_uri,
    )
