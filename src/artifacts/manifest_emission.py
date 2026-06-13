"""Artifact manifest emission helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.artifacts.checksums import compute_sha256
from src.artifacts.manifest_store import promote_manifest, write_manifest
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
from src.common.env import get_env

DEFAULT_PRODUCER_SERVICE = "artifact-producer"


def emit_artifact_manifest(
    *,
    manifest_root: str | Path | None,
    artifact_type: ArtifactType,
    payload_path: str | Path,
    source_file_name: str | Path,
    sub_dir: str,
    default_producer_service: str = DEFAULT_PRODUCER_SERVICE,
    repository_root: str | Path = ".",
    run_id: str | None = None,
    counter_id: str | None = None,
    dataset_version: str | None = None,
    model_version: str | None = None,
    producer_service: str | None = None,
    producer_image: str | None = None,
    producer_version: str | None = None,
    object_uri: str | None = None,
    promote: bool = True,
) -> ArtifactManifest | None:
    """Build and persist an artifact manifest when a root is configured."""

    if not manifest_root:
        return None

    manifest = build_artifact_manifest(
        artifact_type=artifact_type,
        payload_path=payload_path,
        source_file_name=source_file_name,
        sub_dir=sub_dir,
        default_producer_service=default_producer_service,
        repository_root=repository_root,
        run_id=run_id,
        counter_id=counter_id,
        dataset_version=dataset_version,
        model_version=model_version,
        producer_service=producer_service,
        producer_image=producer_image,
        producer_version=producer_version,
        object_uri=object_uri,
    )

    if promote:
        promote_manifest(
            manifest,
            manifest_root=manifest_root,
            repository_root=repository_root,
        )
    else:
        write_manifest(manifest, manifest_root=manifest_root)

    return manifest


def build_artifact_manifest(
    *,
    artifact_type: ArtifactType,
    payload_path: str | Path,
    source_file_name: str | Path,
    sub_dir: str,
    default_producer_service: str = DEFAULT_PRODUCER_SERVICE,
    repository_root: str | Path = ".",
    run_id: str | None = None,
    counter_id: str | None = None,
    dataset_version: str | None = None,
    model_version: str | None = None,
    producer_service: str | None = None,
    producer_image: str | None = None,
    producer_version: str | None = None,
    object_uri: str | None = None,
) -> ArtifactManifest:
    """Create a validated manifest for one artifact payload."""

    payload_file = Path(payload_path)
    local_path = to_repository_relative_path(
        path=payload_file,
        repository_root=repository_root,
        path_label="payload_path",
    )

    return ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        artifact_type=artifact_type,
        status=ArtifactStatus.VALIDATED,
        run_id=resolve_run_id(run_id),
        counter_id=resolve_counter_id(counter_id, sub_dir),
        created_at=datetime.now(UTC),
        producer=ArtifactProducer(
            service=resolve_producer_service(
                explicit_service=producer_service,
                default_service=default_producer_service,
            ),
            image=(
                clean_optional_value(producer_image)
                or clean_optional_value(get_env("ARTIFACT_PRODUCER_IMAGE"))
            ),
            version=(
                clean_optional_value(producer_version)
                or clean_optional_value(get_env("ARTIFACT_PRODUCER_VERSION"))
            ),
        ),
        source=ArtifactSource(
            raw_file_name=Path(source_file_name).name,
            dataset_version=(
                clean_optional_value(dataset_version)
                or clean_optional_value(get_env("DATASET_VERSION"))
            ),
            model_version=(
                clean_optional_value(model_version)
                or clean_optional_value(get_env("MODEL_VERSION"))
            ),
        ),
        storage=ArtifactStorage(
            primary_backend=StorageBackend.LOCAL,
            local_path=local_path,
            object_uri=(
                clean_optional_value(object_uri)
                or clean_optional_value(get_env("ARTIFACT_OBJECT_URI"))
            ),
            checksum_sha256=compute_sha256(payload_file),
        ),
    )


def resolve_producer_service(
    *,
    explicit_service: str | None,
    default_service: str,
) -> str:
    """Resolve producer service from explicit, environment, or default values."""

    return (
        clean_optional_value(explicit_service)
        or clean_optional_value(get_env("ARTIFACT_PRODUCER_SERVICE"))
        or default_service
    )


def resolve_run_id(run_id: str | None) -> str:
    """Resolve a stable run id from explicit or runtime context."""

    candidate = (
        clean_optional_value(run_id)
        or clean_optional_value(get_env("RUN_ID"))
        or clean_optional_value(get_env("AIRFLOW_CTX_DAG_RUN_ID"))
    )
    if candidate:
        return candidate

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_counter_id(counter_id: str | None, sub_dir: str) -> str:
    """Resolve a counter id from explicit or runtime context."""

    return (
        clean_optional_value(counter_id)
        or clean_optional_value(get_env("COUNTER_ID"))
        or sub_dir
    )


def to_repository_relative_path(
    *,
    path: Path,
    repository_root: str | Path,
    path_label: str,
) -> str:
    """Convert absolute paths inside a repository root to relative paths."""

    if not path.is_absolute():
        return path.as_posix()

    root = Path(repository_root).resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"{path_label} must be inside repository_root "
            "when it is provided as an absolute path",
        ) from error


def clean_optional_value(value: str | None) -> str | None:
    """Return stripped optional values and normalize blanks to None."""

    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None
