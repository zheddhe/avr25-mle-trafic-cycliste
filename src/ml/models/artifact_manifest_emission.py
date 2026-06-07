"""Prediction artifact manifest emission helpers."""

from __future__ import annotations

import os
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
from src.ml.artifact_manifest_emission import (
    clean_optional_value,
    resolve_counter_id,
    resolve_run_id,
    to_repository_relative_path,
)

DEFAULT_PRODUCER_SERVICE = "ml-models"


def emit_prediction_artifact_manifest(
    *,
    manifest_root: str | Path | None,
    prediction_path: str | Path,
    processed_path: str | Path,
    sub_dir: str,
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
    """Build and persist a prediction artifact manifest when configured."""

    if not manifest_root:
        return None

    manifest = build_prediction_artifact_manifest(
        prediction_path=prediction_path,
        processed_path=processed_path,
        sub_dir=sub_dir,
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


def build_prediction_artifact_manifest(
    *,
    prediction_path: str | Path,
    processed_path: str | Path,
    sub_dir: str,
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
    """Create a validated local prediction artifact manifest."""

    prediction_file = Path(prediction_path)
    local_path = to_repository_relative_path(
        path=prediction_file,
        repository_root=repository_root,
        path_label="prediction_path",
    )

    return ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        artifact_type=ArtifactType.PREDICTIONS,
        status=ArtifactStatus.VALIDATED,
        run_id=resolve_run_id(run_id),
        counter_id=resolve_counter_id(counter_id, sub_dir),
        created_at=datetime.now(UTC),
        producer=ArtifactProducer(
            service=(
                clean_optional_value(producer_service)
                or clean_optional_value(os.getenv("ARTIFACT_PRODUCER_SERVICE"))
                or DEFAULT_PRODUCER_SERVICE
            ),
            image=(
                clean_optional_value(producer_image)
                or clean_optional_value(os.getenv("ARTIFACT_PRODUCER_IMAGE"))
            ),
            version=(
                clean_optional_value(producer_version)
                or clean_optional_value(os.getenv("ARTIFACT_PRODUCER_VERSION"))
            ),
        ),
        source=ArtifactSource(
            raw_file_name=Path(processed_path).name,
            dataset_version=(
                clean_optional_value(dataset_version)
                or clean_optional_value(os.getenv("DATASET_VERSION"))
            ),
            model_version=(
                clean_optional_value(model_version)
                or clean_optional_value(os.getenv("MODEL_VERSION"))
            ),
        ),
        storage=ArtifactStorage(
            primary_backend=StorageBackend.LOCAL,
            local_path=local_path,
            object_uri=(
                clean_optional_value(object_uri)
                or clean_optional_value(os.getenv("ARTIFACT_OBJECT_URI"))
            ),
            checksum_sha256=compute_sha256(prediction_file),
        ),
    )
