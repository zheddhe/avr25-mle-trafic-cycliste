"""Pydantic schemas for artifact handoff manifests.

The manifest is the shared contract between ML jobs, Airflow, future runners,
and the prediction API. It stays independent from runtime framework internals.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.artifacts.exceptions import ArtifactManifestValidationError

SCHEMA_VERSION = "1.0"
_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


class ArtifactType(StrEnum):
    """Supported artifact categories exchanged through manifests."""

    PREDICTIONS = "predictions"
    MODEL = "model"
    DATASET = "dataset"
    METRICS = "metrics"


class ArtifactStatus(StrEnum):
    """Lifecycle states documented in the artifact handoff strategy."""

    PRODUCED = "produced"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    SERVED = "served"
    ARCHIVED = "archived"


class StorageBackend(StrEnum):
    """Artifact storage backends supported by the manifest contract."""

    LOCAL = "local"
    OBJECT_STORAGE = "object_storage"


class StrictArtifactModel(BaseModel):
    """Base model forbidding undeclared manifest fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ArtifactProducer(StrictArtifactModel):
    """Component that produced the artifact payload and metadata."""

    service: str = Field(
        min_length=1,
        description="Service or job name that produced the artifact.",
    )
    image: str | None = Field(
        default=None,
        description="Container image reference used by the producer.",
    )
    version: str | None = Field(
        default=None,
        description="Optional producer code or package version.",
    )


class ArtifactSource(StrictArtifactModel):
    """Input lineage used to produce the artifact."""

    raw_file_name: str | None = Field(
        default=None,
        description="Raw input file name or external source label.",
    )
    dataset_version: str | None = Field(
        default=None,
        description="DVC revision, snapshot label, or runtime input version.",
    )
    model_version: str | None = Field(
        default=None,
        description="MLflow run, registry version, or local release label.",
    )

    @model_validator(mode="after")
    def validate_lineage_metadata(self) -> ArtifactSource:
        """Require at least one lineage reference in the source block."""

        values = (self.raw_file_name, self.dataset_version, self.model_version)
        if not any(values):
            raise ValueError("source must include at least one lineage reference")

        return self


class ArtifactStorage(StrictArtifactModel):
    """Storage references for local-only and hybrid artifact handoff."""

    primary_backend: StorageBackend = Field(
        description="Primary backend used by consumers for this artifact.",
    )
    local_path: str | None = Field(
        default=None,
        description="Repository-relative path to the local artifact payload.",
    )
    object_uri: str | None = Field(
        default=None,
        description="Optional S3-compatible object URI for hybrid handoff.",
    )
    checksum_sha256: str | None = Field(
        default=None,
        description="SHA-256 checksum of the canonical artifact payload.",
    )

    @field_validator("local_path")
    @classmethod
    def validate_local_path(cls, value: str | None) -> str | None:
        """Reject empty, absolute, parent-traversal, and URI-like paths."""

        if value is None:
            return None
        if not value:
            raise ValueError("local_path must not be empty")

        parsed = urlparse(value)
        if parsed.scheme:
            raise ValueError("local_path must be a repository-relative path")

        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("local_path must stay inside the repository")

        return value

    @field_validator("object_uri")
    @classmethod
    def validate_object_uri(cls, value: str | None) -> str | None:
        """Require S3-compatible object URIs without embedded credentials."""

        if value is None:
            return None
        if not value:
            raise ValueError("object_uri must not be empty")

        parsed = urlparse(value)
        if parsed.scheme != "s3":
            raise ValueError("object_uri must use the s3:// scheme")
        if not parsed.netloc or parsed.netloc.strip() != parsed.netloc:
            raise ValueError("object_uri must include a valid bucket name")
        if not parsed.path or parsed.path == "/":
            raise ValueError("object_uri must include an object key")
        if "@" in parsed.netloc:
            raise ValueError("object_uri must not embed credentials")

        return value

    @field_validator("checksum_sha256")
    @classmethod
    def validate_checksum_sha256(cls, value: str | None) -> str | None:
        """Require checksums to use the standard 64-character hex format."""

        if value is None:
            return None

        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("checksum_sha256 must be a 64-character hex digest")

        return value.lower()

    @model_validator(mode="after")
    def validate_backend_references(self) -> ArtifactStorage:
        """Ensure every backend has the references it needs."""

        if self.primary_backend == StorageBackend.LOCAL and self.local_path is None:
            raise ValueError("local_path is required for local storage")

        if (
            self.primary_backend == StorageBackend.OBJECT_STORAGE
            and self.object_uri is None
        ):
            raise ValueError("object_uri is required for object storage")

        if self.local_path is None and self.object_uri is None:
            raise ValueError("storage must include local_path or object_uri")

        return self


class ArtifactManifest(StrictArtifactModel):
    """Validated manifest describing one generated artifact."""

    schema_version: str = Field(
        description="Manifest schema version. The initial contract uses 1.0.",
    )
    artifact_type: ArtifactType = Field(
        description="Type of artifact referenced by the manifest.",
    )
    status: ArtifactStatus = Field(
        description="Lifecycle status of the referenced artifact.",
    )
    run_id: str = Field(
        min_length=1,
        description="Unique run identifier for the artifact production attempt.",
    )
    counter_id: str = Field(
        min_length=1,
        description="Project counter identifier associated with the artifact.",
    )
    created_at: datetime = Field(
        description="UTC timestamp for manifest creation.",
    )
    producer: ArtifactProducer = Field(
        description="Producer metadata and execution identity.",
    )
    source: ArtifactSource = Field(
        description="Input data and model lineage metadata.",
    )
    storage: ArtifactStorage = Field(
        description="Local and optional object-storage artifact references.",
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        """Keep the implementation pinned to the documented schema version."""

        if value != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        """Require timezone-aware creation timestamps."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")

        return value


def validate_artifact_manifest(payload: Mapping[str, Any]) -> ArtifactManifest:
    """Validate a raw payload and return an artifact manifest instance.

    Raises:
        ArtifactManifestValidationError: if Pydantic rejects the payload.
    """

    try:
        return ArtifactManifest.model_validate(payload)
    except ValueError as error:
        raise ArtifactManifestValidationError(str(error)) from error
