"""Typed pipeline job request contracts.

These Pydantic models are the framework-neutral payloads exchanged between
Airflow, the future job-runner API, and typed ML workers. They describe business
inputs and artifact handoff locations without importing Airflow, FastAPI, Docker,
or concrete ML execution modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.artifacts.schemas import ArtifactType


def utc_now() -> datetime:
    """Return the current UTC timestamp for request defaults."""

    return datetime.now(UTC)


class PipelineJobType(StrEnum):
    """Supported ML pipeline job types."""

    INGEST = "ingest"
    FEATURES = "features"
    MODELS = "models"
    PIPELINE = "pipeline"


class StrictPipelineContract(BaseModel):
    """Base model for strict pipeline contracts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ArtifactManifestReference(StrictPipelineContract):
    """Reference to an artifact manifest emitted by a typed job."""

    artifact_type: ArtifactType = Field(
        description="Artifact type described by the referenced manifest.",
    )
    counter_id: str = Field(
        min_length=1,
        description="Counter identifier associated with the manifest.",
    )
    run_id: str = Field(
        min_length=1,
        description="Pipeline run identifier associated with the manifest.",
    )
    manifest_path: str = Field(
        min_length=1,
        description="Local or repository-relative path to the manifest file.",
    )
    current_path: str | None = Field(
        default=None,
        description="Optional stable current manifest path for this counter.",
    )
    object_uri: str | None = Field(
        default=None,
        description="Optional s3:// URI if the manifest is object-backed.",
    )

    @field_validator("manifest_path", "current_path")
    @classmethod
    def validate_manifest_path(cls, value: str | None) -> str | None:
        """Validate local manifest paths without constraining runtime roots."""

        return validate_filesystem_path(value)

    @field_validator("object_uri")
    @classmethod
    def validate_object_uri(cls, value: str | None) -> str | None:
        """Require optional object references to use the s3:// scheme."""

        if value is None:
            return None

        parsed = urlparse(value)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError("object_uri must be a valid s3:// URI")
        if "@" in parsed.netloc:
            raise ValueError("object_uri must not embed credentials")

        return value


class PipelineJobRequest(StrictPipelineContract):
    """Base request fields shared by every typed pipeline job."""

    job_type: PipelineJobType = Field(
        description="Requested pipeline job type.",
    )
    job_id: str | None = Field(
        default=None,
        description="Optional caller-provided or runner-assigned job id.",
    )
    run_id: str = Field(
        min_length=1,
        description="External pipeline run id shared by related jobs.",
    )
    counter_id: str = Field(
        min_length=1,
        description="Counter identifier processed by the job.",
    )
    requested_at: datetime = Field(
        default_factory=utc_now,
        description="Timezone-aware request creation timestamp.",
    )
    dag_id: str | None = Field(
        default=None,
        description="Optional Airflow DAG id for orchestration traceability.",
    )
    task_id: str | None = Field(
        default=None,
        description="Optional Airflow task id for orchestration traceability.",
    )
    try_number: int | None = Field(
        default=None,
        ge=1,
        description="Optional Airflow try number used for idempotency.",
    )
    manifest_root: str | None = Field(
        default=None,
        description="Optional local root where artifact manifests are written.",
    )

    @field_validator("requested_at")
    @classmethod
    def validate_requested_at(cls, value: datetime) -> datetime:
        """Require timezone-aware request timestamps."""

        return ensure_timezone_aware(value, "requested_at")

    @field_validator("manifest_root")
    @classmethod
    def validate_manifest_root(cls, value: str | None) -> str | None:
        """Validate optional local manifest roots."""

        return validate_filesystem_path(value)


class IngestJobRequest(PipelineJobRequest):
    """Typed request for the ingestion pipeline step."""

    job_type: Literal[PipelineJobType.INGEST] = PipelineJobType.INGEST
    raw_path: str = Field(
        min_length=1,
        description="Raw CSV path consumed by the ingestion step.",
    )
    site: str = Field(
        min_length=1,
        description="Exact source site name to extract from the raw dataset.",
    )
    orientation: str = Field(
        min_length=1,
        description="Counter orientation to extract from the raw dataset.",
    )
    range_start: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Start percentage for chronological source slicing.",
    )
    range_end: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="End percentage for chronological source slicing.",
    )
    timestamp_col: str = Field(
        default="date_et_heure_de_comptage",
        min_length=1,
        description="Raw timestamp column used for chronological ordering.",
    )
    sub_dir: str = Field(
        min_length=1,
        description="Scenario sub-directory used by downstream artifacts.",
    )
    interim_name: str = Field(
        default="initial.csv",
        min_length=1,
        description="Interim CSV file name produced by ingestion.",
    )
    interim_output_path: str = Field(
        min_length=1,
        description="Expected interim CSV path produced by ingestion.",
    )

    @field_validator("raw_path", "interim_output_path")
    @classmethod
    def validate_ingest_paths(cls, value: str) -> str:
        """Validate ingestion input and output paths."""

        return validate_filesystem_path(value) or value

    @model_validator(mode="after")
    def validate_percent_range(self) -> IngestJobRequest:
        """Ensure the chronological slice has an increasing range."""

        if self.range_start >= self.range_end:
            raise ValueError("range_start must be lower than range_end")

        return self


class FeatureJobRequest(PipelineJobRequest):
    """Typed request for the feature engineering pipeline step."""

    job_type: Literal[PipelineJobType.FEATURES] = PipelineJobType.FEATURES
    interim_input_path: str = Field(
        min_length=1,
        description="Interim CSV path produced by ingestion.",
    )
    processed_output_path: str = Field(
        min_length=1,
        description="Processed CSV path produced by feature engineering.",
    )
    processed_name: str = Field(
        default="initial_with_feats.csv",
        min_length=1,
        description="Processed CSV file name produced by feature engineering.",
    )
    timestamp_col: str = Field(
        default="date_et_heure_de_comptage",
        min_length=1,
        description="Timestamp column used to derive periodic features.",
    )
    extra_drop: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Additional columns to drop during feature engineering.",
    )

    @field_validator("interim_input_path", "processed_output_path")
    @classmethod
    def validate_feature_paths(cls, value: str) -> str:
        """Validate feature input and output paths."""

        return validate_filesystem_path(value) or value


class ModelJobRequest(PipelineJobRequest):
    """Typed request for the model training and prediction pipeline step."""

    job_type: Literal[PipelineJobType.MODELS] = PipelineJobType.MODELS
    processed_input_path: str = Field(
        min_length=1,
        description="Processed CSV path produced by feature engineering.",
    )
    prediction_output_path: str | None = Field(
        default=None,
        description="Expected prediction artifact path, usually y_full.csv.",
    )
    model_output_path: str | None = Field(
        default=None,
        description="Expected local model artifact directory or file path.",
    )
    target_col: str = Field(
        default="comptage_horaire",
        min_length=1,
        description="Regression target column.",
    )
    ts_col_utc: str = Field(
        default="date_et_heure_de_comptage_utc",
        min_length=1,
        description="UTC timestamp column consumed by the model step.",
    )
    ts_col_local: str = Field(
        default="date_et_heure_de_comptage_local",
        min_length=1,
        description="Local timestamp column consumed by the model step.",
    )
    ar: int = Field(
        default=7,
        ge=0,
        description="Number of autoregressive lags.",
    )
    mm: int = Field(
        default=1,
        ge=0,
        description="Number of moving-average feature windows.",
    )
    roll: int = Field(
        default=24,
        ge=1,
        description="Base rolling window size in hours.",
    )
    test_ratio: float = Field(
        default=0.25,
        gt=0.0,
        lt=0.95,
        description="Chronological test split ratio.",
    )
    grid_iter: int = Field(
        default=0,
        ge=0,
        description="Bayesian search iterations. Zero disables search.",
    )
    mlflow_uri: str | None = Field(
        default=None,
        description="Optional MLflow tracking URI override.",
    )
    artifact_object_uri: str | None = Field(
        default=None,
        description="Optional s3:// URI for the prediction artifact payload.",
    )
    expected_manifest: ArtifactManifestReference | None = Field(
        default=None,
        description="Optional manifest reference expected after model execution.",
    )

    @field_validator(
        "processed_input_path",
        "prediction_output_path",
        "model_output_path",
    )
    @classmethod
    def validate_model_paths(cls, value: str | None) -> str | None:
        """Validate model input and output paths."""

        return validate_filesystem_path(value)

    @field_validator("artifact_object_uri")
    @classmethod
    def validate_artifact_object_uri(cls, value: str | None) -> str | None:
        """Require optional artifact object references to use s3:// URIs."""

        if value is None:
            return None

        parsed = urlparse(value)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError("artifact_object_uri must be a valid s3:// URI")
        if "@" in parsed.netloc:
            raise ValueError("artifact_object_uri must not embed credentials")

        return value


class FullPipelineJobRequest(PipelineJobRequest):
    """Typed request that chains ingest, features, and model jobs coherently."""

    job_type: Literal[PipelineJobType.PIPELINE] = PipelineJobType.PIPELINE
    ingest: IngestJobRequest = Field(
        description="Ingestion step request.",
    )
    features: FeatureJobRequest = Field(
        description="Feature engineering step request.",
    )
    models: ModelJobRequest = Field(
        description="Model training and prediction step request.",
    )

    @model_validator(mode="after")
    def validate_step_coherence(self) -> FullPipelineJobRequest:
        """Validate shared context and artifact handoff between pipeline steps."""

        steps: tuple[PipelineJobRequest, ...] = (
            self.ingest,
            self.features,
            self.models,
        )
        for step in steps:
            if step.run_id != self.run_id:
                raise ValueError("all pipeline steps must share run_id")
            if step.counter_id != self.counter_id:
                raise ValueError("all pipeline steps must share counter_id")
            if self.manifest_root and step.manifest_root != self.manifest_root:
                raise ValueError("all pipeline steps must share manifest_root")

        if self.ingest.interim_output_path != self.features.interim_input_path:
            raise ValueError(
                "ingest interim_output_path must match "
                "features interim_input_path",
            )
        if self.features.processed_output_path != self.models.processed_input_path:
            raise ValueError(
                "features processed_output_path must match "
                "models processed_input_path",
            )

        return self


def ensure_timezone_aware(value: datetime, field_name: str) -> datetime:
    """Validate that a datetime value includes timezone information."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")

    return value


def validate_filesystem_path(value: str | None) -> str | None:
    """Validate local or container filesystem paths used by job contracts."""

    if value is None:
        return None
    if not value:
        raise ValueError("path values must not be empty")

    parsed = urlparse(value)
    if parsed.scheme:
        raise ValueError("path values must use local filesystem paths")

    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ValueError("path values must not contain parent traversal")

    return value
