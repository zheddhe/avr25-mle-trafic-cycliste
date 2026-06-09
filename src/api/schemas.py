"""Pydantic response schemas for the prediction serving API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Structured API business error payload."""

    type: str = Field(..., description="Business error type.")
    message: str | None = Field(
        ..., description="Optional detailed error message."
    )
    date: str = Field(..., description="Server-side timestamp.")


class PredictionItem(BaseModel):
    """Single prediction item for one counter."""

    date_et_heure_de_comptage_local: datetime = Field(
        ..., description="Local timestamp (Europe/Paris)."
    )
    date_et_heure_de_comptage_utc: datetime = Field(
        ..., description="UTC timestamp."
    )
    y_true: int | float | None = Field(..., description="Observed value.")
    y_pred: float = Field(..., description="Predicted value.")
    forecast_mode: bool = Field(
        ..., description="True if predicted on future timestamps."
    )


class PredictionList(BaseModel):
    """Paginated prediction response."""

    total: int = Field(..., description="Total available predictions.")
    limit: int = Field(..., description="Max returned.")
    offset: int = Field(..., description="Pagination offset.")
    item: list[PredictionItem] = Field(
        ..., description="Paginated list of predictions."
    )


class Counter(BaseModel):
    """Available counter identifier."""

    id: str = Field(..., description="Counter identifier.")


class ArtifactSourceMetadata(BaseModel):
    """Sanitized source metadata for a served artifact."""

    raw_file_name: str | None = Field(default=None)
    dataset_version: str | None = Field(default=None)
    model_version: str | None = Field(default=None)


class CurrentArtifactMetadata(BaseModel):
    """Current promoted prediction artifact metadata."""

    counter_id: str = Field(..., description="Counter served by the artifact.")
    run_id: str = Field(..., description="Run id recorded by the manifest.")
    artifact_type: str = Field(..., description="Manifest artifact type.")
    status: str = Field(..., description="Manifest lifecycle status.")
    created_at: datetime = Field(..., description="Manifest creation timestamp.")
    producer_service: str = Field(..., description="Producer service name.")
    producer_image: str | None = Field(default=None)
    producer_version: str | None = Field(default=None)
    source: ArtifactSourceMetadata = Field(...)
    primary_backend: str = Field(..., description="Primary storage backend.")
    local_path: str | None = Field(default=None)
    object_uri: str | None = Field(default=None)
    checksum_sha256: str | None = Field(default=None)


class AdminRefreshResponse(BaseModel):
    """Admin refresh response for manifest-first prediction serving."""

    message: str = Field(..., description="Operation result.")
    counters_before: int = Field(..., description="Store size before refresh.")
    counters_after: int = Field(..., description="Store size after refresh.")
    manifest_root: str = Field(..., description="Manifest root used.")
    repository_root: str = Field(..., description="Repository root used.")
    loaded: int = Field(..., description="Counters successfully loaded.")
