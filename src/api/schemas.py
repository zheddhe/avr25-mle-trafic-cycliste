"""Pydantic response schemas for the prediction serving API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CustomException(BaseModel):
    """Structured API error payload."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(alias="ErrorId")
    type: str = Field(alias="ErrorType")
    message: str = Field(alias="ErrorMessage")


class Counter(BaseModel):
    """Available counter identifier returned by the API."""

    counter_id: str = Field(alias="CounterId")


class Prediction(BaseModel):
    """Prediction row returned by the API."""

    counter_id: str = Field(alias="CounterId")
    date: str = Field(alias="Date")
    y_true: float | None = Field(default=None, alias="YTrue")
    y_pred: float = Field(alias="YPred")
    forecast_mode: bool = Field(alias="ForecastMode")


class PredictionInput(BaseModel):
    """Prediction query payload."""

    counter_id: str = Field(alias="CounterId")
    start_date: str = Field(alias="StartDate")
    end_date: str = Field(alias="EndDate")


class ArtifactSourceMetadata(BaseModel):
    """Sanitized metadata for the currently served prediction artifact."""

    counter_id: str = Field(alias="CounterId")
    run_id: str = Field(alias="RunId")
    artifact_type: str = Field(alias="ArtifactType")
    status: str = Field(alias="Status")
    primary_backend: str = Field(alias="PrimaryBackend")
    local_path: str | None = Field(default=None, alias="LocalPath")
    checksum_sha256: str | None = Field(default=None, alias="ChecksumSha256")
    loaded_rows: int = Field(alias="LoadedRows")


class AdminRefreshResponse(BaseModel):
    """Admin refresh response for manifest-first prediction serving."""

    status: str = Field(alias="Status")
    manifest_root: str = Field(alias="ManifestRoot")
    repository_root: str = Field(alias="RepositoryRoot")
    loaded: int = Field(alias="Loaded")
    artifacts: list[ArtifactSourceMetadata] = Field(alias="Artifacts")


class CurrentArtifactMetadata(BaseModel):
    """Current promoted prediction artifacts served by the API."""

    artifacts: list[ArtifactSourceMetadata] = Field(alias="Artifacts")
