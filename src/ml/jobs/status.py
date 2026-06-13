"""Typed ML job status and result contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from src.ml.jobs.contracts import (
    ArtifactManifestReference,
    MlJobType,
    StrictMlJobContract,
    ensure_timezone_aware,
)


def utc_now() -> datetime:
    """Return the current UTC timestamp for status defaults."""

    return datetime.now(UTC)


class JobState(StrEnum):
    """Runner job states observed by Airflow and typed workers."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"


TERMINAL_JOB_STATES = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELED,
        JobState.EXPIRED,
    },
)


class JobError(StrictMlJobContract):
    """Structured error information attached to failed job states."""

    code: str = Field(
        min_length=1,
        description="Stable error code produced by the runner or worker.",
    )
    message: str = Field(
        min_length=1,
        description="Human-readable failure reason.",
    )
    retryable: bool = Field(
        default=False,
        description="Whether the caller may retry the failed job safely.",
    )


class MetricsEvidence(StrictMlJobContract):
    """Optional metrics evidence without Prometheus implementation coupling."""

    records: int | None = Field(
        default=None,
        ge=0,
        description="Optional processed record count.",
    )
    metrics_pushed: bool | None = Field(
        default=None,
        description="Whether technical metrics were pushed by the worker.",
    )
    metrics_reference: str | None = Field(
        default=None,
        description="Optional metrics log, run, or dashboard reference.",
    )


class JobResult(StrictMlJobContract):
    """Typed result returned by a completed or partially completed ML job."""

    job_id: str = Field(
        min_length=1,
        description="Runner job identifier.",
    )
    run_id: str = Field(
        min_length=1,
        description="External ML run identifier.",
    )
    counter_id: str = Field(
        min_length=1,
        description="Counter identifier processed by the job.",
    )
    job_type: MlJobType = Field(
        description="ML job type that produced this result.",
    )
    started_at: datetime | None = Field(
        default=None,
        description="Timezone-aware timestamp when execution started.",
    )
    finished_at: datetime | None = Field(
        default=None,
        description="Timezone-aware timestamp when execution finished.",
    )
    output_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Local output paths produced or verified by the job.",
    )
    manifest: ArtifactManifestReference | None = Field(
        default=None,
        description="Optional artifact manifest produced by the job.",
    )
    metrics: MetricsEvidence | None = Field(
        default=None,
        description="Optional technical or business metric evidence.",
    )

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_timestamps(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware result timestamps when present."""

        if value is None:
            return None

        return ensure_timezone_aware(value, "result timestamp")

    @model_validator(mode="after")
    def validate_finished_after_started(self) -> JobResult:
        """Ensure result timestamps are chronological when both are present."""

        started_at = self.started_at
        finished_at = self.finished_at
        if started_at is not None and finished_at is not None and finished_at < started_at:
            raise ValueError(
                "finished_at must be greater than or equal to started_at",
            )

        return self


class JobStatus(StrictMlJobContract):
    """Typed status exposed by the runner API and observed by Airflow."""

    job_id: str = Field(
        min_length=1,
        description="Runner job identifier.",
    )
    run_id: str = Field(
        min_length=1,
        description="External ML run identifier.",
    )
    counter_id: str = Field(
        min_length=1,
        description="Counter identifier processed by the job.",
    )
    job_type: MlJobType = Field(
        description="ML job type represented by this status.",
    )
    state: JobState = Field(
        description="Current runner job state.",
    )
    requested_at: datetime = Field(
        default_factory=utc_now,
        description="Timezone-aware timestamp when the job was requested.",
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        description="Timezone-aware timestamp when the status last changed.",
    )
    result: JobResult | None = Field(
        default=None,
        description="Optional result attached to terminal successful jobs.",
    )
    error: JobError | None = Field(
        default=None,
        description="Optional error attached to failed jobs.",
    )

    @field_validator("requested_at", "updated_at")
    @classmethod
    def validate_status_timestamps(cls, value: datetime) -> datetime:
        """Require timezone-aware status timestamps."""

        return ensure_timezone_aware(value, "status timestamp")

    @model_validator(mode="after")
    def validate_status_consistency(self) -> JobStatus:
        """Validate result and error consistency for terminal states."""

        if self.updated_at < self.requested_at:
            raise ValueError("updated_at must be greater than or equal to requested_at")
        if self.state == JobState.SUCCEEDED and self.result is None:
            raise ValueError("succeeded jobs must include result")
        if self.state == JobState.FAILED and self.error is None:
            raise ValueError("failed jobs must include error")
        if self.state != JobState.FAILED and self.error is not None:
            raise ValueError("only failed jobs may include error")
        if self.state not in TERMINAL_JOB_STATES and self.result is not None:
            raise ValueError("non-terminal jobs must not include result")

        return self
