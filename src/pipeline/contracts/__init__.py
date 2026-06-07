"""Typed pipeline contracts shared by orchestration and runner code."""

from src.pipeline.contracts.jobs import (
    ArtifactManifestReference,
    BasePipelineJobRequest,
    FeatureJobRequest,
    IngestJobRequest,
    ModelJobRequest,
    PipelineJobRequest,
    PipelineJobType,
)
from src.pipeline.contracts.statuses import (
    TERMINAL_JOB_STATES,
    JobError,
    JobResult,
    JobState,
    JobStatus,
    MetricsEvidence,
)

__all__ = [
    "ArtifactManifestReference",
    "BasePipelineJobRequest",
    "FeatureJobRequest",
    "IngestJobRequest",
    "JobError",
    "JobResult",
    "JobState",
    "JobStatus",
    "MetricsEvidence",
    "ModelJobRequest",
    "PipelineJobRequest",
    "PipelineJobType",
    "TERMINAL_JOB_STATES",
]
