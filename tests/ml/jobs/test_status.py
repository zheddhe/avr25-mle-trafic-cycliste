"""Unit tests for typed ML job status contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.ml.jobs.contracts import MlJobType
from src.ml.jobs.status import JobError, JobState, JobStatus


class TestJobStatus:
    """Unit tests for JobStatus."""

    def test_pending_status_can_be_instantiated(self) -> None:
        status = JobStatus.model_validate(
            {
                "job_id": "job-001",
                "run_id": "manual-run-001",
                "counter_id": "Sebastopol_N-S_dvcrepro",
                "job_type": "ingest",
                "state": "pending",
                "requested_at": "2026-06-07T17:00:00Z",
                "updated_at": "2026-06-07T17:00:00Z",
            },
        )

        assert status.state == JobState.PENDING
        assert status.job_type == MlJobType.INGEST
        assert status.result is None
        assert status.error is None

    def test_failed_status_requires_and_accepts_error(self) -> None:
        status = JobStatus.model_validate(
            {
                "job_id": "job-001",
                "run_id": "manual-run-001",
                "counter_id": "Sebastopol_N-S_dvcrepro",
                "job_type": "models",
                "state": "failed",
                "requested_at": "2026-06-07T17:00:00Z",
                "updated_at": "2026-06-07T17:03:00Z",
                "error": {
                    "code": "MODEL_STEP_FAILED",
                    "message": "Model execution failed.",
                    "retryable": True,
                },
            },
        )

        assert status.state == JobState.FAILED
        assert isinstance(status.error, JobError)
        assert status.error.retryable is True

    def test_failed_status_without_error_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="failed jobs must include error"):
            JobStatus.model_validate(
                {
                    "job_id": "job-001",
                    "run_id": "manual-run-001",
                    "counter_id": "Sebastopol_N-S_dvcrepro",
                    "job_type": "models",
                    "state": "failed",
                    "requested_at": "2026-06-07T17:00:00Z",
                    "updated_at": "2026-06-07T17:03:00Z",
                },
            )
