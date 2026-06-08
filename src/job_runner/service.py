"""Service layer for the internal job runner API skeleton."""

from __future__ import annotations

from src.job_runner.errors import JobNotFoundError
from src.job_runner.state import InMemoryJobState
from src.pipeline.contracts.jobs import BasePipelineJobRequest
from src.pipeline.contracts.statuses import JobStatus


class JobRunnerService:
    """Submit and expose local in-memory runner job statuses."""

    def __init__(self, state: InMemoryJobState | None = None) -> None:
        self._state = state or InMemoryJobState()

    def submit_job(self, job_request: BasePipelineJobRequest) -> JobStatus:
        """Submit a typed job request without starting real execution yet."""

        return self._state.submit(job_request)

    def get_job_status(self, job_id: str) -> JobStatus:
        """Return the current job status or raise an explicit not-found error."""

        status = self._state.get(job_id)
        if status is None:
            raise JobNotFoundError(job_id)

        return status
