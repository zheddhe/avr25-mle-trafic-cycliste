"""Service layer for the internal job runner API."""

from __future__ import annotations

import os

from src.job_runner.errors import JobNotFoundError
from src.job_runner.executor import (
    LocalMlJobExecutor,
    MlJobExecutionError,
    MlJobExecutor,
    ServiceMlJobExecutor,
)
from src.job_runner.state import InMemoryJobState
from src.ml.jobs.contracts import StepJobRequest
from src.ml.jobs.status import JobError, JobState, JobStatus


class JobRunnerService:
    """Submit, execute, and expose local in-memory runner job statuses."""

    def __init__(
        self,
        state: InMemoryJobState | None = None,
        executor: MlJobExecutor | None = None,
    ) -> None:
        self._state = state or InMemoryJobState()
        self._executor = executor or build_default_executor()

    def submit_job(self, job_request: StepJobRequest) -> JobStatus:
        """Submit a typed ML step request and execute it synchronously."""

        status = self._state.submit(job_request)
        if status.state != JobState.QUEUED:
            return status

        running_status = self._state.set_running(status.job_id)
        try:
            result = self._executor.execute(
                job_request,
                job_id=status.job_id,
                started_at=running_status.updated_at,
            )
        except MlJobExecutionError as error:
            return self._state.set_failed(
                status.job_id,
                JobError(
                    code=error.code,
                    message=error.message,
                    retryable=error.retryable,
                ),
            )
        except Exception as error:
            return self._state.set_failed(
                status.job_id,
                JobError(
                    code=f"{job_request.job_type.value.upper()}_JOB_FAILED",
                    message=str(error),
                    retryable=True,
                ),
            )

        return self._state.set_succeeded(status.job_id, result)

    def get_job_status(self, job_id: str) -> JobStatus:
        """Return the current job status or raise an explicit not-found error."""

        status = self._state.get(job_id)
        if status is None:
            raise JobNotFoundError(job_id)

        return status


def build_default_executor() -> MlJobExecutor:
    """Build the runner executor selected by runtime configuration."""

    executor_name = os.getenv("JOB_RUNNER_EXECUTOR", "service").strip().lower()
    if executor_name == "local":
        return LocalMlJobExecutor()
    if executor_name == "service":
        return ServiceMlJobExecutor()

    raise ValueError("JOB_RUNNER_EXECUTOR must be 'service' or 'local'.")
