"""Unit tests for job runner domain errors."""

from __future__ import annotations

from types import SimpleNamespace

from src.job_runner.errors import JobNotFoundError, job_runner_error_handler


class TestJobRunnerErrors:
    """Unit tests for runner HTTP error conversion."""

    def test_job_not_found_error_contains_stable_payload(self) -> None:
        error = JobNotFoundError("job-404")

        response = job_runner_error_handler(SimpleNamespace(), error)

        assert response.status_code == 404
        assert b"JOB_NOT_FOUND" in response.body
        assert b"job-404" in response.body
