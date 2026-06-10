"""Unit tests for job runner FastAPI dependencies."""

from __future__ import annotations

from types import SimpleNamespace

from src.job_runner.dependencies import get_job_runner_service


class TestGetJobRunnerService:
    """Unit tests for get_job_runner_service."""

    def test_returns_application_scoped_service(self) -> None:
        service = object()
        state = SimpleNamespace(job_runner_service=service)
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        assert get_job_runner_service(request) is service
