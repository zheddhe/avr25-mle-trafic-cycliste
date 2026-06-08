"""FastAPI dependencies for the internal job runner API."""

from __future__ import annotations

from fastapi import Request

from src.job_runner.service import JobRunnerService


def get_job_runner_service(request: Request) -> JobRunnerService:
    """Return the application-scoped job runner service."""

    return request.app.state.job_runner_service
