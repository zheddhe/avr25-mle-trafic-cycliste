"""FastAPI application factory for the internal job runner service."""

from __future__ import annotations

from fastapi import FastAPI

from src.job_runner.errors import JobRunnerError, job_runner_error_handler
from src.job_runner.routers import health, jobs
from src.job_runner.service import JobRunnerService

OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Internal service liveness endpoint for Compose healthchecks.",
    },
    {
        "name": "Jobs",
        "description": (
            "Internal typed ML step submission and status endpoints. The runner "
            "executes one allow-listed step at a time and keeps state in memory."
        ),
    },
]


def create_app(service: JobRunnerService | None = None) -> FastAPI:
    """Create the internal FastAPI job runner application."""

    app = FastAPI(
        title="Bike traffic internal job runner API",
        description=(
            "Private local production-like API used by Airflow to submit typed "
            "ML step jobs without Docker socket access."
        ),
        version="0.2.0",
        openapi_tags=OPENAPI_TAGS,
    )
    app.state.job_runner_service = service or JobRunnerService()
    app.add_exception_handler(JobRunnerError, job_runner_error_handler)
    app.include_router(health.router)
    app.include_router(jobs.router)

    return app
