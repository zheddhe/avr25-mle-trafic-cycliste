"""Error types and API responses for the internal job runner."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Stable HTTP error response exposed by the runner API."""

    code: str = Field(
        description="Stable machine-readable error code.",
        min_length=1,
    )
    message: str = Field(
        description="Human-readable error message.",
        min_length=1,
    )


class JobRunnerError(Exception):
    """Base error raised by the internal job runner service."""

    status_code = 500
    code = "JOB_RUNNER_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class JobNotFoundError(JobRunnerError):
    """Raised when a job id is unknown to the in-memory runner state."""

    status_code = 404
    code = "JOB_NOT_FOUND"

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job '{job_id}' was not found.")
        self.job_id = job_id


def job_runner_error_handler(
    request: Request,
    exception: Exception,
) -> JSONResponse:
    """Convert runner domain errors to explicit JSON responses."""

    if isinstance(exception, JobRunnerError):
        status_code = exception.status_code
        error_code = exception.code
        message = exception.message
    else:
        status_code = 500
        error_code = "INTERNAL_ERROR"
        message = "An unexpected error occurred."

    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            code=error_code,
            message=message,
        ).model_dump(),
    )
