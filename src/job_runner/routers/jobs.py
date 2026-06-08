"""Job submission and status endpoints for the internal runner API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import Field

from src.job_runner.dependencies import get_job_runner_service
from src.job_runner.errors import ErrorResponse
from src.job_runner.service import JobRunnerService
from src.ml.jobs.contracts import FeatureJobRequest, IngestJobRequest, ModelJobRequest
from src.ml.jobs.status import JobStatus

JobRequest = Annotated[
    IngestJobRequest | FeatureJobRequest | ModelJobRequest,
    Field(discriminator="job_type"),
]

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit and execute a typed ML step job",
    responses={
        202: {"description": "Typed ML step job accepted and executed."},
        422: {"description": "Invalid typed ML step job request payload."},
    },
)
def submit_job(
    job_request: JobRequest,
    service: JobRunnerService = Depends(get_job_runner_service),
) -> JobStatus:
    """Accept one typed ML step request and execute it through the runner."""

    return service.submit_job(job_request)


@router.get(
    "/{job_id}",
    response_model=JobStatus,
    summary="Get a typed ML step job status",
    responses={
        200: {"description": "Current typed ML step job status."},
        404: {"description": "Unknown job id.", "model": ErrorResponse},
    },
)
def get_job_status(
    job_id: str,
    service: JobRunnerService = Depends(get_job_runner_service),
) -> JobStatus:
    """Return the current status for an in-memory job id."""

    return service.get_job_status(job_id)
