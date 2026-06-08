"""FastAPI factory for internal ML step services."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, status

from src.ml.jobs.contracts import MlJobType, StepJobRequest
from src.ml.jobs.execution import MlStepExecutionError, StepCommandExecutor
from src.ml.jobs.status import JobError, JobState, JobStatus, utc_now

OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Internal service liveness endpoint for Compose healthchecks.",
    },
    {
        "name": "Jobs",
        "description": "Internal typed ML step execution endpoint.",
    },
]

JobRequestBody = Annotated[
    StepJobRequest,
    Body(discriminator="job_type"),
]


class MlStepService:
    """Execute one typed ML step for a single allow-listed job type."""

    def __init__(
        self,
        job_type: MlJobType,
        executor: StepCommandExecutor | None = None,
    ) -> None:
        self.job_type = job_type
        self._executor = executor or StepCommandExecutor()

    def execute(self, job_request: StepJobRequest) -> JobStatus:
        """Execute one job request synchronously and return a terminal status."""

        requested_at = utc_now()
        job_id = job_request.job_id or (
            f"{job_request.job_type.value}-{job_request.run_id}"
        )
        if job_request.job_type != self.job_type:
            return _failed_status(
                job_request=job_request,
                job_id=job_id,
                requested_at=requested_at,
                code="UNSUPPORTED_JOB_TYPE",
                message=(
                    f"Service {self.job_type.value} cannot execute "
                    f"{job_request.job_type.value} jobs."
                ),
                retryable=False,
            )

        started_at = utc_now()
        try:
            result = self._executor.execute(
                job_request,
                job_id=job_id,
                started_at=started_at,
            )
        except MlStepExecutionError as error:
            return _failed_status(
                job_request=job_request,
                job_id=job_id,
                requested_at=requested_at,
                code=error.code,
                message=error.message,
                retryable=error.retryable,
            )
        except Exception as error:
            return _failed_status(
                job_request=job_request,
                job_id=job_id,
                requested_at=requested_at,
                code=f"{job_request.job_type.value.upper()}_JOB_FAILED",
                message=str(error),
                retryable=True,
            )

        return JobStatus(
            job_id=job_id,
            run_id=job_request.run_id,
            counter_id=job_request.counter_id,
            job_type=job_request.job_type,
            state=JobState.SUCCEEDED,
            requested_at=requested_at,
            updated_at=utc_now(),
            result=result,
        )


def create_app(
    *,
    service_name: str,
    job_type: MlJobType,
    service: MlStepService | None = None,
) -> FastAPI:
    """Create one internal FastAPI application for a typed ML step."""

    app = FastAPI(
        title=f"Bike traffic {service_name} internal API",
        description=(
            "Private local production-like API used by job-runner-api to "
            "execute one typed ML step without Docker socket access."
        ),
        version="0.1.0",
        openapi_tags=OPENAPI_TAGS,
    )
    app.state.ml_step_service = service or MlStepService(job_type=job_type)

    @app.get("/health", tags=["Health"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": service_name,
            "job_type": job_type.value,
        }

    @app.post(
        "/jobs",
        response_model=JobStatus,
        status_code=status.HTTP_200_OK,
        tags=["Jobs"],
    )
    def submit_job(job_request: JobRequestBody) -> JobStatus:
        if job_request.job_type != job_type:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Service {service_name} only accepts "
                    f"{job_type.value} jobs."
                ),
            )

        return app.state.ml_step_service.execute(job_request)

    return app


def _failed_status(
    *,
    job_request: StepJobRequest,
    job_id: str,
    requested_at: datetime,
    code: str,
    message: str,
    retryable: bool,
) -> JobStatus:
    return JobStatus(
        job_id=job_id,
        run_id=job_request.run_id,
        counter_id=job_request.counter_id,
        job_type=job_request.job_type,
        state=JobState.FAILED,
        requested_at=requested_at,
        updated_at=utc_now(),
        error=JobError(code=code, message=message, retryable=retryable),
    )
