"""Execution adapters for typed ML step jobs."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Protocol

from src.common.env import get_env
from src.ml.jobs.contracts import MlJobType, StepJobRequest
from src.ml.jobs.execution import MlStepExecutionError, StepCommandExecutor
from src.ml.jobs.status import JobResult, JobStatus

MlJobExecutionError = MlStepExecutionError

JOB_RUNNER_MAX_IN_FLIGHT_JOBS_ENV = "JOB_RUNNER_MAX_IN_FLIGHT_JOBS"
DEFAULT_MAX_IN_FLIGHT_JOBS = 2


class MlJobExecutor(Protocol):
    """Protocol implemented by concrete typed ML job executors."""

    def execute(
        self,
        job_request: StepJobRequest,
        *,
        job_id: str,
        started_at: datetime,
    ) -> JobResult:
        """Execute one typed ML step and return its result evidence."""
        ...


class LocalMlJobExecutor(StepCommandExecutor):
    """Execute one allow-listed ML step in the local Python process."""


class ServiceMlJobExecutor:
    """Execute allow-listed ML steps through internal ML service APIs."""

    def __init__(
        self,
        transport: MlServiceTransport | None = None,
        endpoints: dict[MlJobType, str] | None = None,
        max_in_flight_jobs: int | None = None,
        concurrency_limiter: threading.BoundedSemaphore | None = None,
    ) -> None:
        self._transport = transport or UrllibMlServiceTransport()
        self._endpoints = (
            endpoints if endpoints is not None else _load_service_endpoints()
        )
        self.max_in_flight_jobs = _resolve_max_in_flight_jobs(
            max_in_flight_jobs,
        )
        self._concurrency_limiter = concurrency_limiter or threading.BoundedSemaphore(
            self.max_in_flight_jobs,
        )

    def execute(
        self,
        job_request: StepJobRequest,
        *,
        job_id: str,
        started_at: datetime,
    ) -> JobResult:
        """Execute one typed job through the matching ML service endpoint."""

        endpoint = self._endpoints.get(job_request.job_type)
        if endpoint is None:
            raise MlJobExecutionError(
                code="UNSUPPORTED_JOB_TYPE",
                message=(
                    "No ML service endpoint is configured for "
                    f"job type {job_request.job_type.value}."
                ),
                retryable=False,
            )

        with self._concurrency_limiter:
            status = self._transport.submit(
                endpoint=endpoint,
                job_request=job_request,
            )
        if status.state.value != "succeeded" or status.result is None:
            raise MlJobExecutionError(
                code=status.error.code if status.error else "ML_SERVICE_JOB_FAILED",
                message=(
                    status.error.message
                    if status.error
                    else "ML service returned a non-successful status."
                ),
                retryable=status.error.retryable if status.error else True,
            )

        return status.result.model_copy(
            update={
                "job_id": job_id,
                "started_at": started_at,
            },
        )


class MlServiceTransport(Protocol):
    """Transport used by the service executor."""

    def submit(self, *, endpoint: str, job_request: StepJobRequest) -> JobStatus:
        """Submit one typed job request to an ML step service."""
        ...


class UrllibMlServiceTransport:
    """Small stdlib HTTP transport for internal ML service calls."""

    def __init__(self, timeout_seconds: float | None = None) -> None:
        self._timeout_seconds = timeout_seconds or float(
            get_env("JOB_RUNNER_SERVICE_TIMEOUT_SECONDS", default="3600")
        )

    def submit(self, *, endpoint: str, job_request: StepJobRequest) -> JobStatus:
        """POST a typed job request and validate the returned status."""

        payload = _json_bytes(job_request.model_dump(mode="json"))
        request = urllib.request.Request(
            url=_join_url(endpoint, "/jobs"),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise MlJobExecutionError(
                code="ML_SERVICE_HTTP_ERROR",
                message=f"ML service returned HTTP {error.code}: {body}",
                retryable=error.code >= 500,
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise MlJobExecutionError(
                code="ML_SERVICE_UNAVAILABLE",
                message=f"ML service call failed: {error}",
                retryable=True,
            ) from error

        try:
            return JobStatus.model_validate_json(body)
        except ValueError as error:
            raise MlJobExecutionError(
                code="ML_SERVICE_INVALID_RESPONSE",
                message="ML service returned an invalid JobStatus payload.",
                retryable=True,
            ) from error


def _load_service_endpoints() -> dict[MlJobType, str]:
    return {
        MlJobType.INGEST: get_env(
            "ML_INGEST_SERVICE_URL",
            default="http://ml-ingest-prod:10081",
        ),
        MlJobType.FEATURES: get_env(
            "ML_FEATURES_SERVICE_URL",
            default="http://ml-features-prod:10082",
        ),
        MlJobType.MODELS: get_env(
            "ML_MODELS_SERVICE_URL",
            default="http://ml-models-prod:10083",
        ),
    }


def _resolve_max_in_flight_jobs(configured_value: int | None = None) -> int:
    if configured_value is not None:
        return _validate_positive_int(
            configured_value,
            name="max_in_flight_jobs",
        )

    raw_value = get_env(
        JOB_RUNNER_MAX_IN_FLIGHT_JOBS_ENV,
        default=str(DEFAULT_MAX_IN_FLIGHT_JOBS),
    )
    try:
        parsed_value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{JOB_RUNNER_MAX_IN_FLIGHT_JOBS_ENV} must be a positive integer."
        ) from error

    return _validate_positive_int(
        parsed_value,
        name=JOB_RUNNER_MAX_IN_FLIGHT_JOBS_ENV,
    )


def _validate_positive_int(value: int, *, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be greater than or equal to 1.")
    return value


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")
