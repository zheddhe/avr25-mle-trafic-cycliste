"""In-memory job state used by the internal runner."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import sha256
from threading import Lock

from src.ml.jobs.contracts import BaseMlJobRequest
from src.ml.jobs.status import JobError, JobResult, JobState, JobStatus

_ID_PART_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")
_MAX_JOB_ID_SLUG_LENGTH = 48


def utc_now() -> datetime:
    """Return the current UTC timestamp for runner state transitions."""

    return datetime.now(UTC)


def _slugify(value: str) -> str:
    """Return a readable identifier segment safe for logs and URLs."""

    slug = _ID_PART_PATTERN.sub("-", value.strip()).strip("-._")
    if not slug:
        return "unknown"

    return slug[:_MAX_JOB_ID_SLUG_LENGTH]


def build_idempotency_key(job_request: BaseMlJobRequest) -> str:
    """Build the stable submission key for a typed job request."""

    if job_request.job_id:
        return f"job_id={job_request.job_id}"

    key_parts = (
        f"job_type={job_request.job_type.value}",
        f"run_id={job_request.run_id}",
        f"counter_id={job_request.counter_id}",
        f"dag_id={job_request.dag_id or ''}",
        f"task_id={job_request.task_id or ''}",
        f"try_number={job_request.try_number or 0}",
    )
    return "|".join(key_parts)


def build_job_id(job_request: BaseMlJobRequest) -> str:
    """Build an explicit, deterministic runner job id."""

    if job_request.job_id:
        return job_request.job_id

    idempotency_key = build_idempotency_key(job_request)
    digest = sha256(idempotency_key.encode("utf-8")).hexdigest()[:12]
    job_type = _slugify(job_request.job_type.value)
    counter_id = _slugify(job_request.counter_id)

    return f"job-{job_type}-{counter_id}-{digest}"


class InMemoryJobState:
    """Thread-safe in-memory storage for local production-like job statuses."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._statuses_by_id: dict[str, JobStatus] = {}
        self._job_ids_by_idempotency_key: dict[str, str] = {}

    def submit(self, job_request: BaseMlJobRequest) -> JobStatus:
        """Record a typed job request and return its queued status."""

        job_id = build_job_id(job_request)
        idempotency_key = build_idempotency_key(job_request)

        with self._lock:
            existing_job_id = self._job_ids_by_idempotency_key.get(
                idempotency_key,
            )
            if existing_job_id:
                return self._statuses_by_id[existing_job_id]

            existing_status = self._statuses_by_id.get(job_id)
            if existing_status:
                return existing_status

            status = JobStatus(
                job_id=job_id,
                run_id=job_request.run_id,
                counter_id=job_request.counter_id,
                job_type=job_request.job_type,
                state=JobState.QUEUED,
                requested_at=job_request.requested_at,
                updated_at=max(utc_now(), job_request.requested_at),
            )
            self._statuses_by_id[job_id] = status
            self._job_ids_by_idempotency_key[idempotency_key] = job_id

            return status

    def set_running(self, job_id: str) -> JobStatus:
        """Record that execution started for a queued job."""

        return self._transition(job_id=job_id, state=JobState.RUNNING)

    def set_succeeded(self, job_id: str, result: JobResult) -> JobStatus:
        """Record a successful terminal status with result evidence."""

        return self._transition(
            job_id=job_id,
            state=JobState.SUCCEEDED,
            result=result,
        )

    def set_failed(self, job_id: str, error: JobError) -> JobStatus:
        """Record a failed terminal status with structured error details."""

        return self._transition(job_id=job_id, state=JobState.FAILED, error=error)

    def get(self, job_id: str) -> JobStatus | None:
        """Return a stored job status when the id exists."""

        with self._lock:
            return self._statuses_by_id.get(job_id)

    def _transition(
        self,
        *,
        job_id: str,
        state: JobState,
        result: JobResult | None = None,
        error: JobError | None = None,
    ) -> JobStatus:
        with self._lock:
            current = self._statuses_by_id[job_id]
            status = JobStatus(
                job_id=current.job_id,
                run_id=current.run_id,
                counter_id=current.counter_id,
                job_type=current.job_type,
                state=state,
                requested_at=current.requested_at,
                updated_at=max(utc_now(), current.updated_at),
                result=result,
                error=error,
            )
            self._statuses_by_id[job_id] = status

            return status
