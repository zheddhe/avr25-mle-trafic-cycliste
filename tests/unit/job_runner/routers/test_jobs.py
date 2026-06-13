"""Unit tests for job runner jobs router functions."""

from __future__ import annotations

from typing import Any

from src.job_runner.routers.jobs import get_job_status, submit_job
from src.ml.jobs.contracts import IngestJobRequest


class FakeJobRunnerService:
    """Test double for the job runner service dependency."""

    def __init__(self) -> None:
        self.submitted_request = None

    def submit_job(self, job_request):
        self.submitted_request = job_request
        return "submitted"

    def get_job_status(self, job_id: str) -> str:
        return f"status:{job_id}"


def _build_ingest_request() -> IngestJobRequest:
    return IngestJobRequest(
        run_id="run-001",
        counter_id="counter-001",
        raw_path="/app/data/raw/source.csv",
        site="Totem 73 boulevard de Sébastopol",
        orientation="N-S",
        sub_dir="counter-001",
        interim_output_path="/app/data/interim/counter-001/initial.csv",
    )


class TestJobsRouter:
    """Unit tests for job runner jobs endpoints."""

    def test_submit_job_delegates_to_service(self) -> None:
        service: Any = FakeJobRunnerService()
        job_request = _build_ingest_request()

        result = submit_job(job_request=job_request, service=service)

        assert result == "submitted"
        assert service.submitted_request is job_request

    def test_get_job_status_delegates_to_service(self) -> None:
        service: Any = FakeJobRunnerService()

        result = get_job_status(job_id="job-001", service=service)

        assert result == "status:job-001"
