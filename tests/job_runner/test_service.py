"""Unit tests for job-runner service layer."""

from __future__ import annotations

from src.job_runner.executor import ServiceMlJobExecutor
from src.job_runner.service import JobRunnerService, build_default_executor
from src.ml.jobs.contracts import IngestJobRequest
from src.ml.jobs.status import JobResult, JobState

INTERIM_OUTPUT_PATH = "/app/data/interim/counter-001/initial.csv"


class FakeExecutor:
    """Test double returning a deterministic job result."""

    def execute(self, job_request, *, job_id, started_at):
        return JobResult(
            job_id=job_id,
            run_id=job_request.run_id,
            counter_id=job_request.counter_id,
            job_type=job_request.job_type,
            started_at=started_at,
            output_paths=(INTERIM_OUTPUT_PATH,),
        )


def _build_ingest_request() -> IngestJobRequest:
    return IngestJobRequest(
        run_id="run-001",
        counter_id="counter-001",
        raw_path="/app/data/raw/source.csv",
        site="Totem 73 boulevard de Sébastopol",
        orientation="N-S",
        sub_dir="counter-001",
        interim_output_path=INTERIM_OUTPUT_PATH,
    )


class TestJobRunnerService:
    """Unit tests for JobRunnerService."""

    def test_submit_job_executes_and_stores_successful_status(self) -> None:
        service = JobRunnerService(executor=FakeExecutor())

        status = service.submit_job(_build_ingest_request())

        assert status.state == JobState.SUCCEEDED
        assert status.result is not None
        assert service.get_job_status(status.job_id) is status

    def test_build_default_executor_uses_service_mode_by_default(self) -> None:
        executor = build_default_executor()

        assert isinstance(executor, ServiceMlJobExecutor)
