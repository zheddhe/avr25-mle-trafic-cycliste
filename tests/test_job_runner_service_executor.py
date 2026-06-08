"""Tests for job-runner ML service executor dispatch."""

from __future__ import annotations

from src.job_runner.executor import ServiceMlJobExecutor
from src.job_runner.service import build_default_executor
from src.ml.jobs.contracts import IngestJobRequest, MlJobType
from src.ml.jobs.status import JobResult, JobState, JobStatus, utc_now


def _build_ingest_request() -> IngestJobRequest:
    return IngestJobRequest(
        run_id="run-001",
        counter_id="counter-001",
        manifest_root="/app/artifacts/manifests",
        raw_path="/app/data/raw/source.csv",
        site="Totem 73 boulevard de Sébastopol",
        orientation="N-S",
        sub_dir="counter-001",
        interim_output_path="/app/data/interim/counter-001/initial.csv",
    )


class FakeTransport:
    """Test double capturing ML service submissions."""

    def __init__(self, status: JobStatus) -> None:
        self.status = status
        self.calls: list[tuple[str, IngestJobRequest]] = []

    def submit(self, *, endpoint, job_request):
        self.calls.append((endpoint, job_request))
        return self.status


class TestServiceMlJobExecutor:
    """Validate runner-to-service dispatch."""

    def test_execute_dispatches_to_matching_service_endpoint(self) -> None:
        job_request = _build_ingest_request()
        now = utc_now()
        status = JobStatus(
            job_id="service-job-001",
            run_id=job_request.run_id,
            counter_id=job_request.counter_id,
            job_type=job_request.job_type,
            state=JobState.SUCCEEDED,
            requested_at=now,
            updated_at=now,
            result=JobResult(
                job_id="service-job-001",
                run_id=job_request.run_id,
                counter_id=job_request.counter_id,
                job_type=job_request.job_type,
                started_at=now,
                finished_at=now,
                output_paths=(job_request.interim_output_path,),
            ),
        )
        transport = FakeTransport(status)
        executor = ServiceMlJobExecutor(
            transport=transport,
            endpoints={MlJobType.INGEST: "http://ml-ingest-prod:10081"},
        )

        result = executor.execute(
            job_request,
            job_id="runner-job-001",
            started_at=now,
        )

        assert result.job_id == "runner-job-001"
        assert result.output_paths == ("/app/data/interim/counter-001/initial.csv",)
        assert transport.calls == [("http://ml-ingest-prod:10081", job_request)]

    def test_build_default_executor_uses_service_mode_by_default(self) -> None:
        executor = build_default_executor()

        assert isinstance(executor, ServiceMlJobExecutor)
