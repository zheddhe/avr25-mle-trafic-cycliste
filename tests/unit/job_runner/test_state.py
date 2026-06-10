"""Unit tests for in-memory job runner state."""

from __future__ import annotations

from src.job_runner.state import InMemoryJobState, build_job_id
from src.ml.jobs.contracts import IngestJobRequest
from src.ml.jobs.status import JobError, JobState


def _build_ingest_request(job_id: str | None = None) -> IngestJobRequest:
    return IngestJobRequest(
        job_id=job_id,
        run_id="run-001",
        counter_id="counter-001",
        raw_path="/app/data/raw/source.csv",
        site="Totem 73 boulevard de Sébastopol",
        orientation="N-S",
        sub_dir="counter-001",
        interim_output_path="/app/data/interim/counter-001/initial.csv",
    )


class TestInMemoryJobState:
    """Unit tests for InMemoryJobState."""

    def test_submit_is_idempotent_for_same_request(self) -> None:
        state = InMemoryJobState()
        job_request = _build_ingest_request()

        first_status = state.submit(job_request)
        second_status = state.submit(job_request)

        assert first_status.job_id == build_job_id(job_request)
        assert second_status is first_status

    def test_set_failed_records_structured_error(self) -> None:
        state = InMemoryJobState()
        status = state.submit(_build_ingest_request(job_id="job-001"))

        failed_status = state.set_failed(
            status.job_id,
            JobError(code="FAILED", message="boom", retryable=True),
        )

        assert failed_status.state == JobState.FAILED
        assert failed_status.error is not None
        assert failed_status.error.code == "FAILED"
