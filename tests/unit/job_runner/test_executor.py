# tests/job_runner/test_executor.py
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from email.message import Message
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import pytest
from src.job_runner.executor import (
    DEFAULT_MAX_IN_FLIGHT_JOBS,
    JOB_RUNNER_MAX_IN_FLIGHT_JOBS_ENV,
    MlJobExecutionError,
    ServiceMlJobExecutor,
    UrllibMlServiceTransport,
    _join_url,
    _json_bytes,
    _resolve_max_in_flight_jobs,
)
from src.job_runner.service import build_default_executor
from src.ml.jobs.contracts import (
    IngestJobRequest,
    MlJobType,
)
from src.ml.jobs.status import JobError, JobResult, JobState, JobStatus


class TestServiceMlJobExecutor:
    def test_execute_returns_service_result_with_runner_metadata(self) -> None:
        job_request = _ingest_request()
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        service_result = _job_result(job_id="service-job")
        status = _job_status(
            job_id="service-job",
            state=JobState.SUCCEEDED,
            result=service_result,
        )
        transport = Mock()
        transport.submit.return_value = status
        executor = ServiceMlJobExecutor(
            transport=transport,
            endpoints={job_request.job_type: "http://ml-ingest:10081"},
            max_in_flight_jobs=1,
        )

        result = executor.execute(
            job_request,
            job_id="runner-job",
            started_at=started_at,
        )

        assert result.job_id == "runner-job"
        assert result.started_at == started_at
        assert result.run_id == job_request.run_id
        assert result.output_paths == ("data/interim/counter-a/initial.csv",)
        transport.submit.assert_called_once()
        submitted_request = transport.submit.call_args.kwargs["job_request"]
        assert submitted_request.job_id == "runner-job"
        assert submitted_request.run_id == job_request.run_id
        assert submitted_request.counter_id == job_request.counter_id
        assert transport.submit.call_args.kwargs["endpoint"] == "http://ml-ingest:10081"

    def test_execute_raises_when_endpoint_is_missing(self) -> None:
        job_request = _ingest_request()
        executor = ServiceMlJobExecutor(
            transport=Mock(),
            endpoints={},
            max_in_flight_jobs=1,
        )

        with pytest.raises(MlJobExecutionError) as exc_info:
            executor.execute(
                job_request,
                job_id="runner-job",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

        assert exc_info.value.code == "UNSUPPORTED_JOB_TYPE"
        assert exc_info.value.retryable is False

    def test_execute_maps_failed_service_status(self) -> None:
        job_request = _ingest_request()
        transport = Mock()
        transport.submit.return_value = _job_status(
            job_id="service-job",
            state=JobState.FAILED,
            error=JobError(
                code="INGEST_FAILED",
                message="raw file missing",
                retryable=False,
            ),
        )
        executor = ServiceMlJobExecutor(
            transport=transport,
            endpoints={job_request.job_type: "http://ml-ingest:10081"},
            max_in_flight_jobs=1,
        )

        with pytest.raises(MlJobExecutionError) as exc_info:
            executor.execute(
                job_request,
                job_id="runner-job",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

        submitted_request = transport.submit.call_args.kwargs["job_request"]
        assert submitted_request.job_id == "runner-job"
        assert exc_info.value.code == "INGEST_FAILED"
        assert exc_info.value.message == "raw file missing"
        assert exc_info.value.retryable is False

    def test_execute_allows_bounded_parallel_service_dispatch(self) -> None:
        job_request = _ingest_request()
        first_call_started = threading.Event()
        release_calls = threading.Event()
        observed_in_flight = 0
        max_observed_in_flight = 0
        in_flight_lock = threading.Lock()
        observed_job_ids: set[str] = set()

        def submit(endpoint: str, job_request: IngestJobRequest) -> JobStatus:
            nonlocal max_observed_in_flight
            nonlocal observed_in_flight
            assert endpoint == "http://ml-ingest:10081"
            assert job_request.counter_id == "counter-a"
            assert job_request.job_id is not None
            observed_job_ids.add(job_request.job_id)
            with in_flight_lock:
                observed_in_flight += 1
                max_observed_in_flight = max(
                    max_observed_in_flight,
                    observed_in_flight,
                )
                first_call_started.set()
            release_calls.wait(timeout=5)
            with in_flight_lock:
                observed_in_flight -= 1
            return _job_status(
                job_id="service-job",
                state=JobState.SUCCEEDED,
                result=_job_result(job_id="service-job"),
            )

        transport = Mock()
        transport.submit.side_effect = submit
        executor = ServiceMlJobExecutor(
            transport=transport,
            endpoints={job_request.job_type: "http://ml-ingest:10081"},
            max_in_flight_jobs=2,
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                executor.execute,
                job_request,
                job_id="runner-job-a",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            assert first_call_started.wait(timeout=5)
            second = pool.submit(
                executor.execute,
                job_request,
                job_id="runner-job-b",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            time.sleep(0.05)
            release_calls.set()

            assert first.result().job_id == "runner-job-a"
            assert second.result().job_id == "runner-job-b"

        assert observed_job_ids == {"runner-job-a", "runner-job-b"}
        assert max_observed_in_flight == 2

    def test_execute_respects_configured_parallel_dispatch_limit(self) -> None:
        job_request = _ingest_request()
        first_call_started = threading.Event()
        release_calls = threading.Event()
        observed_in_flight = 0
        max_observed_in_flight = 0
        in_flight_lock = threading.Lock()

        def submit(endpoint: str, job_request: IngestJobRequest) -> JobStatus:
            nonlocal max_observed_in_flight
            nonlocal observed_in_flight
            assert endpoint == "http://ml-ingest:10081"
            assert job_request.counter_id == "counter-a"
            assert job_request.job_id in {"runner-job-a", "runner-job-b"}
            with in_flight_lock:
                observed_in_flight += 1
                max_observed_in_flight = max(
                    max_observed_in_flight,
                    observed_in_flight,
                )
                first_call_started.set()
            release_calls.wait(timeout=5)
            with in_flight_lock:
                observed_in_flight -= 1
            return _job_status(
                job_id="service-job",
                state=JobState.SUCCEEDED,
                result=_job_result(job_id="service-job"),
            )

        transport = Mock()
        transport.submit.side_effect = submit
        executor = ServiceMlJobExecutor(
            transport=transport,
            endpoints={job_request.job_type: "http://ml-ingest:10081"},
            max_in_flight_jobs=1,
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                executor.execute,
                job_request,
                job_id="runner-job-a",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            assert first_call_started.wait(timeout=5)
            second = pool.submit(
                executor.execute,
                job_request,
                job_id="runner-job-b",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            time.sleep(0.05)
            assert max_observed_in_flight == 1
            release_calls.set()

            assert first.result().job_id == "runner-job-a"
            assert second.result().job_id == "runner-job-b"

        assert max_observed_in_flight == 1


class TestUrllibMlServiceTransport:
    def test_submit_maps_http_error(self) -> None:
        transport = UrllibMlServiceTransport(timeout_seconds=1)
        error = HTTPError(
            url="http://service/jobs",
            code=500,
            msg="Server error",
            hdrs=Message(),
            fp=Mock(read=Mock(return_value=b"boom")),
        )

        with (
            patch("urllib.request.urlopen", side_effect=error),
            pytest.raises(MlJobExecutionError) as exc_info,
        ):
            transport.submit(
                endpoint="http://service",
                job_request=_ingest_request(),
            )

        assert exc_info.value.code == "ML_SERVICE_HTTP_ERROR"
        assert exc_info.value.retryable is True
        assert "boom" in exc_info.value.message

    def test_submit_maps_url_error(self) -> None:
        transport = UrllibMlServiceTransport(timeout_seconds=1)

        with (
            patch("urllib.request.urlopen", side_effect=URLError("offline")),
            pytest.raises(MlJobExecutionError) as exc_info,
        ):
            transport.submit(
                endpoint="http://service",
                job_request=_ingest_request(),
            )

        assert exc_info.value.code == "ML_SERVICE_UNAVAILABLE"
        assert exc_info.value.retryable is True
        assert "offline" in exc_info.value.message

    def test_submit_maps_invalid_response(self) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b"not-json"
        transport = UrllibMlServiceTransport(timeout_seconds=1)

        with (
            patch("urllib.request.urlopen", return_value=response),
            pytest.raises(MlJobExecutionError) as exc_info,
        ):
            transport.submit(
                endpoint="http://service",
                job_request=_ingest_request(),
            )

        assert exc_info.value.code == "ML_SERVICE_INVALID_RESPONSE"


class TestRunnerExecutorConfiguration:
    def test_build_default_executor_uses_service_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("JOB_RUNNER_EXECUTOR", raising=False)

        with patch("src.job_runner.service.ServiceMlJobExecutor") as executor:
            build_default_executor()

        executor.assert_called_once_with()

    def test_build_default_executor_uses_local_when_requested(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JOB_RUNNER_EXECUTOR", "local")

        with patch("src.job_runner.service.LocalMlJobExecutor") as executor:
            build_default_executor()

        executor.assert_called_once_with()

    def test_build_default_executor_rejects_unknown_executor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JOB_RUNNER_EXECUTOR", "docker")

        with pytest.raises(ValueError, match=r"local|service") as exc_info:
            build_default_executor()

        assert "JOB_RUNNER_EXECUTOR" in str(exc_info.value)

    def test_default_max_in_flight_jobs_is_conservative(self) -> None:
        assert DEFAULT_MAX_IN_FLIGHT_JOBS == 2

    def test_resolve_max_in_flight_jobs_reads_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(JOB_RUNNER_MAX_IN_FLIGHT_JOBS_ENV, "3")

        assert _resolve_max_in_flight_jobs() == 3

    def test_resolve_max_in_flight_jobs_rejects_invalid_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(JOB_RUNNER_MAX_IN_FLIGHT_JOBS_ENV, "0")

        with pytest.raises(ValueError, match=JOB_RUNNER_MAX_IN_FLIGHT_JOBS_ENV):
            _resolve_max_in_flight_jobs()


def test_join_url_normalizes_slashes() -> None:
    assert _join_url("http://service/", "/jobs") == "http://service/jobs"


def test_json_bytes_uses_compact_json() -> None:
    assert _json_bytes({"a": 1, "b": 2}) == b'{"a":1,"b":2}'


def _ingest_request() -> IngestJobRequest:
    return IngestJobRequest(
        run_id="run-a",
        counter_id="counter-a",
        raw_path="data/raw/source.csv",
        site="Site A",
        orientation="N-S",
        sub_dir="counter-a",
        interim_output_path="data/interim/counter-a/initial.csv",
        manifest_root="artifacts/manifests",
    )


def _job_result(job_id: str) -> JobResult:
    return JobResult(
        job_id=job_id,
        run_id="run-a",
        counter_id="counter-a",
        job_type=MlJobType.INGEST,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        finished_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        output_paths=("data/interim/counter-a/initial.csv",),
    )


def _job_status(
    *,
    job_id: str,
    state: JobState,
    result: JobResult | None = None,
    error: JobError | None = None,
) -> JobStatus:
    return JobStatus(
        job_id=job_id,
        run_id="run-a",
        counter_id="counter-a",
        job_type=MlJobType.INGEST,
        state=state,
        requested_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        result=result,
        error=error,
    )
