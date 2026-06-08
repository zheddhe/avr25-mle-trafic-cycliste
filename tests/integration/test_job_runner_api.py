"""Integration tests for the internal job runner API."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.job_runner.api import create_app
from src.job_runner.executor import MlJobExecutionError
from src.job_runner.service import JobRunnerService
from src.job_runner.state import InMemoryJobState
from src.ml.jobs.contracts import (
    ArtifactManifestReference,
    BaseMlJobRequest,
    MlJobType,
)
from src.ml.jobs.status import JobResult, MetricsEvidence


class SuccessfulExecutor:
    """Test executor returning deterministic step-level evidence."""

    def __init__(self) -> None:
        self.calls: list[BaseMlJobRequest] = []

    def execute(
        self,
        job_request: BaseMlJobRequest,
        *,
        job_id: str,
        started_at: datetime,
    ) -> JobResult:
        self.calls.append(job_request)
        return JobResult(
            job_id=job_id,
            run_id=job_request.run_id,
            counter_id=job_request.counter_id,
            job_type=job_request.job_type,
            started_at=started_at,
            finished_at=started_at,
            output_paths=("data/interim/Sebastopol_N-S_dvcrepro/initial.csv",),
            manifest=ArtifactManifestReference(
                artifact_type="interim_dataset",
                counter_id=job_request.counter_id,
                run_id=job_request.run_id,
                manifest_path=(
                    "docker/prod/runtime/artifacts/manifests/"
                    "interim_dataset/Sebastopol_N-S_dvcrepro/"
                    "manual-run-001/manifest.json"
                ),
                current_path=(
                    "docker/prod/runtime/artifacts/manifests/"
                    "interim_dataset/Sebastopol_N-S_dvcrepro/"
                    "current.json"
                ),
            ),
            metrics=MetricsEvidence(records=12, metrics_pushed=False),
        )


class FailingExecutor:
    """Test executor raising a controlled ML step failure."""

    def execute(
        self,
        job_request: BaseMlJobRequest,
        *,
        job_id: str,
        started_at: datetime,
    ) -> JobResult:
        raise MlJobExecutionError(
            code="INGEST_JOB_FAILED",
            message="Raw input file was not found.",
            retryable=True,
        )


@pytest.mark.integration
class TestJobRunnerApi:
    """Integration tests for typed job submission and status retrieval."""

    @pytest.fixture
    def executor(self) -> SuccessfulExecutor:
        return SuccessfulExecutor()

    @pytest.fixture
    def client(self, executor: SuccessfulExecutor) -> TestClient:
        service = JobRunnerService(
            state=InMemoryJobState(),
            executor=executor,
        )
        return TestClient(create_app(service=service))

    @pytest.fixture
    def failing_client(self) -> TestClient:
        service = JobRunnerService(
            state=InMemoryJobState(),
            executor=FailingExecutor(),
        )
        return TestClient(create_app(service=service))

    @pytest.fixture
    def ingest_payload(self) -> dict:
        return {
            "job_type": "ingest",
            "run_id": "manual-run-001",
            "counter_id": "Sebastopol_N-S_dvcrepro",
            "requested_at": "2026-06-07T17:00:00Z",
            "dag_id": "bike_init",
            "task_id": "ingest",
            "try_number": 1,
            "manifest_root": "docker/prod/runtime/artifacts/manifests",
            "raw_path": "data/raw/bike-counts.csv",
            "site": "Totem 73 boulevard de Sébastopol",
            "orientation": "N-S",
            "range_start": 0.0,
            "range_end": 75.0,
            "sub_dir": "Sebastopol_N-S_dvcrepro",
            "interim_output_path": (
                "data/interim/Sebastopol_N-S_dvcrepro/initial.csv"
            ),
        }

    def test_health_returns_healthy_response(self, client: TestClient):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "service": "job-runner-api",
        }

    def test_valid_job_submission_returns_succeeded_status(
        self,
        client: TestClient,
        executor: SuccessfulExecutor,
        ingest_payload: dict,
    ):
        response = client.post("/jobs", json=ingest_payload)

        assert response.status_code == 202
        body = response.json()
        assert body["job_id"].startswith("job-ingest-Sebastopol_N-S_dvcrepro-")
        assert body["run_id"] == ingest_payload["run_id"]
        assert body["counter_id"] == ingest_payload["counter_id"]
        assert body["job_type"] == "ingest"
        assert body["state"] == "succeeded"
        assert body["result"]["output_paths"] == [
            "data/interim/Sebastopol_N-S_dvcrepro/initial.csv",
        ]
        assert body["result"]["metrics"]["records"] == 12
        assert body["result"]["manifest"] == {
            "artifact_type": "interim_dataset",
            "counter_id": ingest_payload["counter_id"],
            "run_id": ingest_payload["run_id"],
            "manifest_path": (
                "docker/prod/runtime/artifacts/manifests/"
                "interim_dataset/Sebastopol_N-S_dvcrepro/"
                "manual-run-001/manifest.json"
            ),
            "current_path": (
                "docker/prod/runtime/artifacts/manifests/"
                "interim_dataset/Sebastopol_N-S_dvcrepro/current.json"
            ),
            "object_uri": None,
        }
        assert body["error"] is None
        assert len(executor.calls) == 1

    def test_valid_job_submission_is_idempotent(
        self,
        client: TestClient,
        executor: SuccessfulExecutor,
        ingest_payload: dict,
    ):
        first_response = client.post("/jobs", json=ingest_payload)
        second_response = client.post("/jobs", json=ingest_payload)

        assert first_response.status_code == 202
        assert second_response.status_code == 202
        assert second_response.json()["job_id"] == first_response.json()["job_id"]
        assert second_response.json()["state"] == "succeeded"
        assert len(executor.calls) == 1

    def test_invalid_job_submission_returns_validation_error(
        self,
        client: TestClient,
        ingest_payload: dict,
    ):
        payload = dict(ingest_payload)
        del payload["raw_path"]

        response = client.post("/jobs", json=payload)

        assert response.status_code == 422
        assert "raw_path" in response.text

    def test_pipeline_job_submission_is_not_exposed(
        self,
        client: TestClient,
        ingest_payload: dict,
    ):
        payload = dict(ingest_payload)
        payload["job_type"] = "pipeline"

        response = client.post("/jobs", json=payload)

        assert response.status_code == 422
        assert "pipeline" in response.text

    def test_status_retrieval_returns_current_terminal_status(
        self,
        client: TestClient,
        ingest_payload: dict,
    ):
        submit_response = client.post("/jobs", json=ingest_payload)
        job_id = submit_response.json()["job_id"]

        response = client.get(f"/jobs/{job_id}")

        assert response.status_code == 200
        assert response.json() == submit_response.json()

    def test_failed_step_returns_structured_job_error(
        self,
        failing_client: TestClient,
        ingest_payload: dict,
    ):
        response = failing_client.post("/jobs", json=ingest_payload)

        assert response.status_code == 202
        body = response.json()
        assert body["state"] == "failed"
        assert body["result"] is None
        assert body["error"] == {
            "code": "INGEST_JOB_FAILED",
            "message": "Raw input file was not found.",
            "retryable": True,
        }

    def test_status_retrieval_returns_explicit_not_found(
        self,
        client: TestClient,
    ):
        response = client.get("/jobs/unknown-job")

        assert response.status_code == 404
        assert response.json() == {
            "code": "JOB_NOT_FOUND",
            "message": "Job 'unknown-job' was not found.",
        }

    def test_openapi_exposes_health_and_jobs_tags(self, client: TestClient):
        response = client.get("/openapi.json")

        assert response.status_code == 200
        tag_names = {tag["name"] for tag in response.json()["tags"]}
        assert {"Health", "Jobs"}.issubset(tag_names)
