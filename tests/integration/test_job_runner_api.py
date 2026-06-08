"""Integration tests for the internal job runner API skeleton."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.job_runner.api import create_app
from src.job_runner.service import JobRunnerService
from src.job_runner.state import InMemoryJobState


@pytest.mark.integration
class TestJobRunnerApi:
    """Integration tests for typed job submission and status retrieval."""

    @pytest.fixture
    def client(self) -> TestClient:
        service = JobRunnerService(state=InMemoryJobState())
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

    def test_valid_job_submission_returns_typed_queued_status(
        self,
        client: TestClient,
        ingest_payload: dict,
    ):
        response = client.post("/jobs", json=ingest_payload)

        assert response.status_code == 202
        body = response.json()
        assert body["job_id"].startswith("job-ingest-Sebastopol_N-S_dvcrepro-")
        assert body["run_id"] == ingest_payload["run_id"]
        assert body["counter_id"] == ingest_payload["counter_id"]
        assert body["job_type"] == "ingest"
        assert body["state"] == "queued"
        assert body["result"] is None
        assert body["error"] is None

    def test_valid_job_submission_is_idempotent(
        self,
        client: TestClient,
        ingest_payload: dict,
    ):
        first_response = client.post("/jobs", json=ingest_payload)
        second_response = client.post("/jobs", json=ingest_payload)

        assert first_response.status_code == 202
        assert second_response.status_code == 202
        assert second_response.json()["job_id"] == first_response.json()["job_id"]

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

    def test_status_retrieval_returns_current_status(
        self,
        client: TestClient,
        ingest_payload: dict,
    ):
        submit_response = client.post("/jobs", json=ingest_payload)
        job_id = submit_response.json()["job_id"]

        response = client.get(f"/jobs/{job_id}")

        assert response.status_code == 200
        assert response.json() == submit_response.json()

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
