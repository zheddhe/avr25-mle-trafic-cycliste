"""Unit tests for job runner FastAPI application factory."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.job_runner.api import create_app


class TestCreateApp:
    """Unit tests for create_app."""

    def test_app_exposes_health_endpoint_and_openapi_tags(self) -> None:
        client = TestClient(create_app())

        response = client.get("/health")
        schema_response = client.get("/openapi.json")

        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "service": "job-runner-api",
        }
        assert schema_response.status_code == 200
        assert "/jobs" in schema_response.json()["paths"]
