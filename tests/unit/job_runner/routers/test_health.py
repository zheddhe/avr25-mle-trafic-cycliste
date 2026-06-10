"""Unit tests for job runner health router."""

from __future__ import annotations

from src.job_runner.routers.health import get_health


class TestGetHealth:
    """Unit tests for get_health."""

    def test_returns_healthy_runner_status(self) -> None:
        response = get_health()

        assert response.status == "healthy"
        assert response.service == "job-runner-api"
