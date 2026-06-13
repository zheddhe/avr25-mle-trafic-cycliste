"""Unit tests for job runner FastAPI dependencies."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from fastapi import Request
from src.job_runner.dependencies import get_job_runner_service


class TestGetJobRunnerService:
    """Unit tests for get_job_runner_service."""

    def test_returns_application_scoped_service(self) -> None:
        service = object()
        request = cast(Request, MagicMock())
        request.app.state.job_runner_service = service

        assert get_job_runner_service(request) is service
