"""Unit tests for job runner ASGI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from src.job_runner.main import app


class TestJobRunnerMain:
    """Unit tests for the module-level ASGI app."""

    def test_app_is_fastapi_application(self) -> None:
        assert isinstance(app, FastAPI)
