"""Unit tests for the ingest ML service ASGI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from src.ml.services.ingest_main import app


class TestIngestMain:
    """Unit tests for the ingest service module-level ASGI app."""

    def test_app_is_fastapi_application(self) -> None:
        assert isinstance(app, FastAPI)
