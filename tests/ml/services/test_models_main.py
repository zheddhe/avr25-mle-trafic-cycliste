"""Unit tests for the models ML service ASGI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from src.ml.services.models_main import app


class TestModelsMain:
    """Unit tests for the models service module-level ASGI app."""

    def test_app_is_fastapi_application(self) -> None:
        assert isinstance(app, FastAPI)
