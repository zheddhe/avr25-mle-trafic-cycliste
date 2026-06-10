"""Unit tests for the features ML service ASGI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from src.ml.services.features_main import app


class TestFeaturesMain:
    """Unit tests for the features service module-level ASGI app."""

    def test_app_is_fastapi_application(self) -> None:
        assert isinstance(app, FastAPI)
