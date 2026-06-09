# tests/api/test_main.py
from __future__ import annotations

from fastapi import FastAPI

from src.api.main import app


class TestApiMain:
    def test_main_exposes_fastapi_app(self) -> None:
        assert isinstance(app, FastAPI)
        assert app.title == "API du trafic cycliste"
