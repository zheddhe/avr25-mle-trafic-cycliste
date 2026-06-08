"""ASGI entrypoint for the internal job runner API."""

from __future__ import annotations

from src.job_runner.api import create_app

app = create_app()
