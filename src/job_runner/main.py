"""ASGI entrypoint for the internal job runner API."""

from __future__ import annotations

from src.common.env import get_env
from src.common.logger import build_log_file_path, configure_logging
from src.job_runner.api import create_app

configure_logging(
    level=get_env("LOG_LEVEL", default="INFO"),
    log_file_path=build_log_file_path("job-runner", "main.log"),
)

app = create_app()
