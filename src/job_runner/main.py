"""ASGI entrypoint for the internal job runner API."""

from __future__ import annotations

from src.common.env import get_env
from src.common.logger import build_service_log_file_path, configure_logging
from src.job_runner.api import create_app

configure_logging(
    level=get_env("LOG_LEVEL", default="INFO"),
    log_file_path=build_service_log_file_path(
        "job-runner",
        service_name=get_env("SERVICE_NAME", default="job-runner-api"),
        hostname=get_env("HOSTNAME", default="local"),
    ),
)

app = create_app()
