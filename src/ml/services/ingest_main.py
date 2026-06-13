"""ASGI entrypoint for the internal ingest ML step service."""

from __future__ import annotations

from src.common.env import get_env
from src.common.logger import build_service_log_file_path, configure_logging
from src.ml.jobs.contracts import MlJobType
from src.ml.services.api import create_app

SERVICE_NAME = get_env("SERVICE_NAME", default="ml-ingest")
HOSTNAME = get_env("HOSTNAME", default="local")

configure_logging(
    level=get_env("LOG_LEVEL", default="INFO"),
    log_file_path=build_service_log_file_path(
        "ml",
        "ingest",
        service_name=SERVICE_NAME,
        hostname=HOSTNAME,
    ),
)

app = create_app(service_name=SERVICE_NAME, job_type=MlJobType.INGEST)
