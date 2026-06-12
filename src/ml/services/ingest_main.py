"""ASGI entrypoint for the internal ingest ML step service."""

from __future__ import annotations

from src.common.env import get_env
from src.common.logger import configure_logging
from src.ml.jobs.contracts import MlJobType
from src.ml.services.api import create_app

configure_logging(level=get_env("LOG_LEVEL", default="INFO"))

app = create_app(service_name="ml-ingest-prod", job_type=MlJobType.INGEST)
