"""ASGI entrypoint for the bike traffic prediction serving API."""

from __future__ import annotations

import logging
import os

from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.app import create_app

LOG_DIR = os.path.join("logs", "api")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "main.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

app = create_app()

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics"],
)
instrumentator.instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)

PRED_PER_RESPONSE = Histogram(
    "api_predictions_per_response",
    "Number of predictions returned by API response.",
    buckets=(1, 5, 10, 20, 50, 100, float("inf")),
)
