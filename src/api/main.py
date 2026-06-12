"""ASGI entrypoint for the bike traffic prediction serving API."""

from __future__ import annotations

from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.app import create_app
from src.common.env import get_env
from src.common.logger import configure_logging

configure_logging(level=get_env("LOG_LEVEL", default="INFO"))

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
