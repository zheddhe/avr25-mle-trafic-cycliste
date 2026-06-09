"""ASGI entrypoint for the internal models ML step service."""

from __future__ import annotations

from src.ml.jobs.contracts import MlJobType
from src.ml.services.api import create_app

app = create_app(service_name="ml-models-prod", job_type=MlJobType.MODELS)
