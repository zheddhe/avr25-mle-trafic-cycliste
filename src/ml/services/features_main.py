"""ASGI entrypoint for the internal features ML step service."""

from __future__ import annotations

from src.ml.jobs.contracts import MlJobType
from src.ml.services.api import create_app

app = create_app(service_name="ml-features-prod", job_type=MlJobType.FEATURES)
