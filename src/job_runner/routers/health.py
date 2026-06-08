"""Health endpoint for the internal job runner API."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Simple service health response."""

    status: Literal["healthy"] = Field(
        description="Current service health status.",
    )
    service: Literal["job-runner-api"] = Field(
        description="Internal service name.",
    )


router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check job runner API health",
)
def get_health() -> HealthResponse:
    """Return a simple healthy response for Compose healthchecks."""

    return HealthResponse(
        status="healthy",
        service="job-runner-api",
    )
