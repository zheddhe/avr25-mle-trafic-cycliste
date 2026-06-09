"""FastAPI factory for the bike traffic prediction serving API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from src.api.auth import (
    check_admin_role,
    check_credentials,
    check_user_or_admin_role,
)
from src.api.config import ApiSettings, load_settings
from src.api.schemas import (
    AdminRefreshResponse,
    Counter,
    CurrentArtifactMetadata,
    ErrorResponse,
    PredictionItem,
    PredictionList,
)
from src.api.serving import PredictionServingError, load_predictions_from_manifests
from src.artifacts.exceptions import ArtifactManifestError

LOGGER = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {
        "name": "Admin",
        "description": "Restricted endpoints for administrators only.",
    },
    {
        "name": "Info",
        "description": "General informational authenticated endpoints.",
    },
    {
        "name": "Predictions",
        "description": "Prediction endpoints loaded from promoted manifests.",
    },
    {
        "name": "Artifacts",
        "description": "Sanitized metadata for currently served artifacts.",
    },
]

ResponsesDict = dict[int | str, dict[str, Any]]
GENERIC_RESPONSES: ResponsesDict = {
    200: {"description": "Success"},
    400: {"description": "Bad request content."},
    403: {"description": "Authentication failed."},
    404: {"description": "Unknown route."},
    418: {"description": "Business error.", "model": ErrorResponse},
    422: {"description": "Validation error."},
    500: {"description": "Server error."},
}


class ApiBusinessException(Exception):
    """Exception mapped to the API business error response."""

    def __init__(self, type: str, date: str, message: str | None) -> None:
        self.type = type
        self.date = date
        self.message = message


class PredictionStore:
    """In-memory prediction store populated from promoted manifests."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.predictions: dict[str, pd.DataFrame] = {}
        self.artifacts: dict[str, CurrentArtifactMetadata] = {}

    def refresh(self) -> None:
        """Reload current promoted manifests and replace the in-memory store."""

        result = load_predictions_from_manifests(self.settings)
        self.predictions = result.predictions
        self.artifacts = result.artifacts
        LOGGER.info(
            f"Store refreshed from manifests: {len(self.predictions)} "
            "counters available."
        )

    def require_predictions(self) -> dict[str, pd.DataFrame]:
        """Return loaded predictions or raise a structured API error."""

        if not self.predictions:
            raise ApiBusinessException(
                type="PredictionsNotLoaded",
                message="No promoted prediction manifest has been loaded.",
                date=str(datetime.now()),
            )
        return self.predictions

    def require_artifacts(self) -> dict[str, CurrentArtifactMetadata]:
        """Return loaded artifact metadata or raise a structured API error."""

        if not self.artifacts:
            raise ApiBusinessException(
                type="ArtifactsNotLoaded",
                message="No promoted prediction artifact metadata has been loaded.",
                date=str(datetime.now()),
            )
        return self.artifacts


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Create the prediction serving FastAPI application."""

    api_settings = settings or load_settings()
    prediction_store = PredictionStore(settings=api_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            prediction_store.refresh()
        except (ArtifactManifestError, PredictionServingError) as error:
            LOGGER.warning(f"Initial manifest refresh skipped: {error}")
        yield

    app = FastAPI(
        title="API du trafic cycliste",
        description=(
            "Expose les prédictions du trafic cycliste pour les compteurs "
            "installés dans la ville de Paris. Les prédictions sont servies "
            "depuis les manifests promus uniquement."
        ),
        version="1.3.0",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )
    app.state.prediction_store = prediction_store

    @app.exception_handler(ApiBusinessException)
    def custom_exception_handler(
        _request: Request,
        exception: ApiBusinessException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=418,
            content=ErrorResponse(
                type=exception.type,
                message=exception.message,
                date=exception.date,
            ).model_dump(),
        )

    @app.get(
        "/verify",
        tags=["Admin"],
        summary="Verify service health",
        description="Simple service health check. [ADMIN ONLY]",
        responses=GENERIC_RESPONSES,
    )
    def get_verify(
        user_info: dict[str, str] = Depends(check_admin_role),
    ) -> dict[str, str]:
        """Health check restricted to administrators."""

        LOGGER.info(
            f"Health check requested by admin user: {user_info['username']}"
        )
        return {
            "message": "API is healthy.",
            "checked_by": user_info["username"],
            "role": user_info["role"],
        }

    @app.get("/health", tags=["Info"], responses=GENERIC_RESPONSES)
    def get_health() -> dict[str, str]:
        """Healthcheck endpoint used by Docker Compose."""

        return {"status": "ok"}

    @app.post(
        "/admin/refresh",
        tags=["Admin"],
        summary="Refresh in-memory store from promoted manifests",
        description=(
            "Reload promoted prediction current.json manifests and their local "
            "payloads. [ADMIN ONLY]"
        ),
        response_model=AdminRefreshResponse,
        responses=GENERIC_RESPONSES,
    )
    def post_refresh(
        user_info: dict[str, str] = Depends(check_admin_role),
    ) -> AdminRefreshResponse:
        """Refresh the prediction store from promoted manifests."""

        before = len(prediction_store.predictions)
        try:
            prediction_store.refresh()
        except (ArtifactManifestError, PredictionServingError) as error:
            _raise_store_error(error)

        after = len(prediction_store.predictions)
        LOGGER.info(
            f"Admin refresh done by {user_info['username']}. "
            f"before={before} after={after} loaded={after}"
        )

        return AdminRefreshResponse(
            message=f"Store refreshed by {user_info['username']}.",
            counters_before=before,
            counters_after=after,
            manifest_root=str(api_settings.manifest_root),
            repository_root=str(api_settings.repository_root),
            loaded=after,
        )

    @app.get(
        "/counters",
        tags=["Predictions"],
        summary="List available counters",
        description=(
            "List counters loaded from promoted prediction manifests. "
            "[USER or ADMIN]"
        ),
        response_model=list[Counter],
        responses=GENERIC_RESPONSES,
    )
    def get_all_counters(
        user_info: dict[str, str] = Depends(check_user_or_admin_role),
    ) -> list[Counter]:
        """Return counters loaded in the manifest-first prediction store."""

        predictions = prediction_store.require_predictions()
        LOGGER.info(
            f"Counters list requested by user: {user_info['username']} "
            f"(role: {user_info['role']})"
        )
        return [Counter(id=name) for name in sorted(predictions.keys())]

    @app.get(
        "/predictions/{counter_id}",
        tags=["Predictions"],
        summary="Get predictions for a counter",
        description=(
            "Return a paginated list of predictions for the given counter id. "
            "Max 100 per page. [USER or ADMIN]"
        ),
        response_model=PredictionList,
        responses=GENERIC_RESPONSES,
    )
    def get_predictions_by_counter(
        counter_id: str,
        limit: int = Query(
            10,
            ge=1,
            le=100,
            description="Max number of predictions to return.",
        ),
        offset: int = Query(
            0,
            ge=0,
            description="Number of predictions to skip (pagination).",
        ),
        user_info: dict[str, str] = Depends(check_user_or_admin_role),
    ) -> PredictionList:
        """Return predictions for one loaded counter."""

        predictions = prediction_store.require_predictions()
        if counter_id not in predictions:
            available_counters = sorted(list(predictions.keys()))
            raise ApiBusinessException(
                type="CounterUnavailable",
                message=f"Available counters: {available_counters}",
                date=str(datetime.now()),
            )

        dataframe = predictions[counter_id]
        dataframe_page = dataframe.iloc[offset: offset + limit]
        LOGGER.info(
            f"Predictions for counter {counter_id} requested by user: "
            f"{user_info['username']} (role: {user_info['role']}, "
            f"limit: {limit}, offset: {offset})"
        )

        return PredictionList(
            total=int(len(dataframe)),
            limit=int(limit),
            offset=int(offset),
            item=[
                PredictionItem(**row)  # type: ignore[arg-type]
                for row in dataframe_page.to_dict(orient="records")
            ],
        )

    @app.get(
        "/artifacts/current",
        tags=["Artifacts"],
        summary="List currently served prediction artifact metadata",
        response_model=list[CurrentArtifactMetadata],
        responses=GENERIC_RESPONSES,
    )
    def get_current_artifacts(
        user_info: dict[str, str] = Depends(check_user_or_admin_role),
    ) -> list[CurrentArtifactMetadata]:
        """Return current artifact metadata for all loaded counters."""

        artifacts = prediction_store.require_artifacts()
        LOGGER.info(
            f"Current artifact metadata requested by user: "
            f"{user_info['username']} (role: {user_info['role']})"
        )
        return [artifacts[counter_id] for counter_id in sorted(artifacts.keys())]

    @app.get(
        "/artifacts/current/{counter_id}",
        tags=["Artifacts"],
        summary="Get current prediction artifact metadata for a counter",
        response_model=CurrentArtifactMetadata,
        responses=GENERIC_RESPONSES,
    )
    def get_current_artifact_by_counter(
        counter_id: str,
        user_info: dict[str, str] = Depends(check_user_or_admin_role),
    ) -> CurrentArtifactMetadata:
        """Return current artifact metadata for one loaded counter."""

        artifacts = prediction_store.require_artifacts()
        if counter_id not in artifacts:
            available_counters = sorted(list(artifacts.keys()))
            raise ApiBusinessException(
                type="ArtifactUnavailable",
                message=f"Available counters: {available_counters}",
                date=str(datetime.now()),
            )

        LOGGER.info(
            f"Current artifact metadata for counter {counter_id} requested "
            f"by user: {user_info['username']}"
        )
        return artifacts[counter_id]

    @app.get(
        "/me",
        tags=["Info"],
        summary="Get current user info",
        description="Get information about the currently authenticated user.",
        responses=GENERIC_RESPONSES,
    )
    def get_current_user(
        user_info: dict[str, str] = Depends(check_credentials),
    ) -> dict[str, Any]:
        """Return information about the authenticated user."""

        return {
            "username": user_info["username"],
            "role": user_info["role"],
            "permissions": {
                "admin_endpoints": user_info["role"] == "admin",
                "prediction_endpoints": user_info["role"] in ["user", "admin"],
            },
        }

    return app


def _raise_store_error(error: Exception) -> None:
    error_type = error.__class__.__name__
    raise ApiBusinessException(
        type=error_type,
        message=str(error),
        date=str(datetime.now()),
    ) from error
