"""FastAPI factory for the bike traffic prediction serving API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.api.auth import check_admin_role, check_user_or_admin_role
from src.api.config import ApiSettings, load_settings
from src.api.schemas import (
    AdminRefreshResponse,
    ArtifactSourceMetadata,
    Counter,
    CurrentArtifactMetadata,
    CustomException,
    Prediction,
    PredictionInput,
)
from src.api.serving import (
    PredictionCsvError,
    PredictionServingError,
    load_predictions_from_manifests,
)
from src.artifacts.exceptions import ArtifactManifestError

OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Service liveness endpoint for Compose healthchecks.",
    },
    {
        "name": "Predictions",
        "description": "Manifest-first bike traffic prediction serving endpoints.",
    },
    {
        "name": "Artifacts",
        "description": "Currently served promoted artifact metadata.",
    },
    {
        "name": "Admin",
        "description": "Operational refresh endpoints for promoted manifests.",
    },
]


class PredictionStore:
    """In-memory prediction store populated from promoted manifests."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.dataframe = pd.DataFrame()
        self.artifacts: dict[str, ArtifactSourceMetadata] = {}

    def refresh(self) -> AdminRefreshResponse:
        """Reload current promoted manifests and replace the in-memory store."""

        result = load_predictions_from_manifests(self.settings)
        self.dataframe = result.dataframe
        self.artifacts = result.artifacts
        return AdminRefreshResponse(
            Status="ok",
            ManifestRoot=str(self.settings.manifest_root),
            RepositoryRoot=str(self.settings.repository_root),
            Loaded=len(self.artifacts),
            Artifacts=list(self.artifacts.values()),
        )

    def require_predictions(self) -> pd.DataFrame:
        """Return loaded predictions or raise a structured API error."""

        if self.dataframe.empty:
            raise CustomException(
                ErrorId=404,
                ErrorType="PredictionsNotLoaded",
                ErrorMessage="No prediction dataframe is currently loaded.",
            )
        return self.dataframe

    def require_artifacts(self) -> dict[str, ArtifactSourceMetadata]:
        """Return loaded artifact metadata or raise a structured API error."""

        if not self.artifacts:
            raise CustomException(
                ErrorId=404,
                ErrorType="ArtifactsNotLoaded",
                ErrorMessage="No promoted prediction artifact is loaded.",
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
        except (ArtifactManifestError, PredictionServingError):
            pass
        yield

    app = FastAPI(
        title="Bike Traffic Prediction API",
        description=(
            "Serve bike traffic predictions from promoted artifact manifests. "
            "The API never scans final data folders implicitly."
        ),
        version="1.0.0",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )
    app.state.prediction_store = prediction_store

    @app.exception_handler(CustomException)
    async def custom_exception_handler(
        request: Request,
        exc: CustomException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=418,
            content=exc.model_dump(by_alias=True),
        )

    @app.get("/health", tags=["Health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/admin/refresh",
        response_model=AdminRefreshResponse,
        tags=["Admin"],
    )
    def admin_refresh(
        user_info: dict[str, str] = Depends(check_admin_role),
    ) -> AdminRefreshResponse:
        try:
            return prediction_store.refresh()
        except (ArtifactManifestError, PredictionServingError) as error:
            raise CustomException(
                ErrorId=500,
                ErrorType=error.__class__.__name__,
                ErrorMessage=str(error),
            ) from error

    @app.get(
        "/counters",
        response_model=list[Counter],
        tags=["Predictions"],
    )
    def list_counters(
        user_info: dict[str, str] = Depends(check_user_or_admin_role),
    ) -> list[Counter]:
        dataframe = prediction_store.require_predictions()
        return [
            Counter(CounterId=counter_id)
            for counter_id in sorted(dataframe["counter_id"].dropna().unique())
        ]

    @app.post(
        "/predict",
        response_model=list[Prediction],
        tags=["Predictions"],
    )
    def get_predictions(
        payload: PredictionInput,
        user_info: dict[str, str] = Depends(check_user_or_admin_role),
    ) -> list[Prediction]:
        dataframe = prediction_store.require_predictions()
        counter_dataframe = dataframe[dataframe["counter_id"] == payload.counter_id]
        if counter_dataframe.empty:
            raise CustomException(
                ErrorId=404,
                ErrorType="CounterNotFound",
                ErrorMessage=f"Counter not found: {payload.counter_id}",
            )

        date_column = "date_et_heure_de_comptage_local"
        filtered_dataframe = counter_dataframe[
            (counter_dataframe[date_column] >= payload.start_date)
            & (counter_dataframe[date_column] <= payload.end_date)
        ]
        if filtered_dataframe.empty:
            raise CustomException(
                ErrorId=404,
                ErrorType="PredictionRangeNotFound",
                ErrorMessage=(
                    "No predictions found for "
                    f"{payload.counter_id} between {payload.start_date} "
                    f"and {payload.end_date}."
                ),
            )

        return [_build_prediction(row) for row in filtered_dataframe.to_dict("records")]

    @app.get(
        "/artifacts/current",
        response_model=CurrentArtifactMetadata,
        tags=["Artifacts"],
    )
    def list_current_artifacts(
        user_info: dict[str, str] = Depends(check_user_or_admin_role),
    ) -> CurrentArtifactMetadata:
        artifacts = prediction_store.require_artifacts()
        return CurrentArtifactMetadata(Artifacts=list(artifacts.values()))

    @app.get(
        "/artifacts/current/{counter_id}",
        response_model=ArtifactSourceMetadata,
        tags=["Artifacts"],
    )
    def get_current_artifact(
        counter_id: str,
        user_info: dict[str, str] = Depends(check_user_or_admin_role),
    ) -> ArtifactSourceMetadata:
        artifacts = prediction_store.require_artifacts()
        if counter_id not in artifacts:
            raise CustomException(
                ErrorId=404,
                ErrorType="ArtifactNotFound",
                ErrorMessage=f"No current prediction artifact for {counter_id}.",
            )
        return artifacts[counter_id]

    return app


def _build_prediction(row: dict[str, Any]) -> Prediction:
    y_true = row.get("y_true")
    if pd.isna(y_true):
        y_true = None

    return Prediction(
        CounterId=row["counter_id"],
        Date=row["date_et_heure_de_comptage_local"],
        YTrue=y_true,
        YPred=float(row["y_pred"]),
        ForecastMode=bool(row["forecast_mode"]),
    )
