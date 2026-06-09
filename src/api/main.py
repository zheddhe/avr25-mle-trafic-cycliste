# src/api/main.py
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.artifacts.exceptions import (
    ArtifactChecksumMismatchError,
    ArtifactManifestNotFoundError,
    ArtifactManifestValidationError,
    ArtifactPayloadNotFoundError,
)
from src.artifacts.manifest_store import read_current_manifest, verify_local_payload
from src.artifacts.schemas import ArtifactManifest, ArtifactType, StorageBackend

log_dir = os.path.join("logs", "api")
os.makedirs(log_dir, exist_ok=True)
LOG_PATH = os.path.join(log_dir, "main.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

ARTIFACT_MANIFEST_ROOT = os.path.abspath(
    os.getenv("ARTIFACT_MANIFEST_ROOT")
    or os.path.join("docker", "prod", "runtime", "artifacts", "manifests")
)
ARTIFACT_REPOSITORY_ROOT = os.path.abspath(
    os.getenv("ARTIFACT_REPOSITORY_ROOT") or "."
)
API_COUNTER_IDS = tuple(
    item.strip()
    for item in os.getenv("API_COUNTER_IDS", "").split(",")
    if item.strip()
)

# df_predictions: key = counter id, value = DataFrame loaded from the local CSV
# referenced by the promoted predictions current.json manifest.
df_predictions: dict[str, pd.DataFrame] = {}
prediction_artifacts: dict[str, "CurrentArtifactMetadata"] = {}

REQUIRED_COLUMNS = {
    "date_et_heure_de_comptage_local",
    "date_et_heure_de_comptage_utc",
    "y_true",
    "y_pred",
    "forecast_mode",
}


class PredictionServingError(ValueError):
    """Base error raised while loading promoted prediction artifacts."""


class PredictionCsvError(PredictionServingError):
    """Raised when the promoted prediction CSV cannot be loaded."""


class UnsupportedPredictionBackendError(PredictionServingError):
    """Raised when a promoted manifest points to an unsupported backend."""


@dataclass(frozen=True)
class PredictionServingConfig:
    """Runtime configuration for manifest-first prediction serving."""

    manifest_root: Path
    repository_root: Path
    counter_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionLoadResult:
    """Predictions and metadata loaded from promoted manifests."""

    predictions: dict[str, pd.DataFrame]
    artifacts: dict[str, "CurrentArtifactMetadata"]


def _default_credentials() -> dict[str, dict[str, str]]:
    """Return local demo credentials without embedding secret constants."""

    return {
        username: {"password": username, "role": role}
        for username, role in {
            "admin1": "admin",
            "admin2": "admin",
            "user1": "user",
            "user2": "user",
        }.items()
    }


dict_credentials = _default_credentials()
security = HTTPBasic()


def _check_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> dict:
    """Validate Basic Auth credentials and return sanitized user details."""

    if credentials.username not in dict_credentials:
        raise HTTPException(
            status_code=403,
            detail=f"Unknown user [{credentials.username}]",
        )

    user_info = dict_credentials[credentials.username]
    if user_info["password"] != credentials.password:
        raise HTTPException(status_code=403, detail="Invalid password.")

    return {"username": credentials.username, "role": user_info["role"]}


def _check_admin_role(user_info: dict = Depends(_check_credentials)) -> dict:
    """Ensure the current user has the admin role."""

    if user_info["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail=(
                "Access denied. Admin role required. "
                f"Current role: {user_info['role']}"
            ),
        )
    return user_info


def _check_user_or_admin_role(
    user_info: dict = Depends(_check_credentials),
) -> dict:
    """Ensure the current user has user or admin privileges."""

    if user_info["role"] not in ["user", "admin"]:
        raise HTTPException(
            status_code=403,
            detail=(
                "Access denied. User or Admin role required. "
                f"Current role: {user_info['role']}"
            ),
        )
    return user_info


tags_metadata = [
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

app = FastAPI(
    title="API du trafic cycliste",
    description=(
        "Expose les prédictions du trafic cycliste pour les compteurs "
        "installés dans la ville de Paris."
    ),
    version="1.3.0",
    openapi_tags=tags_metadata,
)

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
    "Nombre de prédictions renvoyées par appel",
    buckets=(1, 5, 10, 20, 50, 100, float("inf")),
)


class ErrorResponse(BaseModel):
    type: str = Field(..., description="Business error type.")
    message: str | None = Field(
        ..., description="Optional detailed error message."
    )
    date: str = Field(..., description="Server-side timestamp.")


class PredictionItem(BaseModel):
    date_et_heure_de_comptage_local: datetime = Field(
        ..., description="Local timestamp (Europe/Paris)."
    )
    date_et_heure_de_comptage_utc: datetime = Field(
        ..., description="UTC timestamp."
    )
    y_true: int = Field(..., description="Observed value.")
    y_pred: float = Field(..., description="Predicted value.")
    forecast_mode: bool = Field(
        ..., description="True if predicted on future timestamps."
    )


class PredictionList(BaseModel):
    total: int = Field(..., description="Total available predictions.")
    limit: int = Field(..., description="Max returned.")
    offset: int = Field(..., description="Pagination offset.")
    item: list[PredictionItem] = Field(
        ..., description="Paginated list of predictions."
    )


class Counter(BaseModel):
    id: str = Field(..., description="Counter identifier.")


class ArtifactSourceMetadata(BaseModel):
    raw_file_name: str | None = Field(default=None)
    dataset_version: str | None = Field(default=None)
    model_version: str | None = Field(default=None)


class CurrentArtifactMetadata(BaseModel):
    counter_id: str = Field(..., description="Counter served by the artifact.")
    run_id: str = Field(..., description="Run id recorded by the manifest.")
    artifact_type: str = Field(..., description="Manifest artifact type.")
    status: str = Field(..., description="Manifest lifecycle status.")
    created_at: datetime = Field(..., description="Manifest creation timestamp.")
    producer_service: str = Field(..., description="Producer service name.")
    producer_image: str | None = Field(default=None)
    producer_version: str | None = Field(default=None)
    source: ArtifactSourceMetadata = Field(
        ..., description="Sanitized source and lineage metadata."
    )
    primary_backend: str = Field(..., description="Primary storage backend.")
    local_path: str | None = Field(default=None)
    object_uri: str | None = Field(default=None)
    checksum_sha256: str | None = Field(default=None)


class AdminRefreshResponse(BaseModel):
    message: str = Field(..., description="Operation result.")
    counters_before: int = Field(..., description="Store size before refresh.")
    counters_after: int = Field(..., description="Store size after refresh.")
    manifest_root: str = Field(..., description="Manifest root used.")
    repository_root: str = Field(..., description="Repository root used.")
    loaded: int = Field(..., description="Counters successfully loaded.")


def get_prediction_serving_config() -> PredictionServingConfig:
    """Build manifest-first API serving configuration from environment."""

    return PredictionServingConfig(
        manifest_root=Path(ARTIFACT_MANIFEST_ROOT),
        repository_root=Path(ARTIFACT_REPOSITORY_ROOT),
        counter_ids=API_COUNTER_IDS,
    )


def discover_current_prediction_counter_ids(manifest_root: Path) -> tuple[str, ...]:
    """Discover counters that have a promoted prediction current manifest."""

    predictions_root = manifest_root / ArtifactType.PREDICTIONS.value
    if not predictions_root.is_dir():
        return ()

    counter_ids = [
        path.parent.name
        for path in predictions_root.glob("*/current.json")
        if path.is_file()
    ]
    return tuple(sorted(counter_ids))


def load_predictions_from_manifests(
    config: PredictionServingConfig,
) -> PredictionLoadResult:
    """Load promoted prediction manifests and referenced local CSV files."""

    counter_ids = config.counter_ids or discover_current_prediction_counter_ids(
        config.manifest_root,
    )
    predictions: dict[str, pd.DataFrame] = {}
    artifacts: dict[str, CurrentArtifactMetadata] = {}

    for counter_id in counter_ids:
        manifest = read_current_manifest(
            manifest_root=config.manifest_root,
            artifact_type=ArtifactType.PREDICTIONS.value,
            counter_id=counter_id,
        )
        df = load_prediction_dataframe_from_manifest(manifest, config)
        predictions[manifest.counter_id] = df
        artifacts[manifest.counter_id] = current_artifact_metadata(manifest)
        logger.info(
            "Loaded promoted predictions for counter [%s]: %s rows x %s cols",
            manifest.counter_id,
            df.shape[0],
            df.shape[1],
        )

    return PredictionLoadResult(predictions=predictions, artifacts=artifacts)


def load_prediction_dataframe_from_manifest(
    manifest: ArtifactManifest,
    config: PredictionServingConfig,
) -> pd.DataFrame:
    """Validate a prediction manifest and load its referenced local CSV."""

    if manifest.artifact_type != ArtifactType.PREDICTIONS:
        raise ArtifactManifestValidationError(
            "Current artifact manifest must describe predictions."
        )
    if manifest.storage.primary_backend != StorageBackend.LOCAL:
        raise UnsupportedPredictionBackendError(
            "Unsupported prediction artifact backend: "
            f"{manifest.storage.primary_backend.value}"
        )
    if manifest.storage.local_path is None:
        raise ArtifactManifestValidationError(
            "local_path is required for local prediction serving."
        )

    verify_local_payload(manifest, repository_root=config.repository_root)
    csv_path = config.repository_root / manifest.storage.local_path
    return read_prediction_csv(csv_path)


def read_prediction_csv(csv_path: Path) -> pd.DataFrame:
    """Read and validate a promoted prediction CSV file."""

    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as error:
        raise PredictionCsvError(
            f"Failed to read promoted prediction CSV [{csv_path}]: {error}"
        ) from error

    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise PredictionCsvError(
            "Promoted prediction CSV is missing required columns: "
            f"{missing}"
        )

    return df


def current_artifact_metadata(
    manifest: ArtifactManifest,
) -> CurrentArtifactMetadata:
    """Return sanitized metadata for the currently served artifact."""

    return CurrentArtifactMetadata(
        counter_id=manifest.counter_id,
        run_id=manifest.run_id,
        artifact_type=manifest.artifact_type.value,
        status=manifest.status.value,
        created_at=manifest.created_at,
        producer_service=manifest.producer.service,
        producer_image=manifest.producer.image,
        producer_version=manifest.producer.version,
        source=ArtifactSourceMetadata(
            raw_file_name=manifest.source.raw_file_name,
            dataset_version=manifest.source.dataset_version,
            model_version=manifest.source.model_version,
        ),
        primary_backend=manifest.storage.primary_backend.value,
        local_path=manifest.storage.local_path,
        object_uri=manifest.storage.object_uri,
        checksum_sha256=manifest.storage.checksum_sha256,
    )


def refresh_store() -> dict[str, pd.DataFrame]:
    """Refresh the in-memory store from promoted prediction manifests."""

    global df_predictions, prediction_artifacts
    config = get_prediction_serving_config()
    result = load_predictions_from_manifests(config)
    df_predictions = result.predictions
    prediction_artifacts = result.artifacts
    logger.info(
        "Store refreshed from manifests: %s counters available.",
        len(df_predictions),
    )
    return df_predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI."""

    refresh_store()
    yield


class CustomException(Exception):
    def __init__(self, type: str, date: str, message: str):
        self.type = type
        self.date = date
        self.message = message


@app.exception_handler(CustomException)
def custom_exception_handler(
    _request: Request,
    exception: CustomException,
):
    return JSONResponse(
        status_code=418,
        content=ErrorResponse(
            type=exception.type,
            message=exception.message,
            date=exception.date,
        ).model_dump(),
    )


ResponsesDict = dict[int | str, dict[str, Any]]
generic_responses: ResponsesDict = {
    200: {"description": "Success"},
    400: {"description": "Bad request content."},
    403: {"description": "Authentication failed."},
    404: {"description": "Unknown route."},
    418: {"description": "Business error.", "model": ErrorResponse},
    422: {"description": "Validation error."},
    500: {"description": "Server error."},
}


def _raise_store_error(error: Exception) -> None:
    error_type = error.__class__.__name__
    raise CustomException(
        type=error_type,
        message=str(error),
        date=str(datetime.now()),
    ) from error


@app.get(
    "/verify",
    tags=["Admin"],
    summary="Verify service health",
    description="Simple service health check. [ADMIN ONLY]",
    responses=generic_responses,
)
def get_verify(user_info: dict = Depends(_check_admin_role)):
    """Health check restricted to administrators."""

    logger.info("Health check requested by admin user: %s", user_info["username"])
    return {
        "message": "API is healthy.",
        "checked_by": user_info["username"],
        "role": user_info["role"],
    }


@app.post(
    "/admin/refresh",
    tags=["Admin"],
    summary="Refresh in-memory store from promoted manifests",
    description=(
        "Reload promoted prediction current.json manifests and their local "
        "payloads. [ADMIN ONLY]"
    ),
    response_model=AdminRefreshResponse,
    responses=generic_responses,
)
def post_refresh(user_info: dict = Depends(_check_admin_role)):
    """Refresh the prediction store from promoted manifests."""

    global df_predictions, prediction_artifacts
    before = len(df_predictions)
    config = get_prediction_serving_config()

    try:
        result = load_predictions_from_manifests(config)
    except (
        ArtifactChecksumMismatchError,
        ArtifactManifestNotFoundError,
        ArtifactManifestValidationError,
        ArtifactPayloadNotFoundError,
        PredictionServingError,
    ) as error:
        _raise_store_error(error)

    df_predictions = result.predictions
    prediction_artifacts = result.artifacts
    after = len(df_predictions)

    logger.info(
        "Admin refresh done by %s. before=%s after=%s loaded=%s",
        user_info["username"],
        before,
        after,
        after,
    )

    return AdminRefreshResponse(
        message=f"Store refreshed by {user_info['username']}.",
        counters_before=before,
        counters_after=after,
        manifest_root=str(config.manifest_root),
        repository_root=str(config.repository_root),
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
    responses=generic_responses,
)
def get_all_counters(user_info: dict = Depends(_check_user_or_admin_role)):
    """Return counters loaded in the manifest-first prediction store."""

    if not df_predictions:
        raise CustomException(
            type="PredictionsNotLoaded",
            message="No promoted prediction manifest has been loaded.",
            date=str(datetime.now()),
        )

    logger.info(
        "Counters list requested by user: %s (role: %s)",
        user_info["username"],
        user_info["role"],
    )
    return [Counter(id=name) for name in sorted(df_predictions.keys())]


@app.get(
    "/predictions/{counter_id}",
    tags=["Predictions"],
    summary="Get predictions for a counter",
    description=(
        "Return a paginated list of predictions for the given counter id. "
        "Max 100 per page. [USER or ADMIN]"
    ),
    response_model=PredictionList,
    responses=generic_responses,
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
    user_info: dict = Depends(_check_user_or_admin_role),
):
    """Return predictions for one loaded counter."""

    if not df_predictions:
        raise CustomException(
            type="PredictionsNotLoaded",
            message="No promoted prediction manifest has been loaded.",
            date=str(datetime.now()),
        )

    if counter_id not in df_predictions:
        raise CustomException(
            type="CounterUnavailable",
            message=(
                "Available counters: "
                f"{sorted(list(df_predictions.keys()))}"
            ),
            date=str(datetime.now()),
        )

    df = df_predictions[counter_id]
    df_page = df.iloc[offset: offset + limit]

    try:
        PRED_PER_RESPONSE.observe(float(len(df_page)))
    except Exception as error:
        logger.debug("Failed to observe PRED_PER_RESPONSE: %s", error)

    logger.info(
        "Predictions for counter %s requested by user: %s "
        "(role: %s, limit: %s, offset: %s)",
        counter_id,
        user_info["username"],
        user_info["role"],
        limit,
        offset,
    )

    return PredictionList(
        total=int(len(df)),
        limit=int(limit),
        offset=int(offset),
        item=[
            PredictionItem(**row)  # type: ignore
            for row in df_page.to_dict(orient="records")
        ],
    )


@app.get(
    "/artifacts/current",
    tags=["Artifacts"],
    summary="List currently served prediction artifact metadata",
    response_model=list[CurrentArtifactMetadata],
    responses=generic_responses,
)
def get_current_artifacts(user_info: dict = Depends(_check_user_or_admin_role)):
    """Return current artifact metadata for all loaded counters."""

    if not prediction_artifacts:
        raise CustomException(
            type="ArtifactsNotLoaded",
            message="No promoted prediction artifact metadata has been loaded.",
            date=str(datetime.now()),
        )

    logger.info(
        "Current artifact metadata requested by user: %s (role: %s)",
        user_info["username"],
        user_info["role"],
    )
    return [
        prediction_artifacts[counter_id]
        for counter_id in sorted(prediction_artifacts.keys())
    ]


@app.get(
    "/artifacts/current/{counter_id}",
    tags=["Artifacts"],
    summary="Get current prediction artifact metadata for a counter",
    response_model=CurrentArtifactMetadata,
    responses=generic_responses,
)
def get_current_artifact_by_counter(
    counter_id: str,
    user_info: dict = Depends(_check_user_or_admin_role),
):
    """Return current artifact metadata for one loaded counter."""

    if not prediction_artifacts:
        raise CustomException(
            type="ArtifactsNotLoaded",
            message="No promoted prediction artifact metadata has been loaded.",
            date=str(datetime.now()),
        )

    if counter_id not in prediction_artifacts:
        raise CustomException(
            type="ArtifactUnavailable",
            message=(
                "Available counters: "
                f"{sorted(list(prediction_artifacts.keys()))}"
            ),
            date=str(datetime.now()),
        )

    logger.info(
        "Current artifact metadata for counter %s requested by user: %s",
        counter_id,
        user_info["username"],
    )
    return prediction_artifacts[counter_id]


@app.get(
    "/me",
    tags=["Info"],
    summary="Get current user info",
    description="Get information about the currently authenticated user.",
    responses=generic_responses,
)
def get_current_user(user_info: dict = Depends(_check_credentials)):
    """Return information about the authenticated user."""

    return {
        "username": user_info["username"],
        "role": user_info["role"],
        "permissions": {
            "admin_endpoints": user_info["role"] == "admin",
            "prediction_endpoints": user_info["role"] in ["user", "admin"],
        },
    }
