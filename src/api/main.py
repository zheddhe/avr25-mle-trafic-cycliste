# src/api/main.py
from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# Logs configuration
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Data root and in-memory store
# -------------------------------------------------------------------
DATA_FINAL_ROOT = os.getenv("DATA_FINAL_ROOT", os.path.join("data", "final"))

# df_predictions: key = subdir (counter id), value = DataFrame loaded from
# <DATA_FINAL_ROOT>/<subdir>/y_full.csv
df_predictions: Dict[str, pd.DataFrame] = {}

REQUIRED_COLUMNS = {
    "date_et_heure_de_comptage_local",
    "date_et_heure_de_comptage_utc",
    "y_true",
    "y_pred",
    "forecast_mode",
}


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Read a CSV with index_col=0 and validate required columns.
    Returns None if the file is invalid or unreadable.
    """
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        logger.warning("Failed to read CSV [%s]: %s", path, exc)
        return None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(
            f"CSV [{path}] ignored, missing required columns: {missing}"
        )
        return None

    return df


def load_predictions_from_final(
    root_dir: str,
) -> tuple[Dict[str, pd.DataFrame], int, int]:
    """
    Explore <root_dir> to find all subdirs containing a y_full.csv.
    Return (mapping, skipped_no_csv, invalid_csv).
    """
    mapping: Dict[str, pd.DataFrame] = {}
    skipped_no_csv = 0
    invalid_csv = 0

    if not os.path.isdir(root_dir):
        logger.warning(f"Final data root not found: [{root_dir}]")
        return mapping, skipped_no_csv, invalid_csv

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        subdir = entry.name
        csv_path = os.path.join(entry.path, "y_full.csv")
        if not os.path.isfile(csv_path):
            logger.debug(f"No y_full.csv in [{entry.path}], skipping.")
            skipped_no_csv += 1
            continue

        df = _safe_read_csv(csv_path)
        if df is None:
            invalid_csv += 1
            continue

        mapping[subdir] = df
        logger.info(
            f"Loaded predictions for counter [{subdir}]: {df.shape[0]} rows x [{df.shape[1]}] cols"
        )
    return mapping, skipped_no_csv, invalid_csv


def refresh_store() -> Dict[str, pd.DataFrame]:
    """
    Rescan the filesystem and refresh the in-memory store.
    """
    global df_predictions
    mapping, _, _ = load_predictions_from_final(DATA_FINAL_ROOT)
    df_predictions = mapping
    logger.info(f"Store refreshed: {len(df_predictions)} counters available.")
    return df_predictions


# -------------------------------------------------------------------
# Simple security (Basic Auth) avec système de rôles
# -------------------------------------------------------------------

# Dictionnaire des utilisateurs avec leurs rôles
dict_credentials = {
    # Administrateurs - accès complet
    "remy": {"password": "remy", "role": "admin"},
    "elias": {"password": "elias", "role": "admin"},
    "kolade": {"password": "kolade", "role": "admin"},
    "sofia": {"password": "sofia", "role": "admin"},

    # Utilisateurs standard - accès prédictions uniquement
    "user1": {"password": "user1", "role": "user"},
    "user2": {"password": "user2", "role": "user"},
}

security = HTTPBasic()


def _check_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> dict:
    """
    Validate Basic Auth credentials et retourne les infos utilisateur.
    """
    if credentials.username not in dict_credentials:
        raise HTTPException(
            status_code=403,
            detail=f"Unknown user [{credentials.username}]",
        )

    user_info = dict_credentials[credentials.username]
    if user_info["password"] != credentials.password:
        raise HTTPException(
            status_code=403,
            detail="Invalid password.",
        )

    # Retourner les informations utilisateur (sans le mot de passe)
    return {
        "username": credentials.username,
        "role": user_info["role"]
    }


def _check_admin_role(user_info: dict = Depends(_check_credentials)) -> dict:
    """
    Vérifie que l'utilisateur a le rôle admin.
    """
    if user_info["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_info['role']}",
        )
    return user_info


def _check_user_or_admin_role(user_info: dict = Depends(_check_credentials)) -> dict:
    """
    Vérifie que l'utilisateur a le rôle user ou admin.
    """
    if user_info["role"] not in ["user", "admin"]:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. User or Admin role required. Current role: {user_info['role']}",
        )
    return user_info


# -------------------------------------------------------------------
# OpenAPI / tags
# -------------------------------------------------------------------
tags_metadata = [
    {"name": "Admin", "description": "Service health and maintenance."},
    {
        "name": "Predictions",
        "description": "Access cyclist traffic predictions.",
    },
]

app = FastAPI(
    title="API du trafic cycliste",
    description=(
        "Expose les prédictions du trafic cycliste pour les compteurs "
        "installés dans la ville de Paris."
    ),
    version="1.1.0",
    openapi_tags=tags_metadata,
)


# -------------------------------------------------------------------
# Lifespan context
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI.
    - On startup: refresh the store.
    - On shutdown: (placeholder for cleanup if needed).
    """
    refresh_store()
    yield
    # add cleaning steps if necessary


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class ErrorResponse(BaseModel):
    type: str = Field(..., description="Business error type.")
    message: Optional[str] = Field(
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
    item: List[PredictionItem] = Field(
        ..., description="Paginated list of predictions."
    )


class Counter(BaseModel):
    id: str = Field(..., description="Counter identifier (subdir name).")


class AdminRefreshResponse(BaseModel):
    message: str = Field(..., description="Operation result.")
    counters_before: int = Field(..., description="Store size before refresh.")
    counters_after: int = Field(..., description="Store size after refresh.")
    data_root: str = Field(..., description="Absolute root directory used.")
    loaded: int = Field(..., description="Counters successfully loaded.")
    skipped_no_csv: int = Field(
        ..., description="Subdirs without y_full.csv."
    )
    invalid_csv: int = Field(
        ..., description="y_full.csv unreadable or schema-missing."
    )


# -------------------------------------------------------------------
# Custom exception + handler
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Common responses
# -------------------------------------------------------------------
ResponsesDict = Dict[Union[int, str], Dict[str, Any]]
generic_responses: ResponsesDict = {
    200: {"description": "Success"},
    400: {"description": "Bad request content."},
    403: {"description": "Authentication failed."},
    404: {"description": "Unknown route."},
    418: {"description": "Business error.", "model": ErrorResponse},
    422: {"description": "Validation error."},
    500: {"description": "Server error."},
}


# -------------------------------------------------------------------
# Routes avec contrôle d'accès par rôle
# -------------------------------------------------------------------

@app.get(
    "/verify",
    tags=["Admin"],
    summary="Verify service health",
    description="Simple service health check. [ADMIN ONLY]",
    responses=generic_responses,
)
def get_verify(user_info: dict = Depends(_check_admin_role)):
    """
    Health check - Accès limité aux administrateurs
    """
    logger.info(f"Health check requested by admin user: {user_info['username']}")
    return {
        "message": "API is healthy.",
        "checked_by": user_info["username"],
        "role": user_info["role"]
    }


@app.post(
    "/admin/refresh",
    tags=["Admin"],
    summary="Refresh in-memory store",
    description="Rescan the filesystem (DATA_FINAL_ROOT) and reload all counters. [ADMIN ONLY]",
    response_model=AdminRefreshResponse,
    responses=generic_responses,
)
def post_refresh(user_info: dict = Depends(_check_admin_role)):
    """
    Refresh du store - Accès limité aux administrateurs
    """
    global df_predictions
    before = len(df_predictions)
    mapping, skipped_no_csv, invalid_csv = load_predictions_from_final(
        DATA_FINAL_ROOT
    )
    df_predictions = mapping
    after = len(df_predictions)

    logger.info(
        f"Admin refresh done by {user_info['username']}. "
        f"before={before} after={after} loaded={after} "
        f"skipped_no_csv={skipped_no_csv} invalid_csv={invalid_csv}"
    )

    return AdminRefreshResponse(
        message=f"Store refreshed by {user_info['username']}.",
        counters_before=before,
        counters_after=after,
        data_root=os.path.abspath(DATA_FINAL_ROOT),
        loaded=after,
        skipped_no_csv=skipped_no_csv,
        invalid_csv=invalid_csv,
    )


@app.get(
    "/counters",
    tags=["Predictions"],
    summary="List available counters",
    description=(
        "List all counters detected under data/final/<subdir>/y_full.csv "
        "(or under DATA_FINAL_ROOT). [USER or ADMIN]"
    ),
    response_model=List[Counter],
    responses=generic_responses,
)
def get_all_counters(user_info: dict = Depends(_check_user_or_admin_role)):
    """
    Liste des compteurs - Accès pour utilisateurs et administrateurs
    """
    if not df_predictions:
        raise CustomException(
            type="PredictionsNotLoaded",
            message=f"df_predictions content: {df_predictions}",
            date=str(datetime.now()),
        )

    logger.info(
        f"Counters list requested by user: {user_info['username']} (role: {user_info['role']})"
    )
    return [Counter(id=name) for name in sorted(df_predictions.keys())]


@app.get(
    "/predictions/{counter_id}",
    tags=["Predictions"],
    summary="Get predictions for a counter",
    description=(
        "Return a paginated list of predictions for the given counter id "
        "(subdir name). Max 100 per page. [USER or ADMIN]"
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
    """
    Prédictions pour un compteur - Accès pour utilisateurs et administrateurs
    """
    if not df_predictions:
        raise CustomException(
            type="PredictionsNotLoaded",
            message=f"df_predictions content: {df_predictions}",
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

    logger.info(
        f"Predictions for counter {counter_id} requested by user: {user_info['username']} "
        f"(role: {user_info['role']}, limit: {limit}, offset: {offset})"
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


# Endpoint optionnel pour voir les informations de l'utilisateur connecté
@app.get(
    "/me",
    tags=["Admin"],
    summary="Get current user info",
    description="Get information about the currently authenticated user.",
    responses=generic_responses,
)
def get_current_user(user_info: dict = Depends(_check_credentials)):
    """
    Informations sur l'utilisateur connecté
    """
    return {
        "username": user_info["username"],
        "role": user_info["role"],
        "permissions": {
            "admin_endpoints": user_info["role"] == "admin",
            "prediction_endpoints": user_info["role"] in ["user", "admin"]
        }
    }
