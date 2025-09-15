# src/api/main.py
from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

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
        logger.warning(f"Failed to read CSV [{path}]: {exc}")
        return None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(
            f"CSV [{path}] ignored, missing required columns: {missing}"
        )
        return None

    return df


def load_predictions_from_final(root_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Explore <root_dir> to find all subdirs containing a y_full.csv.
    Return a mapping {subdir_name: dataframe}.
    """
    mapping: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(root_dir):
        logger.warning(f"Final data root not found: [{root_dir}]")
        return mapping

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        subdir = entry.name
        csv_path = os.path.join(entry.path, "y_full.csv")
        if not os.path.isfile(csv_path):
            logger.debug(f"No y_full.csv in [{entry.path}], skipping.")
            continue

        df = _safe_read_csv(csv_path)
        if df is None:
            continue

        mapping[subdir] = df
        logger.info(
            f"Loaded predictions for counter [{subdir}]: {df.shape[0]} rows x [{df.shape[1]}] cols"
        )
    return mapping


def refresh_store() -> Dict[str, pd.DataFrame]:
    """
    Rescan the filesystem and refresh the in-memory store.
    """
    global df_predictions
    mapping = load_predictions_from_final(DATA_FINAL_ROOT)
    df_predictions = mapping
    logger.info(f"Store refreshed: {len(df_predictions)} counters available.")
    return df_predictions


# -------------------------------------------------------------------
# Simple security (Basic Auth)
# -------------------------------------------------------------------
dict_credentials = {
    "remy": "remy",
    "elias": "elias",
    "kolade": "kolade",
}
security = HTTPBasic()


def _check_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Validate Basic Auth credentials for docs (/docs, /redoc) and endpoints.
    """
    if credentials.username not in dict_credentials:
        raise HTTPException(
            status_code=403,
            detail=f"Unknown user [{credentials.username}]",
        )
    if dict_credentials[credentials.username] != credentials.password:
        raise HTTPException(
            status_code=403,
            detail="Invalid password.",
        )
    # Do not return secrets; just allow the request to proceed.


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


@app.on_event("startup")
def on_startup() -> None:
    """
    Load all counters at service startup.
    """
    refresh_store()


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
# Routes
# -------------------------------------------------------------------
@app.get(
    "/verify",
    tags=["Admin"],
    summary="Verify service health",
    description="Simple service health check.",
    responses={},
)
def get_verify():
    return {"message": "API is healthy."}


@app.post(
    "/admin/refresh",
    tags=["Admin"],
    summary="Refresh in-memory store",
    description=(
        "Rescan the filesystem (DATA_FINAL_ROOT) and reload all counters."
    ),
    responses=generic_responses,
)
def post_refresh(user: str = Depends(_check_credentials)):
    before = len(df_predictions)
    refresh_store()
    after = len(df_predictions)
    return {
        "message": "Store refreshed.",
        "counters_before": before,
        "counters_after": after,
        "data_root": os.path.abspath(DATA_FINAL_ROOT),
    }


@app.get(
    "/counters",
    tags=["Predictions"],
    summary="List available counters",
    description=(
        "List all counters detected under data/final/<subdir>/y_full.csv "
        "(or under DATA_FINAL_ROOT)."
    ),
    response_model=List[Counter],
    responses=generic_responses,
)
def get_all_counters(user: str = Depends(_check_credentials)):
    if not df_predictions:
        raise CustomException(
            type="PredictionsNotLoaded",
            message=f"df_predictions content: {df_predictions}",
            date=str(datetime.now()),
        )
    return [Counter(id=name) for name in sorted(df_predictions.keys())]


@app.get(
    "/predictions/{counter_id}",
    tags=["Predictions"],
    summary="Get predictions for a counter",
    description=(
        "Return a paginated list of predictions for the given counter id "
        "(subdir name). Max 100 per page."
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
    user: str = Depends(_check_credentials),
):
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
    df_page = df.iloc[offset:offset + limit]

    return PredictionList(
        total=int(len(df)),
        limit=int(limit),
        offset=int(offset),
        item=[
            PredictionItem(**row)  # type: ignore
            for row in df_page.to_dict(orient="records")
        ],
    )
