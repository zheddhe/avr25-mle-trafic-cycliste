# src/api/main.py
from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

# Prometheus / FastAPI instrumentation
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram

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
    Lecture sécurisée d'un CSV + validation du schéma minimal.
    """
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        logger.warning("Failed to read CSV [%s]: %s", path, exc)
        return None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("CSV [%s] ignored, missing required columns: %s", path, missing)
        return None

    return df


def load_predictions_from_final(
    root_dir: str,
) -> tuple[Dict[str, pd.DataFrame], int, int]:
    """
    Explore <root_dir> pour trouver tous les sous-dossiers contenant un y_full.csv.
    Retourne (mapping, skipped_no_csv, invalid_csv).
    """
    mapping: Dict[str, pd.DataFrame] = {}
    skipped_no_csv = 0
    invalid_csv = 0

    if not os.path.isdir(root_dir):
        logger.warning("Final data root not found: [%s]", root_dir)
        return mapping, skipped_no_csv, invalid_csv

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        subdir = entry.name
        csv_path = os.path.join(entry.path, "y_full.csv")
        if not os.path.isfile(csv_path):
            logger.debug("No y_full.csv in [%s], skipping.", entry.path)
            skipped_no_csv += 1
            continue

        df = _safe_read_csv(csv_path)
        if df is None:
            invalid_csv += 1
            continue

        mapping[subdir] = df
        logger.info(
            "Loaded predictions for counter [%s]: %s rows x [%s] cols",
            subdir,
            df.shape[0],
            df.shape[1],
        )
    return mapping, skipped_no_csv, invalid_csv


def refresh_store() -> Dict[str, pd.DataFrame]:
    """
    Rescan filesystem et rafraîchit le store en mémoire.
    """
    global df_predictions
    mapping, _, _ = load_predictions_from_final(DATA_FINAL_ROOT)
    df_predictions = mapping
    logger.info("Store refreshed: %s counters available.", len(df_predictions))
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
    Validation Basic Auth pour endpoints protégés.
    """
    if credentials.username not in dict_credentials:
        raise HTTPException(status_code=403, detail=f"Unknown user [{credentials.username}]")
    if dict_credentials[credentials.username] != credentials.password:
        raise HTTPException(status_code=403, detail="Invalid password.")


# -------------------------------------------------------------------
# OpenAPI / tags
# -------------------------------------------------------------------
tags_metadata = [
    {"name": "Admin", "description": "Service health et maintenance."},
    {"name": "Predictions", "description": "Accès aux prédictions de trafic cycliste."},
]

app = FastAPI(
    title="API du trafic cycliste",
    description=(
        "Expose les prédictions du trafic cycliste pour les compteurs "
        "installés dans la ville de Paris."
    ),
    version="1.2.0",
    openapi_tags=tags_metadata,
)

# -------------------------------------------------------------------
# Prometheus instrumentation
# -------------------------------------------------------------------
# Latence, statut HTTP, throughput, etc.
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers={"/metrics"},
)
instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# Métrique métier: nombre de prédictions renvoyées par appel API
PRED_PER_RESPONSE = Histogram(
    "api_predictions_per_response",
    "Nombre de prédictions renvoyées par appel",
    buckets=(1, 5, 10, 20, 50, 100, float("inf")),
)


@app.on_event("startup")
def on_startup() -> None:
    """
    Chargement en mémoire au démarrage.
    """
    refresh_store()


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class ErrorResponse(BaseModel):
    type: str = Field(..., description="Business error type.")
    message: Optional[str] = Field(..., description="Detailed error message.")
    date: str = Field(..., description="Server-side timestamp.")


class PredictionItem(BaseModel):
    date_et_heure_de_comptage_local: datetime = Field(..., description="Local timestamp (Europe/Paris).")
    date_et_heure_de_comptage_utc: datetime = Field(..., description="UTC timestamp.")
    y_true: int = Field(..., description="Observed value.")
    y_pred: float = Field(..., description="Predicted value.")
    forecast_mode: bool = Field(..., description="True si prédiction future.")


class PredictionList(BaseModel):
    total: int = Field(..., description="Total disponible.")
    limit: int = Field(..., description="Max renvoyé.")
    offset: int = Field(..., description="Décalage pagination.")
    item: List[PredictionItem] = Field(..., description="Liste paginée.")


class Counter(BaseModel):
    id: str = Field(..., description="Identifiant compteur (nom du sous-dossier).")


class AdminRefreshResponse(BaseModel):
    message: str
    counters_before: int
    counters_after: int
    data_root: str
    loaded: int
    skipped_no_csv: int
    invalid_csv: int


# -------------------------------------------------------------------
# Custom exception + handler
# -------------------------------------------------------------------
class CustomException(Exception):
    def __init__(self, type: str, date: str, message: str):
        self.type = type
        self.date = date
        self.message = message


@app.exception_handler(CustomException)
def custom_exception_handler(_request: Request, exception: CustomException):
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
    description="Rescan DATA_FINAL_ROOT et recharge tous les compteurs.",
    response_model=AdminRefreshResponse,
    responses=generic_responses,
)
def post_refresh(user: str = Depends(_check_credentials)):
    global df_predictions
    before = len(df_predictions)
    mapping, skipped_no_csv, invalid_csv = load_predictions_from_final(DATA_FINAL_ROOT)
    df_predictions = mapping
    after = len(df_predictions)

    logger.info(
        "Admin refresh done. before=%s after=%s loaded=%s skipped_no_csv=%s invalid_csv=%s",
        before,
        after,
        after,
        skipped_no_csv,
        invalid_csv,
    )

    return AdminRefreshResponse(
        message="Store refreshed.",
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
    description="Liste tous les compteurs détectés sous data/final/<subdir>/y_full.csv.",
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
    description="Retourne une liste paginée de prédictions pour le compteur donné. Max 100.",
    response_model=PredictionList,
    responses=generic_responses,
)
def get_predictions_by_counter(
    counter_id: str,
    limit: int = Query(10, ge=1, le=100, description="Max number of predictions."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
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
            message=f"Available counters: {sorted(list(df_predictions.keys()))}",
            date=str(datetime.now()),
        )

    df = df_predictions[counter_id]
    df_page = df.iloc[offset : offset + limit]

    # --- métrique métier: nombre d'items renvoyés par appel
    try:
        PRED_PER_RESPONSE.observe(float(len(df_page)))
    except Exception as e:
        logger.debug("Failed to observe PRED_PER_RESPONSE: %s", e)

    return PredictionList(
        total=int(len(df)),
        limit=int(limit),
        offset=int(offset),
        item=[PredictionItem(**row) for row in df_page.to_dict(orient="records")],  # type: ignore
    )
