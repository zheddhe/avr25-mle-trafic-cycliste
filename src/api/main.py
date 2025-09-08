from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Request
from fastapi.responses import JSONResponse
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Dict, Union, Optional
import pandas as pd
import logging
import os

SITE_TEST = {
    "Sebastopol_N-S": "y_full.csv",
}

# redirect logs on uvicorn directly for exposure in the console
logger = logging.getLogger("uvicorn.error")

# dataset load in a dictionnary of dataframe with key = counter name
df_predictions = {}
for counter_name in SITE_TEST.keys():
    # working paths
    input_file_name = SITE_TEST[counter_name]
    input_sub_dir = counter_name
    input_file_path = os.path.join("data", "final", input_sub_dir, input_file_name)
    df = pd.read_csv(input_file_path, index_col=0)
    logger.info(
        f"Dataframe chargé pour [{counter_name}]. "
        f"Il contient {df.shape[0]} lignes et {df.shape[1]} colonnes"
    )
    df_predictions[input_sub_dir] = df

# Simple security database... (a clear dictionnary in the API :p)
dict_credentials = {
  "remy": "remy",
  "elias": "elias",
  "kolade": "kolade"
}
security = HTTPBasic()

# ---------------------------------------------------------------------------
# OpenAPI / tags
# ---------------------------------------------------------------------------
tags_metadata = [
    {
        "name": "Admin",
        "description": "Section de vérification du service.",
    },
    {
        "name": "Predictions",
        "description": "Section de gestion des prédictions",
    },
]

app = FastAPI(
    title="API du trafic cycliste",
    description=(
        "Expose les prediction du trafic cycliste pour les compteurs installés dans la ville"
        " de paris"
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
)


# ---------------------------------------------------------------------------
# Modèles (avec description des champs intégrée)
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    type: str = Field(..., description="type d'erreur métier")
    message: Optional[str] = Field(..., description="contenu précisant l'erreur le cas échéant")
    date: str = Field(..., description="2025-08-26 14:08:19.431291")


class Prediction(BaseModel):
    date: datetime = Field(..., description="Horodatage (time zone Paris) du comptage")
    forecasted_count: int = Field(..., description="Valeur prédite du comptage")


class Counter(BaseModel):
    id: str = Field(..., description="Identifiant du compteur")


# ---------------------------------------------------------------------------
# Exception métier personalisée + son handler
# ---------------------------------------------------------------------------
class CustomException(Exception):
    def __init__(self,
                 type: str,
                 date: str,
                 message: str):
        self.type = type
        self.date = date
        self.message = message


@app.exception_handler(CustomException)
def CustomExceptionHandler(
    _request: Request,  # non utilisé
    exception: CustomException
):
    return JSONResponse(
        status_code=418,
        content=ErrorResponse(
            type=exception.type,
            message=exception.message,
            date=exception.date,
        ).model_dump()
    )


# ---------------------------------------------------------------------------
# Documentation (globale) des réponses
# ---------------------------------------------------------------------------
# on type correctement le dictionnaire des réponses attendu par le décorateur
ResponsesDict = Dict[Union[int, str], Dict[str, Any]]
generic_responses: ResponsesDict = {
    200: {"description": "Succès"},
    400: {"description": "Erreur de valeur dans le contenu de la requête"},
    403: {"description": "Echec lors de l'authentification"},
    404: {"description": "La route demandée est inconnue"},
    418: {
        "description": "Erreur métier détectée",
        "model": ErrorResponse,
    },
    422: {"description": "Erreur de validation des données"},
    500: {"description": "Erreur côté serveur"},
}


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------
def _check_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verification des identifiants en base 64 native pour fastAPI (+ docs/redoc)
    """
    if credentials.username not in dict_credentials.keys():
        raise HTTPException(
            status_code=403,
            detail=f"Utilisateur [{credentials.username}] inconnu"
        )
    if dict_credentials[credentials.username] != credentials.password:
        raise HTTPException(
            status_code=403,
            detail=f"Mot de passe incorrect pour l'utilisateur [{credentials.username}]"
        )


# ---------------------------------------------------------------------------
# Définition des Routes
# ---------------------------------------------------------------------------
@app.get(
    "/verify",
    tags=["Admin"],
    summary="Vérifier le service",
    description="Vérifie que l'API est fonctionnelle.",
    responses={},  # réponses par défaut (200 seulement ici)
)
def get_verify():
    return {
        "message": "L'API est fonctionnelle."
    }


@app.get(
    "/counters",
    tags=["Predictions"],
    summary="Lister les compteurs disponibles",
    description="Affiche l'ensemble des compteurs pour lesquels des prédictions ont été calculées",
    responses=generic_responses,
)
def get_all_predictions(
    user: str = Depends(_check_credentials)
):
    """Return all available counters"""
    if not df_predictions:
        raise CustomException(
            type="Dictionnaire des prédictions non chargé",
            message=f"contenu de df_predictions: {df_predictions}",
            date=str(datetime.now()),
        )
    return [Counter(id=name) for name in df_predictions.keys()]


@app.get(
    "/predictions/{counter_id}",
    tags=["Predictions"],
    summary="Afficher les predictions d'un compteur",
    description=(
        "Affiche l'ensemble des prédictions horodatées calculées pour ce compteur.\n"
        "Limite l'affichage aux 50 derniers enregistrements si aucune limite donnée"
    ),
    responses=generic_responses,
)
def get_predictions_by_counter(
    counter_id: str,
    limit: int = Query(10, ge=1, le=100, description="Nombre de prédictions à retourner"),
    offset: int = Query(0, ge=0, description="Nombre de prédictions à passer"),
    user: str = Depends(_check_credentials)
):
    """Return predictions for a specific counter"""
    if not df_predictions:
        raise CustomException(
            type="Dictionnaire des prédictions non chargé",
            message=f"contenu de df_predictions: {df_predictions}",
            date=str(datetime.now()),
        )
    if counter_id not in df_predictions:
        raise CustomException(
            type="Compteur non disponible",
            message=f"Liste des compteur disponibles: {list(df_predictions.keys())}",
            date=str(datetime.now()),
        )
    df_paginated = df_predictions[counter_id].iloc[offset: offset + limit]

    return {
        "total": len(df_predictions[counter_id]),
        "limit": limit,
        "offset": offset,
        "item": df_paginated.to_dict(orient="records"),
    }
