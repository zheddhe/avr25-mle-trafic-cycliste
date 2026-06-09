"""Manifest-first prediction loading for the serving API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.api.config import ApiSettings
from src.api.schemas import ArtifactSourceMetadata, CurrentArtifactMetadata
from src.artifacts.exceptions import ArtifactManifestNotFoundError
from src.artifacts.manifest_store import read_current_manifest, verify_local_payload
from src.artifacts.schemas import ArtifactManifest, ArtifactType, StorageBackend

LOGGER = logging.getLogger(__name__)

REQUIRED_PREDICTION_COLUMNS = {
    "date_et_heure_de_comptage_local",
    "date_et_heure_de_comptage_utc",
    "y_true",
    "y_pred",
    "forecast_mode",
}


class PredictionServingError(ValueError):
    """Base exception raised by prediction serving helpers."""


class PredictionCsvError(PredictionServingError):
    """Raised when a prediction CSV cannot be loaded or validated."""


class UnsupportedPredictionBackendError(PredictionServingError):
    """Raised when the API cannot serve the manifest storage backend."""


@dataclass(frozen=True)
class PredictionLoadResult:
    """Loaded predictions and sanitized artifact metadata."""

    predictions: dict[str, pd.DataFrame]
    artifacts: dict[str, CurrentArtifactMetadata]


def load_predictions_from_manifests(settings: ApiSettings) -> PredictionLoadResult:
    """Load predictions from promoted current manifests only."""

    counter_ids = settings.counter_ids or discover_current_prediction_counter_ids(
        settings.manifest_root
    )
    if not counter_ids:
        predictions_root = settings.manifest_root / ArtifactType.PREDICTIONS.value
        raise ArtifactManifestNotFoundError(
            "No promoted prediction current.json manifest found under "
            f"{predictions_root}"
        )

    predictions: dict[str, pd.DataFrame] = {}
    artifacts: dict[str, CurrentArtifactMetadata] = {}
    for counter_id in counter_ids:
        manifest = read_current_manifest(
            settings.manifest_root,
            ArtifactType.PREDICTIONS.value,
            counter_id,
        )
        dataframe = load_prediction_dataframe_from_manifest(
            manifest=manifest,
            settings=settings,
        )
        predictions[manifest.counter_id] = dataframe
        artifacts[manifest.counter_id] = build_artifact_metadata(manifest)
        LOGGER.info(
            f"Loaded promoted predictions for counter {manifest.counter_id}: "
            f"{dataframe.shape[0]} rows x {dataframe.shape[1]} cols"
        )

    return PredictionLoadResult(predictions=predictions, artifacts=artifacts)


def discover_current_prediction_counter_ids(manifest_root: Path) -> tuple[str, ...]:
    """Discover counters that expose a promoted prediction current manifest."""

    predictions_root = manifest_root / ArtifactType.PREDICTIONS.value
    if not predictions_root.is_dir():
        return ()

    counter_ids = [
        path.parent.name
        for path in predictions_root.glob("*/current.json")
        if path.is_file()
    ]
    return tuple(sorted(counter_ids))


def load_prediction_dataframe_from_manifest(
    *,
    manifest: ArtifactManifest,
    settings: ApiSettings,
) -> pd.DataFrame:
    """Load one prediction dataframe from a validated manifest."""

    if manifest.artifact_type != ArtifactType.PREDICTIONS:
        raise PredictionServingError(
            "Expected prediction artifact manifest, got "
            f"{manifest.artifact_type.value}."
        )

    if manifest.storage.primary_backend != StorageBackend.LOCAL:
        raise UnsupportedPredictionBackendError(
            "Prediction serving only supports local artifacts, got "
            f"{manifest.storage.primary_backend.value}."
        )

    if manifest.storage.local_path is None:
        raise PredictionServingError(
            "Prediction manifest does not expose a local payload path: "
            f"{manifest.counter_id}"
        )

    verify_local_payload(manifest, repository_root=settings.repository_root)
    payload_path = settings.repository_root / manifest.storage.local_path
    return read_prediction_csv(payload_path)


def read_prediction_csv(path: Path) -> pd.DataFrame:
    """Read and validate one prediction CSV payload."""

    try:
        dataframe = pd.read_csv(path, index_col=0)
    except Exception as error:
        raise PredictionCsvError(
            f"Unable to read prediction CSV payload: {path}"
        ) from error

    missing_columns = sorted(REQUIRED_PREDICTION_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise PredictionCsvError(
            "Prediction CSV is missing required columns: "
            f"{missing_columns}"
        )

    if dataframe.empty:
        raise PredictionCsvError(f"Prediction CSV is empty: {path}")

    return dataframe


def build_artifact_metadata(manifest: ArtifactManifest) -> CurrentArtifactMetadata:
    """Build sanitized metadata for one served prediction artifact."""

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
