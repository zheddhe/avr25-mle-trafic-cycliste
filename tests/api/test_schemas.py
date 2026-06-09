# tests/api/test_schemas.py
from __future__ import annotations

from datetime import UTC, datetime

from src.api.schemas import (
    ArtifactSourceMetadata,
    Counter,
    CurrentArtifactMetadata,
    ErrorResponse,
    PredictionItem,
    PredictionList,
)


class TestApiSchemas:
    def test_error_response_serializes_business_error(self) -> None:
        response = ErrorResponse(
            type="PredictionsNotLoaded",
            message="No promoted prediction manifest has been loaded.",
            date="2026-06-09T10:00:00",
        )

        assert response.model_dump() == {
            "type": "PredictionsNotLoaded",
            "message": "No promoted prediction manifest has been loaded.",
            "date": "2026-06-09T10:00:00",
        }

    def test_counter_schema_keeps_existing_interface_contract(self) -> None:
        counter = Counter(id="Sebastopol_N-S")

        assert counter.model_dump() == {"id": "Sebastopol_N-S"}

    def test_prediction_list_accepts_paginated_items(self) -> None:
        item = PredictionItem(
            date_et_heure_de_comptage_local="2025-09-23T08:00:00+02:00",
            date_et_heure_de_comptage_utc="2025-09-23T06:00:00+00:00",
            y_true=123,
            y_pred=120.5,
            forecast_mode=False,
        )

        response = PredictionList(total=1, limit=10, offset=0, item=[item])

        assert response.total == 1
        assert response.item[0].y_pred == 120.5

    def test_current_artifact_metadata_serializes_sanitized_manifest(self) -> None:
        created_at = datetime(2026, 6, 6, 14, 0, tzinfo=UTC)
        metadata = CurrentArtifactMetadata(
            counter_id="Sebastopol_N-S",
            run_id="run-1",
            artifact_type="predictions",
            status="promoted",
            created_at=created_at,
            producer_service="ml-models-prod",
            producer_image="bike-traffic/ml-models-prod:test",
            producer_version="test",
            source=ArtifactSourceMetadata(
                raw_file_name="comptage-velo.csv",
                dataset_version="dataset-v1",
                model_version="model-v1",
            ),
            primary_backend="local",
            local_path="data/final/Sebastopol_N-S/y_full.csv",
            object_uri=None,
            checksum_sha256="a" * 64,
        )

        assert metadata.counter_id == "Sebastopol_N-S"
        assert metadata.source.dataset_version == "dataset-v1"
        assert metadata.object_uri is None
