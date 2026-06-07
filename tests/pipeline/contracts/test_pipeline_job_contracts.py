"""Unit tests for typed pipeline job request contracts."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.pipeline.contracts import (
    ArtifactManifestReference,
    FeatureJobRequest,
    IngestJobRequest,
    ModelJobRequest,
    PipelineJobRequest,
    PipelineJobType,
)


class TestPipelineJobRequests:
    """Unit tests for typed pipeline job requests."""

    @pytest.fixture
    def ingest_payload(self) -> dict:
        return {
            "job_type": "ingest",
            "run_id": "manual-run-001",
            "counter_id": "Sebastopol_N-S_dvcrepro",
            "requested_at": "2026-06-07T17:00:00Z",
            "dag_id": "bike_init",
            "task_id": "ingest",
            "try_number": 1,
            "manifest_root": "docker/prod/runtime/artifacts/manifests",
            "raw_path": "data/raw/bike-counts.csv",
            "site": "Totem 73 boulevard de Sébastopol",
            "orientation": "N-S",
            "range_start": 0.0,
            "range_end": 75.0,
            "sub_dir": "Sebastopol_N-S_dvcrepro",
            "interim_output_path": (
                "data/interim/Sebastopol_N-S_dvcrepro/initial.csv"
            ),
        }

    @pytest.fixture
    def features_payload(self) -> dict:
        return {
            "job_type": "features",
            "run_id": "manual-run-001",
            "counter_id": "Sebastopol_N-S_dvcrepro",
            "requested_at": "2026-06-07T17:00:00Z",
            "manifest_root": "docker/prod/runtime/artifacts/manifests",
            "interim_input_path": (
                "data/interim/Sebastopol_N-S_dvcrepro/initial.csv"
            ),
            "processed_output_path": (
                "data/processed/Sebastopol_N-S_dvcrepro/"
                "initial_with_feats.csv"
            ),
            "extra_drop": ["unused_column"],
        }

    @pytest.fixture
    def model_payload(self) -> dict:
        return {
            "job_type": "models",
            "run_id": "manual-run-001",
            "counter_id": "Sebastopol_N-S_dvcrepro",
            "requested_at": "2026-06-07T17:00:00Z",
            "manifest_root": "docker/prod/runtime/artifacts/manifests",
            "processed_input_path": (
                "data/processed/Sebastopol_N-S_dvcrepro/"
                "initial_with_feats.csv"
            ),
            "prediction_output_path": (
                "data/final/Sebastopol_N-S_dvcrepro/y_full.csv"
            ),
            "model_output_path": "models/Sebastopol_N-S_dvcrepro",
            "target_col": "comptage_horaire",
            "ts_col_utc": "date_et_heure_de_comptage_utc",
            "ts_col_local": "date_et_heure_de_comptage_local",
            "ar": 7,
            "mm": 1,
            "roll": 24,
            "test_ratio": 0.25,
            "grid_iter": 0,
            "expected_manifest": {
                "artifact_type": "predictions",
                "counter_id": "Sebastopol_N-S_dvcrepro",
                "run_id": "manual-run-001",
                "manifest_path": (
                    "docker/prod/runtime/artifacts/manifests/predictions/"
                    "Sebastopol_N-S_dvcrepro/manual-run-001/manifest.json"
                ),
                "current_path": (
                    "docker/prod/runtime/artifacts/manifests/predictions/"
                    "Sebastopol_N-S_dvcrepro/current.json"
                ),
            },
        }

    def test_valid_ingest_request_can_be_instantiated(self, ingest_payload):
        request = IngestJobRequest.model_validate(ingest_payload)

        assert request.job_type == PipelineJobType.INGEST
        assert request.range_start == 0.0
        assert request.range_end == 75.0
        assert request.interim_name == "initial.csv"
        assert request.requested_at == datetime(2026, 6, 7, 17, tzinfo=UTC)

    def test_valid_features_request_can_be_instantiated(self, features_payload):
        request = FeatureJobRequest.model_validate(features_payload)

        assert request.job_type == PipelineJobType.FEATURES
        assert request.processed_name == "initial_with_feats.csv"
        assert request.extra_drop == ("unused_column",)

    def test_valid_model_request_can_reference_expected_manifest(
        self,
        model_payload,
    ):
        request = ModelJobRequest.model_validate(model_payload)

        assert request.job_type == PipelineJobType.MODELS
        assert isinstance(request.expected_manifest, ArtifactManifestReference)
        assert request.expected_manifest.artifact_type == "predictions"

    def test_valid_pipeline_request_checks_step_artifact_handoff(
        self,
        ingest_payload,
        features_payload,
        model_payload,
    ):
        request = PipelineJobRequest.model_validate(
            {
                "job_type": "pipeline",
                "run_id": "manual-run-001",
                "counter_id": "Sebastopol_N-S_dvcrepro",
                "requested_at": "2026-06-07T17:00:00Z",
                "manifest_root": "docker/prod/runtime/artifacts/manifests",
                "ingest": ingest_payload,
                "features": features_payload,
                "models": model_payload,
            }
        )

        assert request.job_type == PipelineJobType.PIPELINE
        assert request.ingest.interim_output_path == (
            request.features.interim_input_path
        )
        assert request.features.processed_output_path == (
            request.models.processed_input_path
        )

    def test_invalid_percent_range_raises_validation_error(self, ingest_payload):
        payload = deepcopy(ingest_payload)
        payload["range_start"] = 90.0
        payload["range_end"] = 10.0

        with pytest.raises(ValidationError, match="range_start"):
            IngestJobRequest.model_validate(payload)

    def test_invalid_path_traversal_raises_validation_error(self, model_payload):
        payload = deepcopy(model_payload)
        payload["processed_input_path"] = "../data/processed/input.csv"

        with pytest.raises(ValidationError, match="parent traversal"):
            ModelJobRequest.model_validate(payload)

    def test_pipeline_rejects_inconsistent_step_handoff(
        self,
        ingest_payload,
        features_payload,
        model_payload,
    ):
        payload = deepcopy(features_payload)
        payload["interim_input_path"] = "data/interim/other/initial.csv"

        with pytest.raises(ValidationError, match="interim_output_path"):
            PipelineJobRequest.model_validate(
                {
                    "job_type": "pipeline",
                    "run_id": "manual-run-001",
                    "counter_id": "Sebastopol_N-S_dvcrepro",
                    "requested_at": "2026-06-07T17:00:00Z",
                    "manifest_root": "docker/prod/runtime/artifacts/manifests",
                    "ingest": ingest_payload,
                    "features": payload,
                    "models": model_payload,
                }
            )

    def test_naive_requested_at_raises_validation_error(self, ingest_payload):
        payload = deepcopy(ingest_payload)
        payload["requested_at"] = "2026-06-07T17:00:00"

        with pytest.raises(ValidationError, match="requested_at"):
            IngestJobRequest.model_validate(payload)
