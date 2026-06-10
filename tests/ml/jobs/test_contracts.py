"""Unit tests for typed ML job request contracts."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.ml.jobs.contracts import (
    ArtifactManifestReference,
    FeatureJobRequest,
    IngestJobRequest,
    MlJobType,
    ModelJobRequest,
    validate_filesystem_path,
)


class TestMlJobRequests:
    """Unit tests for typed ML step job requests."""

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

    def test_active_job_types_do_not_expose_pipeline(self) -> None:
        assert {job_type.value for job_type in MlJobType} == {
            "ingest",
            "features",
            "models",
        }

    def test_valid_ingest_request_can_be_instantiated(
        self,
        ingest_payload: dict,
    ) -> None:
        request = IngestJobRequest.model_validate(ingest_payload)

        assert request.job_type == MlJobType.INGEST
        assert request.range_start == 0.0
        assert request.range_end == 75.0
        assert request.interim_name == "initial.csv"
        assert request.requested_at == datetime(2026, 6, 7, 17, tzinfo=UTC)

    def test_valid_features_request_can_be_instantiated(
        self,
        features_payload: dict,
    ) -> None:
        request = FeatureJobRequest.model_validate(features_payload)

        assert request.job_type == MlJobType.FEATURES
        assert request.processed_name == "initial_with_feats.csv"
        assert request.extra_drop == ("unused_column",)

    def test_valid_model_request_can_reference_expected_manifest(
        self,
        model_payload: dict,
    ) -> None:
        request = ModelJobRequest.model_validate(model_payload)

        assert request.job_type == MlJobType.MODELS
        assert isinstance(request.expected_manifest, ArtifactManifestReference)
        assert request.expected_manifest.artifact_type == "predictions"

    def test_invalid_percent_range_raises_validation_error(
        self,
        ingest_payload: dict,
    ) -> None:
        payload = deepcopy(ingest_payload)
        payload["range_start"] = 90.0
        payload["range_end"] = 10.0

        with pytest.raises(ValidationError, match="range_start"):
            IngestJobRequest.model_validate(payload)

    def test_invalid_path_traversal_raises_validation_error(
        self,
        model_payload: dict,
    ) -> None:
        payload = deepcopy(model_payload)
        payload["processed_input_path"] = "../data/processed/input.csv"

        with pytest.raises(ValidationError, match="parent traversal"):
            ModelJobRequest.model_validate(payload)

    def test_naive_requested_at_raises_validation_error(
        self,
        ingest_payload: dict,
    ) -> None:
        payload = deepcopy(ingest_payload)
        payload["requested_at"] = "2026-06-07T17:00:00"

        with pytest.raises(ValidationError, match="requested_at"):
            IngestJobRequest.model_validate(payload)

    def test_manifest_reference_rejects_invalid_object_uri(self) -> None:
        with pytest.raises(ValidationError, match="object_uri"):
            ArtifactManifestReference(
                artifact_type="predictions",
                counter_id="counter-001",
                run_id="run-001",
                manifest_path="artifacts/manifests/predictions/run/manifest.json",
                object_uri="https://bucket/predictions.csv",
            )

    def test_manifest_reference_rejects_embedded_credentials(self) -> None:
        with pytest.raises(ValidationError, match="credentials"):
            ArtifactManifestReference(
                artifact_type="predictions",
                counter_id="counter-001",
                run_id="run-001",
                manifest_path="artifacts/manifests/predictions/run/manifest.json",
                object_uri="s3://user:secret@bucket/predictions.csv",
            )

    def test_model_request_rejects_invalid_artifact_object_uri(
        self,
        model_payload: dict,
    ) -> None:
        payload = deepcopy(model_payload)
        payload["artifact_object_uri"] = "https://bucket/predictions.csv"

        with pytest.raises(ValidationError, match="artifact_object_uri"):
            ModelJobRequest.model_validate(payload)

    def test_model_request_rejects_artifact_uri_credentials(
        self,
        model_payload: dict,
    ) -> None:
        payload = deepcopy(model_payload)
        payload["artifact_object_uri"] = "s3://user:secret@bucket/predictions.csv"

        with pytest.raises(ValidationError, match="credentials"):
            ModelJobRequest.model_validate(payload)

    def test_validate_filesystem_path_allows_none(self) -> None:
        assert validate_filesystem_path(None) is None

    def test_validate_filesystem_path_rejects_empty_value(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            validate_filesystem_path("")

    def test_validate_filesystem_path_rejects_uri_scheme(self) -> None:
        with pytest.raises(ValueError, match="local filesystem"):
            validate_filesystem_path("s3://bucket/path.csv")
