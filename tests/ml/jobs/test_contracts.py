"""Unit tests for typed ML job request contracts."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.ml.jobs.contracts import IngestJobRequest, MlJobType, ModelJobRequest


class TestMlJobRequests:
    """Unit tests for typed ML step job requests."""

    @pytest.fixture
    def ingest_payload(self) -> dict:
        return {
            "job_type": "ingest",
            "run_id": "manual-run-001",
            "counter_id": "Sebastopol_N-S_dvcrepro",
            "requested_at": "2026-06-07T17:00:00Z",
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
        assert request.requested_at == datetime(2026, 6, 7, 17, tzinfo=UTC)

    def test_invalid_percent_range_raises_validation_error(
        self,
        ingest_payload: dict,
    ) -> None:
        payload = deepcopy(ingest_payload)
        payload["range_start"] = 90.0
        payload["range_end"] = 10.0

        with pytest.raises(ValidationError, match="range_start"):
            IngestJobRequest.model_validate(payload)

    def test_invalid_path_traversal_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="parent traversal"):
            ModelJobRequest.model_validate(
                {
                    "job_type": "models",
                    "run_id": "run-001",
                    "counter_id": "counter-001",
                    "processed_input_path": "../data/processed/input.csv",
                },
            )
