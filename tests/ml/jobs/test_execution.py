"""Unit tests for typed ML step execution helpers."""

from __future__ import annotations

from src.ml.jobs.contracts import IngestJobRequest, MlJobType
from src.ml.jobs.execution import _artifact_type_for_job, _metrics_label_values, _path_parent_name


class TestMlJobExecutionHelpers:
    """Unit tests for execution helper functions."""

    def test_artifact_type_for_job_maps_supported_steps(self) -> None:
        assert _artifact_type_for_job(MlJobType.INGEST) == "interim_dataset"
        assert _artifact_type_for_job(MlJobType.FEATURES) == "feature_dataset"
        assert _artifact_type_for_job(MlJobType.MODELS) == "predictions"

    def test_path_parent_name_extracts_output_scenario(self) -> None:
        assert _path_parent_name("/app/data/final/counter-001/y_full.csv") == (
            "counter-001"
        )

    def test_metrics_label_values_uses_ingest_orientation(self) -> None:
        job_request = IngestJobRequest(
            run_id="run-001",
            counter_id="Sebastopol_S-N_airflow",
            raw_path="/app/data/raw/source.csv",
            site="Totem 73 boulevard de Sébastopol",
            orientation="N-S",
            sub_dir="counter-001",
            interim_output_path="/app/data/interim/counter-001/initial.csv",
        )

        assert _metrics_label_values(job_request) == ("Sebastopol", "N-S")
