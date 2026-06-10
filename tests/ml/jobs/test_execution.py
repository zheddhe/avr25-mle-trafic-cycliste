"""Unit tests for typed ML step execution helpers."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from src.artifacts.schemas import ArtifactType
from src.ml.jobs.contracts import (
    ArtifactManifestReference,
    FeatureJobRequest,
    IngestJobRequest,
    MlJobType,
    ModelJobRequest,
)
from src.ml.jobs.execution import (
    MlStepExecutionError,
    StepCommandExecutor,
    _artifact_type_for_job,
    _execution_env,
    _execution_error,
    _metrics_label_values,
    _model_sub_dir,
    _patched_environ,
    _path_parent_name,
)


def _build_ingest_request() -> IngestJobRequest:
    return IngestJobRequest(
        run_id="run-001",
        counter_id="Sebastopol_S-N_airflow",
        raw_path="/app/data/raw/source.csv",
        site="Totem 73 boulevard de Sébastopol",
        orientation="N-S",
        sub_dir="counter-001",
        interim_output_path="/app/data/interim/counter-001/initial.csv",
    )


def _build_feature_request() -> FeatureJobRequest:
    return FeatureJobRequest(
        run_id="run-001",
        counter_id="Sebastopol_N-S_airflow",
        dag_id="bike_init",
        task_id="features",
        manifest_root="/app/artifacts/manifests",
        interim_input_path="/app/data/interim/counter-001/initial.csv",
        processed_output_path=(
            "/app/data/processed/counter-001/initial_with_feats.csv"
        ),
        extra_drop=("unused_column",),
    )


def _build_model_request(
    expected_manifest: ArtifactManifestReference | None = None,
) -> ModelJobRequest:
    return ModelJobRequest(
        run_id="run-001",
        counter_id="Sebastopol_N-S_airflow",
        manifest_root="/app/artifacts/manifests",
        processed_input_path=(
            "/app/data/processed/counter-001/initial_with_feats.csv"
        ),
        prediction_output_path="/app/data/final/counter-001/y_full.csv",
        model_output_path="/app/models/counter-001",
        mlflow_uri="http://mlflow:5000",
        artifact_object_uri="s3://bucket/counter-001/y_full.csv",
        expected_manifest=expected_manifest,
    )


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
        job_request = _build_ingest_request()

        assert _metrics_label_values(job_request) == ("Sebastopol", "N-S")

    def test_execution_env_contains_traceability_labels(self) -> None:
        job_request = _build_feature_request()

        env = _execution_env(job_request)

        assert env == {
            "RUN_ID": "run-001",
            "COUNTER_ID": "Sebastopol_N-S_airflow",
            "SITE_SHORT": "Sebastopol",
            "SITE": "Sebastopol",
            "ORIENTATION": "N-S",
            "ARTIFACT_PRODUCER_SERVICE": "ml-features",
            "AIRFLOW_CTX_DAG_ID": "bike_init",
            "AIRFLOW_CTX_TASK_ID": "features",
            "ARTIFACT_MANIFEST_ROOT": "/app/artifacts/manifests",
        }

    def test_patched_environ_restores_previous_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EXISTING_KEY", "before")

        with _patched_environ({"EXISTING_KEY": "after", "NEW_KEY": "value"}):
            assert os.environ["EXISTING_KEY"] == "after"
            assert os.environ["NEW_KEY"] == "value"

        assert os.environ["EXISTING_KEY"] == "before"
        assert "NEW_KEY" not in os.environ

    def test_execution_error_uses_default_message_when_empty(self) -> None:
        error = _execution_error(_build_ingest_request(), "")

        assert error.code == "INGEST_JOB_FAILED"
        assert error.message == "ML step execution failed without output."
        assert error.retryable is True

    def test_model_sub_dir_prefers_prediction_output_path(self) -> None:
        job_request = _build_model_request()

        assert _model_sub_dir(job_request) == "counter-001"


class TestStepCommandExecutor:
    """Unit tests for the framework-neutral step command executor."""

    def test_execute_feature_job_returns_manifest_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        executor = StepCommandExecutor()
        job_request = _build_feature_request()

        def fake_execute_features(request: FeatureJobRequest) -> tuple[str, ...]:
            return (request.processed_output_path,)

        monkeypatch.setattr(executor, "_execute_features", fake_execute_features)

        result = executor.execute(
            job_request,
            job_id="runner-job-001",
            started_at=datetime(2026, 6, 7, 17, tzinfo=UTC),
        )

        assert result.job_id == "runner-job-001"
        assert result.output_paths == (job_request.processed_output_path,)
        assert result.manifest is not None
        assert result.manifest.artifact_type == ArtifactType.FEATURE_DATASET
        assert result.manifest.manifest_path == (
            "/app/artifacts/manifests/feature_dataset/"
            "Sebastopol_N-S_airflow/run-001/manifest.json"
        )

    def test_execute_model_job_reuses_expected_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected_manifest = ArtifactManifestReference(
            artifact_type=ArtifactType.PREDICTIONS,
            counter_id="Sebastopol_N-S_airflow",
            run_id="run-001",
            manifest_path="artifacts/manifests/predictions/counter/run/manifest.json",
            current_path="artifacts/manifests/predictions/counter/current.json",
            object_uri="s3://bucket/counter-001/y_full.csv",
        )
        job_request = _build_model_request(expected_manifest=expected_manifest)
        executor = StepCommandExecutor()

        def fake_execute_models(request: ModelJobRequest) -> tuple[str, ...]:
            return (request.prediction_output_path, request.model_output_path)

        monkeypatch.setattr(executor, "_execute_models", fake_execute_models)

        result = executor.execute(
            job_request,
            job_id="runner-job-002",
            started_at=datetime(2026, 6, 7, 17, tzinfo=UTC),
        )

        assert result.manifest == expected_manifest
        assert result.output_paths == (
            "/app/data/final/counter-001/y_full.csv",
            "/app/models/counter-001",
        )

    def test_invoke_ignores_successful_system_exit(self) -> None:
        executor = StepCommandExecutor()

        class SuccessfulCommand:
            name = "success"

            def main(self, *args, **kwargs) -> None:
                raise SystemExit(0)

        executor._invoke(SuccessfulCommand(), [], _build_ingest_request())

    def test_invoke_maps_failed_system_exit_to_step_error(self) -> None:
        executor = StepCommandExecutor()

        class FailingCommand:
            name = "failing"

            def main(self, *args, **kwargs) -> None:
                raise SystemExit(2)

        with pytest.raises(MlStepExecutionError) as exc_info:
            executor._invoke(FailingCommand(), [], _build_ingest_request())

        assert exc_info.value.code == "INGEST_JOB_FAILED"
        assert exc_info.value.retryable is True
