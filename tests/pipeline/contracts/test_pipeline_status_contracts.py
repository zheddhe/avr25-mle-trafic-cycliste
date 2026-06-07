"""Unit tests for typed pipeline job status contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.pipeline.contracts import (
    TERMINAL_JOB_STATES,
    JobError,
    JobResult,
    JobState,
    JobStatus,
    MetricsEvidence,
    PipelineJobType,
)


class TestJobStatus:
    """Unit tests for JobStatus."""

    @pytest.fixture
    def result_payload(self) -> dict:
        return {
            "job_id": "job-001",
            "run_id": "manual-run-001",
            "counter_id": "Sebastopol_N-S_dvcrepro",
            "job_type": "models",
            "started_at": "2026-06-07T17:01:00Z",
            "finished_at": "2026-06-07T17:05:00Z",
            "output_paths": [
                "data/final/Sebastopol_N-S_dvcrepro/y_full.csv",
                "models/Sebastopol_N-S_dvcrepro/pipe_model.pkl",
            ],
            "manifest": {
                "artifact_type": "predictions",
                "counter_id": "Sebastopol_N-S_dvcrepro",
                "run_id": "manual-run-001",
                "manifest_path": (
                    "docker/prod/runtime/artifacts/manifests/predictions/"
                    "Sebastopol_N-S_dvcrepro/manual-run-001/manifest.json"
                ),
            },
            "metrics": {
                "records": 120,
                "metrics_pushed": True,
                "metrics_reference": "pushgateway://bike-traffic/job-001",
            },
        }

    def test_pending_status_can_be_instantiated(self):
        status = JobStatus.model_validate(
            {
                "job_id": "job-001",
                "run_id": "manual-run-001",
                "counter_id": "Sebastopol_N-S_dvcrepro",
                "job_type": "ingest",
                "state": "pending",
                "requested_at": "2026-06-07T17:00:00Z",
                "updated_at": "2026-06-07T17:00:00Z",
            }
        )

        assert status.state == JobState.PENDING
        assert status.job_type == PipelineJobType.INGEST
        assert status.result is None
        assert status.error is None

    def test_running_status_can_be_instantiated(self):
        status = JobStatus.model_validate(
            {
                "job_id": "job-001",
                "run_id": "manual-run-001",
                "counter_id": "Sebastopol_N-S_dvcrepro",
                "job_type": "features",
                "state": "running",
                "requested_at": "2026-06-07T17:00:00Z",
                "updated_at": "2026-06-07T17:01:00Z",
            }
        )

        assert status.state == JobState.RUNNING
        assert status.job_type == PipelineJobType.FEATURES

    def test_succeeded_status_requires_and_accepts_result(self, result_payload):
        status = JobStatus.model_validate(
            {
                "job_id": "job-001",
                "run_id": "manual-run-001",
                "counter_id": "Sebastopol_N-S_dvcrepro",
                "job_type": "models",
                "state": "succeeded",
                "requested_at": "2026-06-07T17:00:00Z",
                "updated_at": "2026-06-07T17:05:00Z",
                "result": result_payload,
            }
        )

        assert status.state == JobState.SUCCEEDED
        assert isinstance(status.result, JobResult)
        assert isinstance(status.result.metrics, MetricsEvidence)
        assert status.result.metrics.records == 120

    def test_failed_status_requires_and_accepts_error(self):
        status = JobStatus.model_validate(
            {
                "job_id": "job-001",
                "run_id": "manual-run-001",
                "counter_id": "Sebastopol_N-S_dvcrepro",
                "job_type": "pipeline",
                "state": "failed",
                "requested_at": "2026-06-07T17:00:00Z",
                "updated_at": "2026-06-07T17:03:00Z",
                "error": {
                    "code": "MODEL_STEP_FAILED",
                    "message": "Model execution failed.",
                    "retryable": True,
                },
            }
        )

        assert status.state == JobState.FAILED
        assert isinstance(status.error, JobError)
        assert status.error.retryable is True

    def test_failed_status_without_error_raises_validation_error(self):
        with pytest.raises(ValidationError, match="failed jobs must include error"):
            JobStatus.model_validate(
                {
                    "job_id": "job-001",
                    "run_id": "manual-run-001",
                    "counter_id": "Sebastopol_N-S_dvcrepro",
                    "job_type": "models",
                    "state": "failed",
                    "requested_at": "2026-06-07T17:00:00Z",
                    "updated_at": "2026-06-07T17:03:00Z",
                }
            )

    def test_succeeded_status_without_result_raises_validation_error(self):
        with pytest.raises(ValidationError, match="succeeded jobs"):
            JobStatus.model_validate(
                {
                    "job_id": "job-001",
                    "run_id": "manual-run-001",
                    "counter_id": "Sebastopol_N-S_dvcrepro",
                    "job_type": "models",
                    "state": "succeeded",
                    "requested_at": "2026-06-07T17:00:00Z",
                    "updated_at": "2026-06-07T17:03:00Z",
                }
            )

    def test_non_terminal_status_with_result_raises_validation_error(
        self,
        result_payload,
    ):
        with pytest.raises(ValidationError, match="non-terminal"):
            JobStatus.model_validate(
                {
                    "job_id": "job-001",
                    "run_id": "manual-run-001",
                    "counter_id": "Sebastopol_N-S_dvcrepro",
                    "job_type": "models",
                    "state": "running",
                    "requested_at": "2026-06-07T17:00:00Z",
                    "updated_at": "2026-06-07T17:03:00Z",
                    "result": result_payload,
                }
            )

    def test_result_rejects_reversed_timestamps(self, result_payload):
        payload = deepcopy(result_payload)
        payload["started_at"] = "2026-06-07T17:05:00Z"
        payload["finished_at"] = "2026-06-07T17:01:00Z"

        with pytest.raises(ValidationError, match="finished_at"):
            JobResult.model_validate(payload)

    def test_terminal_job_states_include_failure_and_success(self):
        assert JobState.SUCCEEDED in TERMINAL_JOB_STATES
        assert JobState.FAILED in TERMINAL_JOB_STATES
        assert JobState.PENDING not in TERMINAL_JOB_STATES
