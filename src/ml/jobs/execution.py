"""Shared typed ML step execution adapter.

The adapter remains framework-neutral so it can be reused by the local runner
fallback and by the internal FastAPI ML step services.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import PurePosixPath
from typing import Iterator

from click import Command

from src.artifacts.schemas import ArtifactType
from src.ml.jobs.contracts import (
    ArtifactManifestReference,
    FeatureJobRequest,
    IngestJobRequest,
    MlJobType,
    ModelJobRequest,
    StepJobRequest,
)
from src.ml.jobs.status import JobResult, MetricsEvidence, utc_now


class MlStepExecutionError(Exception):
    """Controlled error raised when a typed ML step fails."""

    def __init__(self, code: str, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class StepCommandExecutor:
    """Execute one allow-listed ML step by reusing existing Click commands."""

    def execute(
        self,
        job_request: StepJobRequest,
        *,
        job_id: str,
        started_at: datetime,
    ) -> JobResult:
        """Execute one typed ML job and build step-level evidence."""

        if isinstance(job_request, IngestJobRequest):
            output_paths = self._execute_ingest(job_request)
        elif isinstance(job_request, FeatureJobRequest):
            output_paths = self._execute_features(job_request)
        elif isinstance(job_request, ModelJobRequest):
            output_paths = self._execute_models(job_request)
        else:  # pragma: no cover - guarded by Pydantic request union.
            raise MlStepExecutionError(
                code="UNSUPPORTED_JOB_TYPE",
                message="Unsupported typed ML job request.",
                retryable=False,
            )

        return JobResult(
            job_id=job_id,
            run_id=job_request.run_id,
            counter_id=job_request.counter_id,
            job_type=job_request.job_type,
            started_at=started_at,
            finished_at=utc_now(),
            output_paths=output_paths,
            manifest=self._build_manifest_reference(job_request),
            metrics=MetricsEvidence(
                metrics_reference=f"logs/ml/{job_request.job_type.value}.log",
            ),
        )

    def _execute_ingest(self, job_request: IngestJobRequest) -> tuple[str, ...]:
        from src.ml.ingest.import_raw_data import main as ingest_command

        args = [
            "--raw-path",
            job_request.raw_path,
            "--site",
            job_request.site,
            "--orientation",
            job_request.orientation,
            "--range-start",
            str(job_request.range_start),
            "--range-end",
            str(job_request.range_end),
            "--timestamp-col",
            job_request.timestamp_col,
            "--sub-dir",
            job_request.sub_dir,
            "--interim-name",
            job_request.interim_name,
        ]
        self._append_manifest_root(args, job_request)
        self._invoke(ingest_command, args, job_request)
        return (job_request.interim_output_path,)

    def _execute_features(
        self,
        job_request: FeatureJobRequest,
    ) -> tuple[str, ...]:
        from src.ml.features.build_features import main as features_command

        args = [
            "--interim-path",
            job_request.interim_input_path,
            "--sub-dir",
            _path_parent_name(job_request.processed_output_path),
            "--processed-name",
            job_request.processed_name,
            "--timestamp-col",
            job_request.timestamp_col,
        ]
        for column in job_request.extra_drop:
            args.extend(["--extra-drop", column])

        self._append_manifest_root(args, job_request)
        self._invoke(features_command, args, job_request)
        return (job_request.processed_output_path,)

    def _execute_models(self, job_request: ModelJobRequest) -> tuple[str, ...]:
        from src.ml.models.train_and_predict import main as models_command

        sub_dir = _model_sub_dir(job_request)
        args = [
            "--processed-path",
            job_request.processed_input_path,
            "--sub-dir",
            sub_dir,
            "--target-col",
            job_request.target_col,
            "--ts-col-utc",
            job_request.ts_col_utc,
            "--ts-col-local",
            job_request.ts_col_local,
            "--ar",
            str(job_request.ar),
            "--mm",
            str(job_request.mm),
            "--roll",
            str(job_request.roll),
            "--test-ratio",
            str(job_request.test_ratio),
            "--grid-iter",
            str(job_request.grid_iter),
        ]
        if job_request.mlflow_uri:
            args.extend(["--mlflow-uri", job_request.mlflow_uri])
        if job_request.artifact_object_uri:
            args.extend(["--artifact-object-uri", job_request.artifact_object_uri])

        self._append_manifest_root(args, job_request)
        self._invoke(models_command, args, job_request)

        outputs = [job_request.prediction_output_path, job_request.model_output_path]
        return tuple(path for path in outputs if path is not None)

    def _append_manifest_root(
        self,
        args: list[str],
        job_request: StepJobRequest,
    ) -> None:
        if job_request.manifest_root:
            args.extend(["--artifact-manifest-root", job_request.manifest_root])

        repository_root = os.getenv("ARTIFACT_REPOSITORY_ROOT", ".")
        args.extend(["--artifact-repository-root", repository_root])

    def _invoke(
        self,
        command: Command,
        args: list[str],
        job_request: StepJobRequest,
    ) -> None:
        env = _execution_env(job_request)
        try:
            with _patched_environ(env):
                command.main(
                    args=args,
                    prog_name=command.name,
                    standalone_mode=False,
                )
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 1
            if code == 0:
                return
            raise _execution_error(job_request, str(error)) from error
        except Exception as error:
            raise _execution_error(job_request, str(error)) from error

    def _build_manifest_reference(
        self,
        job_request: StepJobRequest,
    ) -> ArtifactManifestReference | None:
        if isinstance(job_request, ModelJobRequest) and job_request.expected_manifest:
            return job_request.expected_manifest
        if not job_request.manifest_root:
            return None

        artifact_type = _artifact_type_for_job(job_request.job_type)
        manifest_path = (
            f"{job_request.manifest_root}/{artifact_type.value}/"
            f"{job_request.counter_id}/{job_request.run_id}/manifest.json"
        )
        current_path = (
            f"{job_request.manifest_root}/{artifact_type.value}/"
            f"{job_request.counter_id}/current.json"
        )
        object_uri = None
        if isinstance(job_request, ModelJobRequest):
            object_uri = job_request.artifact_object_uri

        return ArtifactManifestReference(
            artifact_type=artifact_type,
            counter_id=job_request.counter_id,
            run_id=job_request.run_id,
            manifest_path=manifest_path,
            current_path=current_path,
            object_uri=object_uri,
        )


def _execution_env(job_request: StepJobRequest) -> dict[str, str]:
    env = {
        "RUN_ID": job_request.run_id,
        "COUNTER_ID": job_request.counter_id,
        "ARTIFACT_PRODUCER_SERVICE": f"ml-{job_request.job_type.value}",
    }
    if job_request.dag_id:
        env["AIRFLOW_CTX_DAG_ID"] = job_request.dag_id
    if job_request.task_id:
        env["AIRFLOW_CTX_TASK_ID"] = job_request.task_id
    if job_request.manifest_root:
        env["ARTIFACT_MANIFEST_ROOT"] = job_request.manifest_root

    return env


def _execution_error(
    job_request: StepJobRequest,
    message: str,
) -> MlStepExecutionError:
    if not message:
        message = "ML step execution failed without output."
    return MlStepExecutionError(
        code=f"{job_request.job_type.value.upper()}_JOB_FAILED",
        message=message,
        retryable=True,
    )


@contextmanager
def _patched_environ(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _artifact_type_for_job(job_type: MlJobType) -> ArtifactType:
    artifact_types = {
        MlJobType.INGEST: ArtifactType.INTERIM_DATASET,
        MlJobType.FEATURES: ArtifactType.FEATURE_DATASET,
        MlJobType.MODELS: ArtifactType.PREDICTIONS,
    }
    return artifact_types[job_type]


def _path_parent_name(path: str) -> str:
    parent_name = PurePosixPath(path).parent.name
    return parent_name or "counter"


def _model_sub_dir(job_request: ModelJobRequest) -> str:
    if job_request.prediction_output_path:
        return _path_parent_name(job_request.prediction_output_path)

    return _path_parent_name(job_request.processed_input_path)
