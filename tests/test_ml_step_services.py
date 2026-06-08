"""Tests for internal ML step FastAPI services."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.ml.jobs.contracts import IngestJobRequest, MlJobType
from src.ml.jobs.status import JobResult
from src.ml.services.api import MlStepService, create_app


def _build_ingest_request() -> IngestJobRequest:
    return IngestJobRequest(
        run_id="run-001",
        counter_id="counter-001",
        manifest_root="/app/artifacts/manifests",
        raw_path="/app/data/raw/source.csv",
        site="Totem 73 boulevard de Sébastopol",
        orientation="N-S",
        sub_dir="counter-001",
        interim_output_path="/app/data/interim/counter-001/initial.csv",
    )


class FakeStepExecutor:
    """Test double returning deterministic ML step evidence."""

    def __init__(self) -> None:
        self.requests: list[IngestJobRequest] = []

    def execute(self, job_request, *, job_id, started_at):
        self.requests.append(job_request)
        return JobResult(
            job_id=job_id,
            run_id=job_request.run_id,
            counter_id=job_request.counter_id,
            job_type=job_request.job_type,
            started_at=started_at,
            output_paths=(job_request.interim_output_path,),
        )


class TestMlStepService:
    """Validate the internal ML step service contract."""

    def test_post_job_returns_succeeded_status(self) -> None:
        executor = FakeStepExecutor()
        service = MlStepService(job_type=MlJobType.INGEST, executor=executor)
        app = create_app(
            service_name="ml-ingest-prod",
            job_type=MlJobType.INGEST,
            service=service,
        )
        client = TestClient(app)

        response = client.post(
            "/jobs",
            json=_build_ingest_request().model_dump(mode="json"),
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["state"] == "succeeded"
        assert payload["job_type"] == "ingest"
        assert payload["result"]["output_paths"] == [
            "/app/data/interim/counter-001/initial.csv",
        ]
        assert len(executor.requests) == 1

    def test_post_wrong_job_type_is_rejected(self) -> None:
        app = create_app(
            service_name="ml-features-prod",
            job_type=MlJobType.FEATURES,
        )
        client = TestClient(app)

        response = client.post(
            "/jobs",
            json=_build_ingest_request().model_dump(mode="json"),
        )

        assert response.status_code == 422
        assert "features" in response.json()["detail"]
