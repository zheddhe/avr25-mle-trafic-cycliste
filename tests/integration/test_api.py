# tests/integration/test_api.py
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.config import ApiSettings
from src.artifacts.checksums import compute_sha256

COUNTER_ID = "Sebastopol_N-S"
RUN_ID = "api-integration-run"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _auth_headers(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _write_prediction_payload(repository_root: Path) -> Path:
    payload_path = repository_root / "data" / "final" / COUNTER_ID / "y_full.csv"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(
        {
            "date_et_heure_de_comptage_local": [
                "2025-09-23 08:00:00",
                "2025-09-23 09:00:00",
            ],
            "date_et_heure_de_comptage_utc": [
                "2025-09-23 06:00:00",
                "2025-09-23 07:00:00",
            ],
            "y_true": [123, 130],
            "y_pred": [120.5, 129.5],
            "forecast_mode": [False, False],
        }
    )
    dataframe.to_csv(payload_path)
    return payload_path


def _write_current_manifest(
    *,
    manifest_root: Path,
    repository_root: Path,
    payload_path: Path,
    counter_id: str = COUNTER_ID,
) -> Path:
    local_path = payload_path.relative_to(repository_root).as_posix()
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "predictions",
        "status": "promoted",
        "run_id": RUN_ID,
        "counter_id": counter_id,
        "created_at": "2026-06-06T14:00:00+00:00",
        "producer": {
            "service": "ml-models-prod",
            "image": "bike-traffic/ml-models-prod:test",
            "version": "test",
        },
        "source": {
            "raw_file_name": "comptage-velo.csv",
            "dataset_version": "test-dataset",
            "model_version": "test-model",
        },
        "storage": {
            "primary_backend": "local",
            "local_path": local_path,
            "checksum_sha256": compute_sha256(payload_path),
        },
    }
    current_path = manifest_root / "predictions" / counter_id / "current.json"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps(manifest), encoding="utf-8")
    return current_path


def _build_client(tmp_path: Path) -> TestClient:
    repository_root = tmp_path / "repository"
    manifest_root = repository_root / "artifacts" / "manifests"
    payload_path = _write_prediction_payload(repository_root)
    _write_current_manifest(
        manifest_root=manifest_root,
        repository_root=repository_root,
        payload_path=payload_path,
    )
    settings = ApiSettings(
        manifest_root=manifest_root,
        repository_root=repository_root,
        counter_ids=(),
    )
    return TestClient(create_app(settings=settings))


def _build_empty_client(tmp_path: Path) -> TestClient:
    repository_root = tmp_path / "repository"
    settings = ApiSettings(
        manifest_root=repository_root / "artifacts" / "manifests",
        repository_root=repository_root,
        counter_ids=(),
    )
    return TestClient(create_app(settings=settings))


# ---------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------
@pytest.mark.integration
class TestApiIntegration:
    def test_health_is_public(self, tmp_path: Path) -> None:
        with _build_empty_client(tmp_path) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_verify_admin_access(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.post(
                "/admin/refresh",
                headers=_auth_headers("admin1", "admin1"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "Store refreshed by admin1."

    def test_verify_forbidden_for_user(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.post(
                "/admin/refresh",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 403

    def test_me_endpoint(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.get(
                "/me",
                headers=_auth_headers("user2", "user2"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "user2"
        assert body["role"] == "user"
        assert body["permissions"]["prediction_endpoints"] is True

    def test_authenticated_endpoint_requires_credentials(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.get("/counters")

        assert response.status_code == 401

    def test_counters_return_promoted_manifest_counter(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.get(
                "/counters",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 200
        assert response.json() == [{"id": COUNTER_ID}]

    def test_counters_empty_predictions(self, tmp_path: Path) -> None:
        with _build_empty_client(tmp_path) as client:
            response = client.get(
                "/counters",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 418
        body = response.json()
        assert body["type"] == "PredictionsNotLoaded"

    def test_refresh_admin_loads_promoted_manifest(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.post(
                "/admin/refresh",
                headers=_auth_headers("admin1", "admin1"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["counters_after"] == 1
        assert body["loaded"] == 1
        assert list(Path(body["manifest_root"]).parts[-2:]) == [
            "artifacts",
            "manifests",
        ]

    def test_refresh_admin_reports_missing_manifest(self, tmp_path: Path) -> None:
        with _build_empty_client(tmp_path) as client:
            response = client.post(
                "/admin/refresh",
                headers=_auth_headers("admin1", "admin1"),
            )

        assert response.status_code == 418
        body = response.json()
        assert body["type"] == "ArtifactManifestNotFoundError"

    def test_get_predictions_with_data(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.get(
                f"/predictions/{COUNTER_ID}",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert len(body["item"]) == 2
        assert body["item"][0]["y_true"] == 123
        assert pytest.approx(body["item"][0]["y_pred"], rel=1e-6) == 120.5

    def test_get_predictions_supports_pagination(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.get(
                f"/predictions/{COUNTER_ID}?limit=1&offset=1",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["limit"] == 1
        assert body["offset"] == 1
        assert len(body["item"]) == 1
        assert body["item"][0]["y_true"] == 130

    def test_get_predictions_rejects_invalid_pagination(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.get(
                f"/predictions/{COUNTER_ID}?limit=101",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 422

    def test_get_predictions_unknown_counter(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.get(
                "/predictions/unknown-counter",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 418
        body = response.json()
        assert body["type"] == "CounterUnavailable"
        assert COUNTER_ID in body["message"]

    def test_current_artifacts_expose_sanitized_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        with _build_client(tmp_path) as client:
            response = client.get(
                "/artifacts/current",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["counter_id"] == COUNTER_ID
        assert body[0]["run_id"] == RUN_ID
        assert body[0]["primary_backend"] == "local"
        assert body[0]["object_uri"] is None

    def test_current_artifact_by_counter(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.get(
                f"/artifacts/current/{COUNTER_ID}",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["counter_id"] == COUNTER_ID
        assert body["source"]["dataset_version"] == "test-dataset"

    def test_current_artifact_unknown_counter(self, tmp_path: Path) -> None:
        with _build_client(tmp_path) as client:
            response = client.get(
                "/artifacts/current/unknown-counter",
                headers=_auth_headers("user1", "user1"),
            )

        assert response.status_code == 418
        body = response.json()
        assert body["type"] == "ArtifactUnavailable"
