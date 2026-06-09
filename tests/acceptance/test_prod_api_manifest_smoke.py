# tests/acceptance/test_prod_api_manifest_smoke.py
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = (
    REPOSITORY_ROOT / "docker" / "prod" / "runtime" / "artifacts" / "manifests"
)
API_URL = os.getenv("ACCEPTANCE_API_URL", "http://localhost:10000")
API_ADMIN_USER = os.getenv("API_ADMIN_USER", "admin1")
API_ADMIN_PASSWORD = os.getenv("API_ADMIN_PASSWORD", "admin1")
API_USER = os.getenv("API_USER", "user1")
API_PASSWORD = os.getenv("API_PASS", "user1")
ACCEPTANCE_COUNTER_ID = os.getenv(
    "ACCEPTANCE_COUNTER_ID",
    "Sebastopol_N-S_airflow_day0",
)


@pytest.mark.acceptance
class TestProdApiManifestSmoke:
    def test_expected_current_manifests_exist(self) -> None:
        expected_paths = [
            _current_manifest_path("interim_dataset"),
            _current_manifest_path("feature_dataset"),
            _current_manifest_path("predictions"),
        ]

        missing_paths = [str(path) for path in expected_paths if not path.is_file()]

        assert not missing_paths, (
            "The production-like pipeline has not produced all expected "
            f"current manifests for {ACCEPTANCE_COUNTER_ID}: {missing_paths}"
        )

    def test_prediction_manifest_is_promoted_current_manifest(self) -> None:
        manifest = _read_current_manifest("predictions")

        assert manifest["counter_id"] == ACCEPTANCE_COUNTER_ID
        assert manifest["artifact_type"] == "predictions"
        assert manifest["status"] == "promoted"
        assert manifest["storage"]["primary_backend"] == "local"
        assert manifest["storage"].get("local_path")

    def test_authenticated_refresh_and_prediction_endpoints(self) -> None:
        refresh = _request_json(
            "POST",
            "/admin/refresh",
            username=API_ADMIN_USER,
            password=API_ADMIN_PASSWORD,
        )
        counters = _request_json(
            "GET",
            "/counters",
            username=API_USER,
            password=API_PASSWORD,
        )
        artifacts = _request_json(
            "GET",
            "/artifacts/current",
            username=API_USER,
            password=API_PASSWORD,
        )
        predictions = _request_json(
            "GET",
            f"/predictions/{ACCEPTANCE_COUNTER_ID}?limit=1&offset=0",
            username=API_USER,
            password=API_PASSWORD,
        )

        counter_ids = {counter["id"] for counter in counters}
        artifact_counter_ids = {artifact["counter_id"] for artifact in artifacts}

        assert refresh["loaded"] >= 1
        assert ACCEPTANCE_COUNTER_ID in counter_ids
        assert ACCEPTANCE_COUNTER_ID in artifact_counter_ids
        assert predictions["total"] >= 1
        assert predictions["limit"] == 1
        assert predictions["item"]


def _current_manifest_path(artifact_type: str) -> Path:
    return MANIFEST_ROOT / artifact_type / ACCEPTANCE_COUNTER_ID / "current.json"


def _read_current_manifest(artifact_type: str) -> dict[str, Any]:
    path = _current_manifest_path(artifact_type)
    return json.loads(path.read_text(encoding="utf-8"))


def _request_json(
    method: str,
    path: str,
    *,
    username: str,
    password: str,
) -> Any:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8"))
    request = urllib.request.Request(
        url=f"{API_URL.rstrip('/')}/{path.lstrip('/')}",
        method=method,
        headers={"Authorization": f"Basic {token.decode('ascii')}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        pytest.fail(f"API returned HTTP {error.code} for {path}: {body}")
        raise AssertionError from error
    except urllib.error.URLError as error:
        pytest.fail(f"API is not reachable at {API_URL}: {error}")
        raise AssertionError from error

    return json.loads(body)
