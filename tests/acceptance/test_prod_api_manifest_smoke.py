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
COUNTER_ID_ENV = "ACCEPTANCE_COUNTER_ID"
PROMOTABLE_STATUSES = {"produced", "validated", "promoted"}


@pytest.mark.acceptance
class TestProdApiManifestSmoke:
    def test_expected_current_manifests_exist(self) -> None:
        counter_id = _acceptance_counter_id()
        expected_paths = [
            _current_manifest_path("interim_dataset", counter_id),
            _current_manifest_path("feature_dataset", counter_id),
            _current_manifest_path("predictions", counter_id),
        ]

        missing_paths = [str(path) for path in expected_paths if not path.is_file()]

        assert not missing_paths, (
            "The production-like pipeline has not produced all expected "
            f"current manifests for {counter_id}: {missing_paths}"
        )

    def test_prediction_manifest_is_current_publishable_manifest(self) -> None:
        counter_id = _acceptance_counter_id()
        manifest = _read_current_manifest("predictions", counter_id)

        assert manifest["counter_id"] == counter_id
        assert manifest["artifact_type"] == "predictions"
        assert manifest["status"] in PROMOTABLE_STATUSES
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
        counter_id = _acceptance_counter_id_from_api(counters)
        predictions = _request_json(
            "GET",
            f"/predictions/{counter_id}?limit=1&offset=0",
            username=API_USER,
            password=API_PASSWORD,
        )

        counter_ids = {counter["id"] for counter in counters}
        artifact_counter_ids = {artifact["counter_id"] for artifact in artifacts}

        assert refresh["loaded"] >= 1
        assert counter_id in counter_ids
        assert counter_id in artifact_counter_ids
        assert predictions["total"] >= 1
        assert predictions["limit"] == 1
        assert predictions["item"]


def _acceptance_counter_id() -> str:
    configured_counter_id = os.getenv(COUNTER_ID_ENV)
    if configured_counter_id:
        return configured_counter_id

    current_manifests = sorted((MANIFEST_ROOT / "predictions").glob("*/current.json"))
    if not current_manifests:
        pytest.fail(
            "No prediction current manifest found. Run the production-like "
            f"Airflow chain before make acceptance, or set {COUNTER_ID_ENV}."
        )

    return current_manifests[0].parent.name


def _acceptance_counter_id_from_api(counters: list[dict[str, Any]]) -> str:
    configured_counter_id = os.getenv(COUNTER_ID_ENV)
    counter_ids = sorted(counter["id"] for counter in counters)
    if configured_counter_id:
        assert configured_counter_id in counter_ids, (
            f"Configured {COUNTER_ID_ENV}={configured_counter_id} is not served. "
            f"Available counters: {counter_ids}"
        )
        return configured_counter_id

    manifest_counter_id = _acceptance_counter_id()
    if manifest_counter_id in counter_ids:
        return manifest_counter_id

    assert counter_ids, "The API did not expose any counter after refresh."
    return counter_ids[0]


def _current_manifest_path(artifact_type: str, counter_id: str) -> Path:
    return MANIFEST_ROOT / artifact_type / counter_id / "current.json"


def _read_current_manifest(artifact_type: str, counter_id: str) -> dict[str, Any]:
    path = _current_manifest_path(artifact_type, counter_id)
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
