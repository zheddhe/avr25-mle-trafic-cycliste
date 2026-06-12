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
COUNTER_ID_ENV = "ACCEPTANCE_COUNTER_ID"
PROMOTABLE_STATUSES = {"produced", "validated", "promoted"}


@pytest.mark.acceptance
class TestProdApiManifestSmoke:
    def test_authenticated_refresh_loads_prediction_manifests(self) -> None:
        refresh = _refresh_prediction_store()
        artifacts = _current_artifacts()

        assert refresh["loaded"] >= 1
        assert artifacts, "The API did not expose any current prediction artifact."

    def test_prediction_manifest_is_current_publishable_manifest(self) -> None:
        artifact = _acceptance_artifact()

        assert artifact["counter_id"] == _acceptance_counter_id_from_artifacts()
        assert artifact["artifact_type"] == "predictions"
        assert artifact["status"] in PROMOTABLE_STATUSES
        assert artifact["primary_backend"] == "local"
        assert artifact["local_path"]
        assert artifact["checksum_sha256"]

    def test_authenticated_refresh_and_prediction_endpoints(self) -> None:
        refresh = _refresh_prediction_store()
        counters = _request_json(
            "GET",
            "/counters",
            username=_setting("API_USER"),
            password=_setting("API_PASS"),
        )
        artifacts = _current_artifacts()
        counter_id = _acceptance_counter_id_from_api(counters)
        predictions = _request_json(
            "GET",
            f"/predictions/{counter_id}?limit=1&offset=0",
            username=_setting("API_USER"),
            password=_setting("API_PASS"),
        )

        counter_ids = {counter["id"] for counter in counters}
        artifact_counter_ids = {artifact["counter_id"] for artifact in artifacts}

        assert refresh["loaded"] >= 1
        assert counter_id in counter_ids
        assert counter_id in artifact_counter_ids
        assert predictions["total"] >= 1
        assert predictions["limit"] == 1
        assert predictions["item"]


def _env_file_path() -> Path:
    env_file = Path(os.getenv("ENV_FILE", ".env"))
    if env_file.is_absolute():
        return env_file

    return REPOSITORY_ROOT / env_file


def _dotenv_values() -> dict[str, str]:
    env_file = _env_file_path()
    if not env_file.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        values[key.strip()] = raw_value.strip().strip('"').strip("'")

    return values


def _setting(name: str) -> str:
    return os.getenv(name) or _dotenv_values().get(name, "")


def _api_url() -> str:
    explicit_url = os.getenv("API_URL")
    if explicit_url:
        return explicit_url
    else:
        return "undefined_api_url"


def _refresh_prediction_store() -> dict[str, Any]:
    return _request_json(
        "POST",
        "/admin/refresh",
        username=_setting("API_ADMIN_USER"),
        password=_setting("API_ADMIN_PASS"),
    )


def _current_artifacts() -> list[dict[str, Any]]:
    artifacts = _request_json(
        "GET",
        "/artifacts/current",
        username=_setting("API_USER"),
        password=_setting("API_PASS"),
    )
    assert isinstance(artifacts, list), f"Unexpected artifacts response: {artifacts}"
    return artifacts


def _acceptance_artifact() -> dict[str, Any]:
    counter_id = _acceptance_counter_id_from_artifacts()
    artifact = _request_json(
        "GET",
        f"/artifacts/current/{counter_id}",
        username=_setting("API_USER"),
        password=_setting("API_PASS"),
    )
    assert isinstance(artifact, dict), f"Unexpected artifact response: {artifact}"
    return artifact


def _acceptance_counter_id_from_artifacts() -> str:
    configured_counter_id = os.getenv(COUNTER_ID_ENV)
    artifacts = _current_artifacts()
    counter_ids = sorted(artifact["counter_id"] for artifact in artifacts)

    if configured_counter_id:
        assert configured_counter_id in counter_ids, (
            f"Configured {COUNTER_ID_ENV}={configured_counter_id} is not served. "
            f"Available counters: {counter_ids}"
        )
        return configured_counter_id

    assert counter_ids, "The API did not expose any current artifact."
    return counter_ids[0]


def _acceptance_counter_id_from_api(counters: list[dict[str, Any]]) -> str:
    configured_counter_id = os.getenv(COUNTER_ID_ENV)
    counter_ids = sorted(counter["id"] for counter in counters)
    if configured_counter_id:
        assert configured_counter_id in counter_ids, (
            f"Configured {COUNTER_ID_ENV}={configured_counter_id} is not served. "
            f"Available counters: {counter_ids}"
        )
        return configured_counter_id

    manifest_counter_id = _acceptance_counter_id_from_artifacts()
    if manifest_counter_id in counter_ids:
        return manifest_counter_id

    assert counter_ids, "The API did not expose any counter after refresh."
    return counter_ids[0]


def _request_json(
    method: str,
    path: str,
    *,
    username: str,
    password: str,
) -> Any:
    token = base64.b64encode(f"{username}:{password}".encode())
    api_url = _api_url()
    request = urllib.request.Request(
        url=f"{api_url.rstrip('/')}/{path.lstrip('/')}",
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
        pytest.fail(f"API is not reachable at {api_url}: {error}")
        raise AssertionError from error

    return json.loads(body)
