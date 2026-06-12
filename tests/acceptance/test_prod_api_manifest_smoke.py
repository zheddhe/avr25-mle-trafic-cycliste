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
from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COUNTER_ID_ENV = "ACCEPTANCE_COUNTER_ID"
PROMOTABLE_STATUSES = {"produced", "validated", "promoted"}
REQUIRED_ACCEPTANCE_SETTINGS = {
    "API_URL",
    "API_USER",
    "API_PASS",
    "API_ADMIN_USER",
    "API_ADMIN_PASS",
}
ENV_LOADED = False


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
            username=_required_setting("API_USER"),
            password=_required_setting("API_PASS"),
        )
        artifacts = _current_artifacts()
        counter_id = _acceptance_counter_id_from_api(counters)
        predictions = _request_json(
            "GET",
            f"/predictions/{counter_id}?limit=1&offset=0",
            username=_required_setting("API_USER"),
            password=_required_setting("API_PASS"),
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


def _load_acceptance_env() -> None:
    global ENV_LOADED

    if ENV_LOADED:
        return

    env_file = _env_file_path()
    if env_file.is_file():
        load_dotenv(dotenv_path=env_file, override=True)

    ENV_LOADED = True


def _required_setting(name: str) -> str:
    _load_acceptance_env()
    value = os.getenv(name)
    if value:
        return value

    required_settings = ", ".join(sorted(REQUIRED_ACCEPTANCE_SETTINGS))
    pytest.fail(
        f"Missing required acceptance variable {name}. "
        f"Define it in {_env_file_path()} or export it in the shell. "
        f"Required variables: {required_settings}."
    )
    raise AssertionError


def _optional_setting(name: str) -> str | None:
    _load_acceptance_env()
    return os.getenv(name) or None


def _api_url() -> str:
    return _required_setting("API_URL")


def _refresh_prediction_store() -> dict[str, Any]:
    return _request_json(
        "POST",
        "/admin/refresh",
        username=_required_setting("API_ADMIN_USER"),
        password=_required_setting("API_ADMIN_PASS"),
    )


def _current_artifacts() -> list[dict[str, Any]]:
    artifacts = _request_json(
        "GET",
        "/artifacts/current",
        username=_required_setting("API_USER"),
        password=_required_setting("API_PASS"),
    )
    assert isinstance(artifacts, list), f"Unexpected artifacts response: {artifacts}"
    return artifacts


def _acceptance_artifact() -> dict[str, Any]:
    counter_id = _acceptance_counter_id_from_artifacts()
    artifact = _request_json(
        "GET",
        f"/artifacts/current/{counter_id}",
        username=_required_setting("API_USER"),
        password=_required_setting("API_PASS"),
    )
    assert isinstance(artifact, dict), f"Unexpected artifact response: {artifact}"
    return artifact


def _acceptance_counter_id_from_artifacts() -> str:
    configured_counter_id = _optional_setting(COUNTER_ID_ENV)
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
    configured_counter_id = _optional_setting(COUNTER_ID_ENV)
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
