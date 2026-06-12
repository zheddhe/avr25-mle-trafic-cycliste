# tests/acceptance/test_prod_api_manifest_smoke.py
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest

COUNTER_ID_ENV = "ACCEPTANCE_COUNTER_ID"
PROMOTABLE_STATUSES = {"produced", "validated", "promoted"}
RequiredEnvVar = Callable[[str], str]
OptionalEnvVar = Callable[[str], str | None]


@pytest.mark.acceptance
class TestProdApiManifestSmoke:
    def test_authenticated_refresh_loads_prediction_manifests(
        self,
        required_env_var: RequiredEnvVar,
    ) -> None:
        refresh = _refresh_prediction_store(required_env_var)
        artifacts = _current_artifacts(required_env_var)

        assert refresh["loaded"] >= 1
        assert artifacts, "The API did not expose any current prediction artifact."

    def test_prediction_manifest_is_current_publishable_manifest(
        self,
        required_env_var: RequiredEnvVar,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        artifact = _acceptance_artifact(required_env_var, optional_env_var)
        counter_id = _acceptance_counter_id_from_artifacts(
            required_env_var,
            optional_env_var,
        )

        assert artifact["counter_id"] == counter_id
        assert artifact["artifact_type"] == "predictions"
        assert artifact["status"] in PROMOTABLE_STATUSES
        assert artifact["primary_backend"] == "local"
        assert artifact["local_path"]
        assert artifact["checksum_sha256"]

    def test_authenticated_refresh_and_prediction_endpoints(
        self,
        required_env_var: RequiredEnvVar,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        refresh = _refresh_prediction_store(required_env_var)
        counters = _request_json(
            "GET",
            "/counters",
            username=required_env_var("API_USER"),
            password=required_env_var("API_PASS"),
            api_url=required_env_var("API_URL"),
        )
        artifacts = _current_artifacts(required_env_var)
        counter_id = _acceptance_counter_id_from_api(
            counters,
            required_env_var,
            optional_env_var,
        )
        predictions = _request_json(
            "GET",
            f"/predictions/{counter_id}?limit=1&offset=0",
            username=required_env_var("API_USER"),
            password=required_env_var("API_PASS"),
            api_url=required_env_var("API_URL"),
        )

        counter_ids = {counter["id"] for counter in counters}
        artifact_counter_ids = {artifact["counter_id"] for artifact in artifacts}

        assert refresh["loaded"] >= 1
        assert counter_id in counter_ids
        assert counter_id in artifact_counter_ids
        assert predictions["total"] >= 1
        assert predictions["limit"] == 1
        assert predictions["item"]


def _refresh_prediction_store(
    required_env_var: RequiredEnvVar,
) -> dict[str, Any]:
    return _request_json(
        "POST",
        "/admin/refresh",
        username=required_env_var("API_ADMIN_USER"),
        password=required_env_var("API_ADMIN_PASS"),
        api_url=required_env_var("API_URL"),
    )


def _current_artifacts(required_env_var: RequiredEnvVar) -> list[dict[str, Any]]:
    artifacts = _request_json(
        "GET",
        "/artifacts/current",
        username=required_env_var("API_USER"),
        password=required_env_var("API_PASS"),
        api_url=required_env_var("API_URL"),
    )
    assert isinstance(artifacts, list), f"Unexpected artifacts response: {artifacts}"
    return artifacts


def _acceptance_artifact(
    required_env_var: RequiredEnvVar,
    optional_env_var: OptionalEnvVar,
) -> dict[str, Any]:
    counter_id = _acceptance_counter_id_from_artifacts(
        required_env_var,
        optional_env_var,
    )
    artifact = _request_json(
        "GET",
        f"/artifacts/current/{counter_id}",
        username=required_env_var("API_USER"),
        password=required_env_var("API_PASS"),
        api_url=required_env_var("API_URL"),
    )
    assert isinstance(artifact, dict), f"Unexpected artifact response: {artifact}"
    return artifact


def _acceptance_counter_id_from_artifacts(
    required_env_var: RequiredEnvVar,
    optional_env_var: OptionalEnvVar,
) -> str:
    configured_counter_id = optional_env_var(COUNTER_ID_ENV)
    artifacts = _current_artifacts(required_env_var)
    counter_ids = sorted(artifact["counter_id"] for artifact in artifacts)

    if configured_counter_id:
        assert configured_counter_id in counter_ids, (
            f"Configured {COUNTER_ID_ENV}={configured_counter_id} is not served. "
            f"Available counters: {counter_ids}"
        )
        return configured_counter_id

    assert counter_ids, "The API did not expose any current artifact."
    return counter_ids[0]


def _acceptance_counter_id_from_api(
    counters: list[dict[str, Any]],
    required_env_var: RequiredEnvVar,
    optional_env_var: OptionalEnvVar,
) -> str:
    configured_counter_id = optional_env_var(COUNTER_ID_ENV)
    counter_ids = sorted(counter["id"] for counter in counters)
    if configured_counter_id:
        assert configured_counter_id in counter_ids, (
            f"Configured {COUNTER_ID_ENV}={configured_counter_id} is not served. "
            f"Available counters: {counter_ids}"
        )
        return configured_counter_id

    manifest_counter_id = _acceptance_counter_id_from_artifacts(
        required_env_var,
        optional_env_var,
    )
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
    api_url: str,
) -> Any:
    token = base64.b64encode(f"{username}:{password}".encode())
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
