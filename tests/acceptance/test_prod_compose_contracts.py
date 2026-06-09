# tests/acceptance/test_prod_compose_contracts.py
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPOSITORY_ROOT / "docker" / "prod" / "docker-compose.yaml"
DEFAULT_PROJECT_NAME = "bike-traffic-prod"

AIRFLOW_SERVICES = {
    "airflow-api-server",
    "airflow-scheduler",
    "airflow-dag-processor",
    "airflow-worker",
    "airflow-triggerer",
}
INTERNAL_ONLY_SERVICES = {
    "job-runner-api",
    "ml-ingest-prod",
    "ml-features-prod",
    "ml-models-prod",
}


def _compose_command(*args: str) -> list[str]:
    project_name = os.getenv("PROD_PROJECT_NAME", DEFAULT_PROJECT_NAME)
    return [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "-p",
        project_name,
        *args,
    ]


def _run_compose(*args: str) -> str:
    command = _compose_command(*args)
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        pytest.fail("Docker Compose is required for acceptance tests.")
        raise AssertionError from error
    except subprocess.CalledProcessError as error:
        pytest.fail(
            "Docker Compose command failed: "
            f"{' '.join(command)}\n{error.stderr}"
        )
        raise AssertionError from error

    return completed.stdout


def _load_compose_config() -> dict[str, Any]:
    return json.loads(
        _run_compose("--profile", "all", "config", "--format", "json")
    )


@pytest.mark.acceptance
class TestProdComposeContracts:
    def test_prod_runtime_services_are_running(self) -> None:
        output = _run_compose("--profile", "all", "ps", "--format", "json")
        rows = [json.loads(line) for line in output.splitlines() if line.strip()]
        running_services = {
            row["Service"]
            for row in rows
            if str(row.get("State", "")).lower() == "running"
        }

        required_services = AIRFLOW_SERVICES | INTERNAL_ONLY_SERVICES | {"api-prod"}
        missing_services = sorted(required_services - running_services)

        assert not missing_services, (
            "Production-like runtime is not fully running. "
            f"Missing services: {missing_services}"
        )

    def test_airflow_services_do_not_mount_docker_socket(self) -> None:
        config = _load_compose_config()
        services = config["services"]

        for service_name in AIRFLOW_SERVICES:
            volumes = services[service_name].get("volumes", [])
            serialized_volumes = json.dumps(volumes)
            assert "/var/run/docker.sock" not in serialized_volumes

    def test_runner_and_ml_step_services_are_internal_only(self) -> None:
        config = _load_compose_config()
        services = config["services"]

        for service_name in INTERNAL_ONLY_SERVICES:
            ports = services[service_name].get("ports", [])
            assert ports == [], f"{service_name} exposes host ports: {ports}"
