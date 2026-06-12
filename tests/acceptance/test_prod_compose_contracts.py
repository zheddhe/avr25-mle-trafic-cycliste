# tests/acceptance/test_prod_compose_contracts.py
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPOSITORY_ROOT / "docker" / "prod" / "docker-compose.yaml"
DEFAULT_PROJECT_NAME = "bike-traffic-prod"
PIPELINE_RUNTIME_NETWORK = "prod_pipeline_runtime_net"
OBSERVABILITY_NETWORK = "prod_observability_net"
DEFAULT_PUSHGATEWAY_ADDR = "monitoring-pushgateway:9091"

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
ML_STEP_SERVICES = {
    "ml-ingest-prod",
    "ml-features-prod",
    "ml-models-prod",
}
PIPELINE_RUNTIME_SERVICES = ML_STEP_SERVICES | {"job-runner-api"}
CONTAINER_ENGINE_SOCKET = "/var/run/" + "docker.sock"
OptionalEnvVar = Callable[[str], str | None]


def _compose_command(
    *args: str,
    optional_env_var: OptionalEnvVar,
) -> list[str]:
    project_name = optional_env_var("PROD_PROJECT_NAME") or DEFAULT_PROJECT_NAME
    command = ["docker", "compose"]
    env_file = optional_env_var("ENV_FILE") or ".env"
    env_file_path = Path(env_file)

    if not env_file_path.is_absolute():
        env_file_path = REPOSITORY_ROOT / env_file_path

    if env_file_path.is_file():
        command.extend(["--env-file", str(env_file_path)])

    command.extend(
        [
            "-f",
            str(COMPOSE_FILE),
            "-p",
            project_name,
            *args,
        ]
    )
    return command


def _run_compose(
    *args: str,
    optional_env_var: OptionalEnvVar,
) -> str:
    command = _compose_command(*args, optional_env_var=optional_env_var)
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            cwd=REPOSITORY_ROOT,
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


def _load_compose_config(
    optional_env_var: OptionalEnvVar,
) -> dict[str, Any]:
    return json.loads(
        _run_compose(
            "--profile",
            "all",
            "config",
            "--format",
            "json",
            optional_env_var=optional_env_var,
        )
    )


@pytest.mark.acceptance
class TestProdComposeContracts:
    def test_prod_runtime_services_are_running(
        self,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        output = _run_compose(
            "--profile",
            "all",
            "ps",
            "--format",
            "json",
            optional_env_var=optional_env_var,
        )
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

    def test_airflow_services_do_not_mount_container_engine_socket(
        self,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        config = _load_compose_config(optional_env_var)
        services = config["services"]

        for service_name in AIRFLOW_SERVICES:
            volumes = services[service_name].get("volumes", [])
            serialized_volumes = json.dumps(volumes)
            assert CONTAINER_ENGINE_SOCKET not in serialized_volumes

    def test_runner_and_ml_step_services_are_internal_only(
        self,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        config = _load_compose_config(optional_env_var)
        services = config["services"]

        for service_name in INTERNAL_ONLY_SERVICES:
            ports = services[service_name].get("ports", [])
            assert ports == [], f"{service_name} exposes host ports: {ports}"

    def test_runner_and_ml_step_services_share_runtime_network(
        self,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        config = _load_compose_config(optional_env_var)
        services = config["services"]

        for service_name in PIPELINE_RUNTIME_SERVICES:
            networks = services[service_name].get("networks", {})
            assert PIPELINE_RUNTIME_NETWORK in networks, (
                f"{service_name} is not attached to "
                f"{PIPELINE_RUNTIME_NETWORK}: {networks}"
            )

    def test_ml_step_services_join_observability_network(
        self,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        config = _load_compose_config(optional_env_var)
        services = config["services"]

        for service_name in ML_STEP_SERVICES:
            networks = services[service_name].get("networks", {})
            assert OBSERVABILITY_NETWORK in networks, (
                f"{service_name} is not attached to "
                f"{OBSERVABILITY_NETWORK}: {networks}"
            )

    def test_ml_step_services_push_metrics_to_internal_pushgateway(
        self,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        config = _load_compose_config(optional_env_var)
        services = config["services"]

        for service_name in ML_STEP_SERVICES:
            environment = services[service_name]["environment"]
            assert environment["PUSHGATEWAY_ADDR"] == DEFAULT_PUSHGATEWAY_ADDR
            assert str(environment["DISABLE_METRICS_PUSH"]) == "0"

    def test_pushgateway_is_scraped_by_prometheus(
        self,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        config = _load_compose_config(optional_env_var)
        services = config["services"]
        prometheus_volumes = services["monitoring-prometheus"].get("volumes", [])
        pushgateway_networks = services["monitoring-pushgateway"].get(
            "networks",
            {},
        )

        assert OBSERVABILITY_NETWORK in pushgateway_networks
        assert PIPELINE_RUNTIME_NETWORK in pushgateway_networks
        assert any("prometheus" in volume["source"] for volume in prometheus_volumes)

    def test_prod_grafana_mounts_prod_dashboards(
        self,
        optional_env_var: OptionalEnvVar,
    ) -> None:
        config = _load_compose_config(optional_env_var)
        services = config["services"]
        volumes = services["monitoring-grafana"].get("volumes", [])
        serialized_volumes = json.dumps(volumes)

        assert "/docker/prod/grafana/dashboards" in serialized_volumes
        assert "/docker/dev/grafana/dashboards" not in serialized_volumes
