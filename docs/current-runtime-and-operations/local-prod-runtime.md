# Local Compose runtimes

This document describes the implemented local Docker Compose runtime model. It is
a current-state operational guide for `docker/dev` and `docker/prod`.

The two runtimes now share the same functional execution topology:

```text
Airflow DAG
  -> job-runner-api
  -> ml-gateway
  -> ml-ingest-* / ml-features-* / ml-models-*
  -> promoted artifact manifests
  -> authenticated API refresh
```

The important runtime differences are intentional and limited to local exposure,
workspace ownership, image tags, runtime users, and debug ergonomics.

## When to use each runtime

| Runtime | Entry point | Primary use |
| ------- | ----------- | ----------- |
| Development | `docker/dev/docker-compose.yaml` | Full local runtime with visible host ports, host bind-mounted `docker/dev/runtime`, and the same runner/gateway/ML-service path as prod-like. |
| Local production-like | `docker/prod/docker-compose.yaml` | Reduced host exposure, named Docker runtime volume, explicit init service, internal runner/gateway/ML services, and production-like boundary checks. |

Use `docker/dev` for day-to-day MLOps development, DAG debugging, service logs,
OpenAPI inspection, MLflow inspection, metrics debugging, and parallel local runs
with `docker/prod`.

Use `docker/prod` for production-like validation of the service topology,
runtime volume ownership, internal-only services, reduced host exposure, and
manifest-first serving behavior.

Root `data`, `models`, and `logs` remain local experimentation and DVC
workspaces. Compose-driven runtime outputs must stay under the runtime-specific
workspace or volume described below.

## Operational commands

The root Makefile includes the dedicated runtime Makefiles:

- `docker/dev/Makefile` for local development Compose operations;
- `docker/prod/Makefile` for local production-like Compose operations.

Common validation and startup commands:

```bash
make dev-compose-config
make dev-build
make dev-start
make dev-ps

make prod-compose-config
make prod-build
make prod-start
make prod-ps
```

Scale ML replicas through the gateway path:

```bash
make dev-scale-ml ML_INGEST_REPLICAS=2 ML_FEATURES_REPLICAS=2 ML_MODELS_REPLICAS=2
make prod-scale-ml ML_INGEST_REPLICAS=2 ML_FEATURES_REPLICAS=2 ML_MODELS_REPLICAS=2
```

Inspect the production-like runtime Docker volume with:

```bash
make prod-dir-runtime
```

Runtime-scoped cleanup commands:

```bash
make dev-clean
make prod-clean
```

## Runtime workspace strategy

The development runtime uses host-visible bind mounts under `docker/dev/runtime`.
This keeps debugging simple and makes generated files easy to inspect from the
host.

| Development path | Purpose |
| ---------------- | ------- |
| `docker/dev/runtime/data` | Generated raw, interim, processed, and final runtime data. |
| `docker/dev/runtime/models` | Runtime model artifacts written by `ml-models-dev`. |
| `docker/dev/runtime/logs` | API, Airflow, ML service, and runner logs. |
| `docker/dev/runtime/artifacts` | Manifest-first artifact handoff root. |

The production-like runtime uses the named Docker volume `prod-runtime`. The
`init-volumes` service creates the expected subdirectories, seeds the raw CSV
from root `data/raw`, and applies runtime ownership before dependent services
start.

| Production-like volume path | Purpose |
| --------------------------- | ------- |
| `prod-runtime:/data` | Generated production-like data workspace. |
| `prod-runtime:/models` | Production-like model artifacts. |
| `prod-runtime:/logs` | Production-like service, Airflow, ML, and runner logs. |
| `prod-runtime:/artifacts` | Manifest-first artifact handoff root. |

The required business source CSV is still owned by the root/DVC workspace and is
seeded into the production-like runtime volume by `init-volumes`:

```text
data/raw/comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv
```

## Shared execution boundary

Both runtimes expose the same typed execution boundary:

- Airflow loads a file-based DAG configuration from
  `airflow/config/bike_dag_config.json`.
- `bike_traffic_orchestrator` enumerates configured counters.
- `bike_traffic_init` and `bike_traffic_daily` submit typed jobs to
  `job-runner-api`.
- `job-runner-api` accepts only `ingest`, `features`, and `models` jobs.
- `ml-gateway` routes those requests to the matching ML step service.
- ML step services write data, logs, models, metrics, and promoted manifests.
- The FastAPI prediction API refreshes from promoted prediction manifests.

`job-runner-api` is intentionally not a generic shell runner, Docker SDK wrapper,
Kubernetes controller, or full-pipeline orchestrator. Airflow keeps ordering
responsibility; the runner keeps typed service dispatch and status mapping.

## Runtime services

| Service family | Development | Production-like |
| -------------- | ----------- | --------------- |
| Prediction API | `api-dev`, host-exposed on `API_HOST_PORT_DEV`. | `api-prod`, host-exposed on `API_HOST_PORT_PROD`. |
| Runner API | `job-runner-api`, internal-only. | `job-runner-api`, internal-only. |
| Gateway | `ml-gateway`, internal-only on `10090`. | `ml-gateway`, internal-only on `10090`. |
| ML services | `ml-ingest-dev`, `ml-features-dev`, `ml-models-dev`. | `ml-ingest-prod`, `ml-features-prod`, `ml-models-prod`. |
| Airflow | Host-exposed UI/API on `AIRFLOW_HOST_PORT_DEV`. | Host-exposed UI/API on `AIRFLOW_HOST_PORT_PROD`. |
| MLflow | Host-exposed on `MLFLOW_HOST_PORT_DEV`. | Host-exposed on `MLFLOW_HOST_PORT_PROD`. |
| MinIO | Console host-exposed for dev debugging only. | Internal-only. |
| Monitoring | Prometheus, Pushgateway, cAdvisor, Grafana, Alertmanager exposed for dev. | Grafana host-exposed; support services internal-only. |
| MailHog | Host-exposed local helper. | Internal support service only when configured. |

## Airflow configuration

Dev and prod-like Airflow are file based. `variables.json` is no longer part of
the normal runtime contract.

| Runtime | DAG config | API refresh connection |
| ------- | ---------- | ---------------------- |
| Development | `docker/dev/airflow/config/bike_dag_config.json` | `api_dev` from `connections.json`. |
| Production-like | `docker/prod/airflow/config/bike_dag_config.json` | `api_prod` from `connections.json`. |

The dev and prod-like DAGs are intentionally aligned. Differences should remain
limited to runtime tags, API connection IDs, service names, and runtime paths.

## MLflow runtime view

Compose services use the internal MLflow service name directly:

```text
http://mlflow-server:5000
```

The previous local `mlops-*` chain is no longer the main path. Host-side MLflow
switching is handled by the root Makefile environment helpers for local unset
mode or DagsHub mode. Compose mode does not need mirrored MLflow target variables
because the runtime service URLs are explicit in Compose.

## Host port strategy

`.env.template` separates dev and prod-like host port ranges so both runtimes can
run in parallel on the same host:

| Range family | Prod-like variables | Dev variables |
| ------------ | ------------------- | ------------- |
| Business API | `API_HOST_PORT_PROD` | `API_HOST_PORT_DEV` |
| Airflow | `AIRFLOW_HOST_PORT_PROD` | `AIRFLOW_HOST_PORT_DEV` |
| MLflow | `MLFLOW_HOST_PORT_PROD` | `MLFLOW_HOST_PORT_DEV` |
| Grafana | `GRAFANA_HOST_PORT_PROD` | `GRAFANA_HOST_PORT_DEV` |
| Dev-only tools | n/a | MinIO console, Prometheus, Pushgateway, cAdvisor, Alertmanager, MailHog. |

Detailed ports are documented in
[`ports-and-services.md`](ports-and-services.md).

## Manifest-first API serving

The prediction API loads promoted prediction manifests from:

```text
<manifest_root>/predictions/<counter_id>/current.json
```

The API does not infer current predictions by scanning `data/final`. It resolves
`storage.local_path` from `ARTIFACT_REPOSITORY_ROOT`, verifies the local payload
checksum when present, and exposes current artifact metadata through the artifact
endpoints.

## Validation checklist

For this runtime slice, validate at least:

```bash
make lint
make tests
make dev-compose-config
make prod-compose-config
make dev-build
make prod-build
```

Then validate runtime behavior with:

```bash
make dev-start DEV_PROFILE=ptf
make prod-start PROD_PROFILE=ptf
make dev-ps
make prod-ps
```

Expected runtime properties:

- dev and prod-like DAGs submit ML work through `job-runner-api`;
- `job-runner-api` reaches ML services only through `ml-gateway`;
- Airflow workers do not mount `/var/run/docker.sock`;
- only cAdvisor keeps the Docker socket observability exception;
- dev writes to host-visible `docker/dev/runtime`;
- prod-like writes to the `prod-runtime` Docker volume;
- prod-like MinIO, Prometheus, Pushgateway, Alertmanager, cAdvisor, MailHog,
  runner, gateway, and ML step services stay internal-only.
