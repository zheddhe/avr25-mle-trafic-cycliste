# Dependency strategy

This document records the dependency strategy used for the local Python
environment, custom Docker images, and Docker Compose runtime services.

## Goals

The project must remain reproducible without silently changing machine learning
behavior, API contracts, MLflow artifact compatibility, or containerized runtime
behavior.

Dependency changes are split into two categories:

- tooling changes, which can usually be upgraded with local CI validation;
- runtime changes, which require explicit service-level and model-level
  regression checks.

## Local uv environment

The local Python environment is managed by uv and is not used to package this
repository as a distributable Python package.

The dependency groups have the following responsibilities:

| Group | Responsibility |
| ----- | -------------- |
| `app` | Local import and execution surface for API, DAGs, ML scripts, MLflow, metrics, and orchestration code. |
| `test` | Test and lint harness. Includes `app`. |
| `dev` | Development harness. Includes `test` and DVC tooling. |

The local uv environment is primarily used for:

- IDE and Pylance import resolution;
- local tests and static checks;
- local script execution;
- host-side experimentation.

It is not the production runtime source of truth. Runtime containers keep their
own minimal requirements files or use upstream images.

## Custom Docker images

Custom images are the runtime source of truth for services built by this
repository.

| Service image | Runtime | Dockerfile | Requirements file | Python baseline |
| ------------- | ------- | ---------- | ----------------- | --------------- |
| API | Dev | `docker/dev/api/Dockerfile` | `docker/dev/api/requirements.txt` | Python 3.12 |
| ML ingest | Dev | `docker/dev/ml/ingest/Dockerfile` | `docker/dev/ml/ingest/requirements.txt` | Python 3.12 |
| ML features | Dev | `docker/dev/ml/features/Dockerfile` | `docker/dev/ml/features/requirements.txt` | Python 3.12 |
| ML models | Dev | `docker/dev/ml/models/Dockerfile` | `docker/dev/ml/models/requirements.txt` | Python 3.12 |
| API | Prod-like | `docker/prod/api/Dockerfile` | `docker/dev/api/requirements.txt` | Python 3.12 |
| ML ingest | Prod-like | `docker/prod/ml/ingest/Dockerfile` | `docker/dev/ml/ingest/requirements.txt` | Python 3.12 |
| ML features | Prod-like | `docker/prod/ml/features/Dockerfile` | `docker/dev/ml/features/requirements.txt` | Python 3.12 |
| ML models | Prod-like | `docker/prod/ml/models/Dockerfile` | `docker/dev/ml/models/requirements.txt` | Python 3.12 |

The standard baseline for custom images is Python 3.12. A different Python
version should only be used if a service-specific dependency constraint requires
it and the reason is documented.

The prod-like Dockerfiles currently reuse the same requirements files as the dev
images to avoid dependency drift while the runtime boundary is being introduced.
A future hardening story may split requirements only when the operational need is
clear and validated.

## Compose runtime image policy

Runtime image variables are centralized in `.env.template` and consumed by the
Compose entrypoints with `:?required` guards. They keep readable version tags for
local developer ergonomics and for tools such as the VS Code Docker extension.

Validated content digests are documented here when known. A future production
hardening phase may replace readable tags with direct `@sha256:` image
references.

| Variable | Runtime image | Validated digest or note |
| -------- | ------------- | ------------------------ |
| `AIRFLOW_IMAGE_NAME` | `apache/airflow:3.2.2-python3.12` | `sha256:bbe58e3204d550ab98dbf738a42c0e6663c455357ecd0e2d1440ef9cb6a75f00` |
| `AIRFLOW_POSTGRES_IMAGE` | `postgres:16` | `sha256:4b7183ac05f8ef417db21fd72d71047a4238340c261d3cc3ddb6d579ab5071ae` |
| `AIRFLOW_REDIS_IMAGE` | `redis:latest` | `sha256:aa049e689e141a4358ad1d4562dc49c88a89fbab711fd8fcc33f684c80b26301` |
| `MLFLOW_IMAGE` | `ghcr.io/mlflow/mlflow:v3.13.0-full` | `sha256:45bdcc9439dac5c51c160a863e3c1cadae1757de9d6d1b9403e0a648a6f2333b` |
| `MLFLOW_POSTGRES_IMAGE` | `postgres:16` | `sha256:4b7183ac05f8ef417db21fd72d71047a4238340c261d3cc3ddb6d579ab5071ae` |
| `MLFLOW_MINIO_IMAGE` | `minio/minio:RELEASE.2025-09-07T16-13-09Z` | `sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e` |
| `MLFLOW_MINIO_MC_IMAGE` | `minio/mc:RELEASE.2025-08-13T08-35-41Z` | `sha256:a7fe349ef4bd8521fb8497f55c6042871b2ae640607cf99d9bede5e9bdf11727` |
| `PROMETHEUS_IMAGE` | `prom/prometheus:v3.12.0` | `sha256:69f5241418838263316593f7274a304b095c40bcf22e57272865da91bd60a8ac` |
| `GRAFANA_IMAGE` | `grafana/grafana:13.0.2` | `sha256:5dad0df181cb644a14e13617b913b261a54f7d4fd4510721dba420929f35bea2` |
| `PUSHGATEWAY_IMAGE` | `prom/pushgateway:v1.11.3` | `sha256:5dad0df181cb644a14e13617b913b261a54f7d4fd4510721dba420929f35bea2` |
| `ALERTMANAGER_IMAGE` | `prom/alertmanager:v0.32.1` | `sha256:51a825c2a40acc3e338fdd00d622e01ec090f72be2b3ea46be0839cd47a4d286` |
| `CADVISOR_IMAGE` | `gcr.io/cadvisor/cadvisor:v0.55.1` | `sha256:3de2bd5203120b866d74a9b283b2ffb8ec382fbf9dc321814700c6ea6f44ec57` |
| `MAILHOG_IMAGE` | `mailhog/mailhog` | `sha256:8d76a3d4ffa32a3661311944007a415332c4bb855657f4f6c57996405c009bea` |

## Compose managed service families

Some services are provided by upstream images and are not governed by the local
uv environment.

| Service family | Current image policy |
| -------------- | -------------------- |
| Orchestration services | Uses Airflow, PostgreSQL, and Redis images |
| Experimentation services | Uses MLflow, PostgreSQL, MinIO, and MC images |
| Monitoring services | Uses Prometheus, Grafana, Pushgateway, Alertmanager, and cAdvisor images |
| Development helpers | Uses MailHog images |

## Healthchecks

Docker Compose healthchecks should rely on endpoints or commands that are
available in the upstream image itself. Avoid assuming that `curl` exists unless
it is part of the selected image.

| Service | Healthcheck strategy |
| ------- | -------------------- |
| `api-dev` | Python `urllib` read check on `/docs`. |
| `mlflow-postgres` | `pg_isready` against the MLflow metadata database. |
| `mlflow-minio` | curl check on `/minio/health/live`. |
| `mlflow-server` | Python `urllib` read check on `/health`. |
| `airflow-postgres` | `pg_isready` against the Airflow metadata database. |
| `airflow-redis` | `redis-cli ping`. |
| `airflow-api-server` | curl check on `/api/v2/monitor/health`. |
| `airflow-scheduler` | `airflow jobs check --job-type SchedulerJob`. |
| `airflow-dag-processor` | `airflow jobs check --job-type DagProcessorJob`. |
| `airflow-worker` | Celery ping. The dev worker executes the check as the `airflow` user because its entrypoint starts as root only to adjust Docker socket access. |
| `airflow-triggerer` | `airflow jobs check --job-type TriggererJob`. |
| `airflow-flower` | curl check on `/` in the dev runtime. |
| `monitoring-prometheus` | wget check on `/-/ready`. |
| `monitoring-grafana` | wget check on `/api/health`. |
| `monitoring-pushgateway` | wget check on `/-/ready`. |
| `monitoring-alertmanager` | wget check on `/-/ready`. |
| `monitoring-cadvisor` | wget check on `/healthz`. |
| `monitoring-mailhog` | wget check on `/`. |

## Runtime-sensitive dependencies

The following dependencies are runtime-sensitive and should not be upgraded as a
bulk maintenance action:

- `pandas`
- `scikit-learn`
- `xgboost`
- `scikit-optimize`
- `fastapi`
- `uvicorn`
- `mlflow`
- `prometheus-client`
- `prometheus-fastapi-instrumentator`
- Airflow image versions
- monitoring runtime images

Runtime-sensitive upgrades must be isolated and validated with a dedicated PR or
explicit validation section.

## MLflow compatibility

The ML models container acts as an MLflow client and the MLflow server container
acts as the tracking server. Their versions do not need to be byte-for-byte
identical, but compatibility must be intentionally reviewed before changing either
side.

For this project, MLflow compatibility checks should include:

- successful experiment creation;
- parameter and metric logging;
- model artifact logging;
- artifact retrieval from the configured backend;
- local Docker Compose tracking through MinIO and PostgreSQL;
- optional DagsHub remote tracking when credentials are available.

## Monitoring compatibility

Monitoring runtime changes should validate at least:

- Prometheus configuration parsing;
- Prometheus targets under `/targets`;
- API metrics scraping;
- Pushgateway scraping;
- cAdvisor scraping;
- Grafana datasource provisioning;
- Grafana dashboard loading;
- Alertmanager configuration loading;
- MailHog alert notification capture when alert routing is tested.

## Regression checklist for runtime dependency changes

Before merging a runtime dependency upgrade, validate at least:

```bash
make sync
make lint
make tests
make dev-compose-config
make prod-compose-config
make dev-build
make prod-build
make dev-start DEV_PROFILE=ptf
make prod-start PROD_PROFILE=ptf
make dev-mlops-pipeline
make dev-logs SERVICE=api-dev
```

Then verify:

- prediction files are produced under root `data/final` for the development
  runtime;
- production-like generated artifacts stay under `docker/prod/runtime`;
- model artifacts are produced under root `models` for the development runtime;
- MLflow runs contain expected parameters, metrics, and artifacts;
- API `/docs` is reachable;
- API prediction endpoints still return the expected schema;
- Prometheus targets are healthy and ML/API metrics are visible;
- Grafana starts and loads the provisioned Prometheus datasource;
- Grafana dashboards load with current Prometheus metrics;
- Prometheus metrics are not pushed during local tests when
  `DISABLE_METRICS_PUSH=1`;
- Docker Compose services start without dependency resolution errors.

## Upgrade policy

Tooling upgrades may be merged when CI and local validation pass.

Runtime upgrades require a short compatibility note in the PR body describing:

- what changed;
- which service is affected;
- expected risks;
- validation evidence;
- rollback strategy.
