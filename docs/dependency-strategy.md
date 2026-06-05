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

| Service image | Dockerfile | Requirements file | Python baseline |
| ------------- | ---------- | ----------------- | --------------- |
| API | `docker/dev/api/Dockerfile` | `docker/dev/api/requirements.txt` | Python 3.12 |
| ML ingest | `docker/dev/ml/ingest/Dockerfile` | `docker/dev/ml/ingest/requirements.txt` | Python 3.12 |
| ML features | `docker/dev/ml/features/Dockerfile` | `docker/dev/ml/features/requirements.txt` | Python 3.12 |
| ML models | `docker/dev/ml/models/Dockerfile` | `docker/dev/ml/models/requirements.txt` | Python 3.12 |

The standard baseline for custom images is Python 3.12. A different Python
version should only be used if a service-specific dependency constraint requires
it and the reason is documented.

## Compose runtime image policy

Runtime image variables are centralized in `.env.template` and consumed by
`docker-compose.yaml` with `:?required` guards. They keep readable version tags
for local developer ergonomics and for tools such as the VS Code Docker
extension.

Validated content digests are documented here when known. A future production
hardening phase may replace readable tags with direct `@sha256:` image
references.

| Variable | Runtime image | Validated digest or note |
| -------- | ------------- | ------------------------ |
| `AIRFLOW_IMAGE_NAME` | `apache/airflow:3.2.2-python3.12` | Digest not recorded yet |
| `AIRFLOW_POSTGRES_IMAGE` | `postgres:16` | Digest not recorded yet |
| `MLFLOW_IMAGE` | `ghcr.io/mlflow/mlflow:v3.13.0-full` | `sha256:45bdcc9439dac5c51c160a863e3c1cadae1757de9d6d1b9403e0a648a6f2333b` |
| `MLFLOW_POSTGRES_IMAGE` | `postgres:16` | Digest not recorded yet |
| `MINIO_IMAGE` | `minio/minio:RELEASE.2025-09-07T16-13-09Z` | `sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e` |
| `MINIO_MC_IMAGE` | `minio/mc:RELEASE.2025-08-13T08-35-41Z` | `sha256:a7fe349ef4bd8521fb8497f55c6042871b2ae640607cf99d9bede5e9bdf11727` |
| `PROMETHEUS_IMAGE` | `prom/prometheus:v3.12.0` | Digest not recorded yet |
| `GRAFANA_IMAGE` | `grafana/grafana:13.0.1-security-01` | Digest not recorded yet |
| `PUSHGATEWAY_IMAGE` | `prom/pushgateway:v1.11.3` | Digest not recorded yet |
| `ALERTMANAGER_IMAGE` | `prom/alertmanager:v0.32.1` | Digest not recorded yet |
| `CADVISOR_IMAGE` | `gcr.io/cadvisor/cadvisor:v0.57.0` | Digest not recorded yet |

## Compose managed service families

Some services are provided by upstream images and are not governed by the local
uv environment.

| Service family | Current image policy |
| -------------- | -------------------- |
| Airflow | Uses `AIRFLOW_IMAGE_NAME`. Local `apache-airflow` in uv is only for local import support and should not be treated as the container runtime version. |
| Airflow PostgreSQL | Uses `AIRFLOW_POSTGRES_IMAGE` as the local Airflow metadata database. |
| Redis | Uses an upstream Redis image for the Airflow Celery broker. |
| MLflow server | Uses `MLFLOW_IMAGE` from the upstream GHCR MLflow image. |
| MLflow PostgreSQL | Uses `MLFLOW_POSTGRES_IMAGE` as the local MLflow metadata database. |
| MinIO | Uses `MINIO_IMAGE` and `MINIO_MC_IMAGE` for local MLflow artifact storage. |
| Monitoring | Uses Prometheus, Grafana, Pushgateway, Alertmanager, and cAdvisor through image variables. |
| Development helpers | Uses upstream MailHog for local email capture. |

## Runtime baseline

The local MLOps stack currently uses these major runtime versions:

| Component | Runtime version |
| --------- | --------------- |
| Airflow | `3.2.2` with Python 3.12 image |
| MLflow | `3.13.0` server and model client |
| PostgreSQL | `16` for Airflow and MLflow metadata stores |
| MinIO | `RELEASE.2025-09-07T16-13-09Z` |
| MinIO client | `RELEASE.2025-08-13T08-35-41Z` |
| Prometheus | `3.12.0` |
| Grafana | `13.0.1-security-01` |
| Pushgateway | `1.11.3` |
| Alertmanager | `0.32.1` |
| cAdvisor | `0.57.0` |

## Healthchecks

Docker Compose healthchecks should rely on endpoints or commands that are
available in the upstream image itself. Avoid assuming that `curl` exists unless
it is part of the selected image.

| Service | Healthcheck strategy |
| ------- | -------------------- |
| `mlflow-postgres` | `pg_isready` against the MLflow metadata database. |
| `mlflow-minio` | MinIO live endpoint `/minio/health/live`. |
| `mlflow-server` | MLflow `/health` endpoint through Python `urllib`. |
| `airflow-postgres` | `pg_isready` against the Airflow metadata database. |
| `airflow-redis` | `redis-cli ping`. |
| `airflow-api-server` | Airflow REST `/api/v2/version`. |
| `airflow-scheduler` | `airflow jobs check --job-type SchedulerJob`. |
| `airflow-dag-processor` | `airflow jobs check --job-type DagProcessorJob`. |
| `airflow-triggerer` | `airflow jobs check --job-type TriggererJob`. |
| `airflow-flower` | Celery executor app status check. |
| `monitoring-prometheus` | Prometheus readiness endpoint `/-/ready`. |
| `monitoring-grafana` | Grafana API health endpoint `/api/health`. |
| `monitoring-pushgateway` | Prometheus-style readiness endpoint `/-/ready`. |
| `monitoring-alertmanager` | Prometheus-style readiness endpoint `/-/ready`. |
| `monitoring-cadvisor` | cAdvisor health endpoint `/healthz`. |

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
make test
make compose-config
make build
make start PROFILE=ptf
make mlops-pipeline
make logs SERVICE=api-dev
```

Then verify:

- prediction files are produced under `data/final`;
- model artifacts are produced under `models`;
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
