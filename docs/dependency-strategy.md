# Dependency strategy

This document records the dependency strategy used for the local Python
environment, custom Docker images, and Docker Compose services.

## Goals

The project must remain reproducible without silently changing machine learning
behavior, API contracts, MLflow artifact compatibility, or containerized runtime
behavior.

Dependency changes must therefore be split into two categories:

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
own minimal requirements files.

## Custom Docker images

Custom images are the runtime source of truth for the project services built by
this repository.

| Service image | Dockerfile | Requirements file | Python baseline |
| ------------- | ---------- | ----------------- | --------------- |
| API | `docker/dev/api/Dockerfile` | `docker/dev/api/requirements.txt` | Python 3.12 |
| ML ingest | `docker/dev/ml/ingest/Dockerfile` | `docker/dev/ml/ingest/requirements.txt` | Python 3.12 |
| ML features | `docker/dev/ml/features/Dockerfile` | `docker/dev/ml/features/requirements.txt` | Python 3.12 |
| ML models | `docker/dev/ml/models/Dockerfile` | `docker/dev/ml/models/requirements.txt` | Python 3.12 |
| MLflow server | `docker/dev/mlflow/Dockerfile` | inline pip install | Python 3.12 |

The standard baseline for custom images is Python 3.12. A different Python
version should only be used if a service-specific dependency constraint requires
it and the reason is documented.

## Docker Compose managed services

Some services are provided by upstream images and are not governed by the local
uv environment.

| Service family | Current image policy |
| -------------- | -------------------- |
| Airflow | Uses `apache/airflow:3.2.2-python3.12` in Docker Compose. Local `apache-airflow` in uv is only for local import support and should not be treated as the container runtime version. |
| Airflow PostgreSQL | Uses `postgres:16` as the local Airflow metadata database. |
| Redis | Uses an upstream Redis image for the Airflow Celery broker. |
| PostgreSQL | Uses an upstream PostgreSQL image for the MLflow metadata store. |
| MinIO | Uses upstream MinIO images for the local MLflow artifact store. |
| Monitoring | Uses upstream Prometheus, Grafana, cAdvisor, Pushgateway, Alertmanager, and MailHog images. |

## Phase 6 runtime upgrade status

Story #51 upgrades the MLOps infrastructure stack in focused increments. The
current branch has validated the Airflow increment first because Airflow 2 was
blocking DAG parsing through an obsolete Python runtime.

| Component | Target in story #51 | Current branch status |
| --------- | ------------------- | --------------------- |
| Airflow | `3.2.2` | Upgraded and validated with Python 3.12 image |
| MLflow | `3.13.0` | Pending follow-up increment |
| Prometheus | `3.12.0` | Pending follow-up increment |
| Grafana | `13.0.1` | Pending follow-up increment |

Do not document pending targets as active runtime versions in the README until
the related Compose and validation commits have landed.

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
