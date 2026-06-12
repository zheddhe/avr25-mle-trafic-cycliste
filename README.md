# 🚲 Cyclist Traffic MLOps Project

[![CI Main](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_main.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)
[![CI Branch](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_branch.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)

Machine learning and MLOps project for daily refreshed bike traffic prediction in
Paris.

The project was developed as part of the April 2025 Machine Learning Engineering
training program. It combines data processing, time-series modelling, FastAPI
serving, Airflow orchestration, MLflow tracking, Prometheus/Grafana monitoring,
and local production-like runtime validation.

![Local production-like architecture overview](docs/assets/diagrams/local-prod-architecture-overview.png)

This rendered diagram is an onboarding illustration. The maintained architecture
contract is documented in Mermaid diagrams, service tables, and runtime contract
notes under [`docs/`](docs/).

## Project scope

The MLOps architecture is organized around one business goal:

> an external user can access daily refreshed bike traffic predictions.

The current implementation provides:

- reproducible local Python tooling with uv;
- DVC-oriented development data and model workspaces;
- FastAPI prediction serving from promoted artifact manifests;
- ML pipeline services for ingestion, features, and modelling;
- manifest-first artifact handoff for pipeline outputs;
- Airflow orchestration for multi-counter workflows;
- an internal typed `job-runner-api` execution boundary;
- an internal `ml-gateway` for scaled ML step service dispatch;
- MLflow tracking with PostgreSQL and MinIO;
- Prometheus, Grafana, Pushgateway, Alertmanager, cAdvisor, and MailHog;
- explicit `docker/dev` and `docker/prod` Compose runtimes;
- dev/prod-compatible host port ranges for parallel local runs;
- production-like smoke validation for runtime, API, manifests, and monitoring
  wiring.

## Repository layout

```text
avr25-mle-trafic-cycliste/
├── README.md           <- Project entrypoint
├── Makefile            <- Setup, validation, local execution, and runtime targets
├── .env.template       <- Versioned local runtime variable template
├── pyproject.toml      <- Python project, dependency groups, pytest, coverage, Ruff
├── uv.lock             <- uv lockfile for reproducible local validation
├── data/               <- Development and DVC data workspace
├── logs/               <- Local non-Compose developer logs
├── models/             <- Development and DVC model artifacts
├── src/                <- API, runner, ML pipeline, artifacts, and metrics code
├── docker/             <- Dev/prod container architecture and runtime assets
├── docs/               <- Runtime, architecture, assets, and remaining work docs
└── tests/              <- Unit, integration, and acceptance tests
```

Documentation assets such as icons and rendered diagrams live under
[`docs/assets/`](docs/assets/). Runtime outputs, model artifacts, data payloads,
and generated reports must not be stored there.

Start with [`docs/README.md`](docs/README.md) for the documentation map.

## Documentation map

The documentation is intentionally split into four groups:

| Area | Folder | Purpose |
| ---- | ------ | ------- |
| Current runtime and operations | `docs/current-runtime-and-operations/` | How to run, validate, and reason about implemented local runtimes. |
| Architecture references | `docs/architecture-references/` | Runtime communication, security boundaries, and implemented network topology. |
| Documentation assets | `docs/assets/` | Versioned icons and rendered diagrams used by Markdown documentation. |
| Remaining work | `docs/remaining-work/` | Future improvement axes outside the current local runtime baseline. |

Key entrypoints:

- [`docs/current-runtime-and-operations/local-prod-runtime.md`](docs/current-runtime-and-operations/local-prod-runtime.md)
- [`docs/current-runtime-and-operations/ports-and-services.md`](docs/current-runtime-and-operations/ports-and-services.md)
- [`docs/current-runtime-and-operations/repository-structure.md`](docs/current-runtime-and-operations/repository-structure.md)
- [`docs/architecture-references/runtime-communication-matrix.md`](docs/architecture-references/runtime-communication-matrix.md)
- [`docs/architecture-references/runtime-security-boundaries.md`](docs/architecture-references/runtime-security-boundaries.md)
- [`docs/architecture-references/local-prod-network-topology.md`](docs/architecture-references/local-prod-network-topology.md)
- [`docs/remaining-work/global-remaining-work.md`](docs/remaining-work/global-remaining-work.md)

## First setup

From a fresh machine, install the bootstrap tools before cloning if needed:

```bash
sudo apt update
sudo apt install --fix-missing
sudo apt install -y python3 python3-pip pipx git
pipx ensurepath
pipx install uv
```

Clone the repository:

```bash
git clone git@github.com:zheddhe/avr25-mle-trafic-cycliste.git
cd avr25-mle-trafic-cycliste
```

Create local configuration:

```bash
make env
```

Then replace `[replace_me]` placeholders in `.env` before running targets that
need secrets or user-specific values.

Configure Git and local DVC credentials when needed:

```bash
make git-setup
make dvc-setup
```

For a first Docker installation on Ubuntu:

```bash
make docker-install
```

Open a new shell, or run `newgrp docker`, after Docker installation.

## Local validation

Common repository-level checks:

```bash
make sync
make lock-check
make lint
make unit
make integration
make tests
make checks
```

Production-like acceptance tests require the local production-like runtime and
its required environment values:

```bash
make prod-compose-config
make prod-build
make prod-start
make acceptance
```

Useful cleanup targets:

```bash
make clean-repo
make clean-env
make clean-docker
```

`clean-docker` is a global Docker garbage-collection helper. Runtime-scoped
cleanup is handled by `dev-clean` and `prod-clean`.

## Compose runtimes

The project has two explicit runtime entrypoints.

| Runtime | Compose file | Make targets | Main purpose |
| ------- | ------------ | ------------ | ------------ |
| Development | `docker/dev/docker-compose.yaml` | `dev-*` | Full local runtime, broad host visibility, host bind-mounted `docker/dev/runtime`, and the same runner/gateway/ML-service path as prod-like. |
| Local production-like | `docker/prod/docker-compose.yaml` | `prod-*` | Reduced host exposure, functional networks, `prod-runtime` Docker volume, runner/gateway/ML-service path, manifest-first API serving, and acceptance validation. |

Development runtime:

```bash
make dev-compose-config
make dev-build
make dev-start
make dev-ps
make dev-logs SERVICE=api-dev
make dev-clean
```

Local production-like runtime:

```bash
make prod-compose-config
make prod-build
make prod-start
make prod-ps
make prod-logs SERVICE=api-prod
make prod-dir-runtime
make prod-clean
```

The default profile is `ptf`, combining MLflow, Airflow, monitoring, runner,
gateway, ML services, and API services. Use `DEV_PROFILE=api` or
`PROD_PROFILE=api` for targeted startup.

Scale ML step service replicas through the internal gateway path:

```bash
make dev-scale-ml ML_INGEST_REPLICAS=2 ML_FEATURES_REPLICAS=2 ML_MODELS_REPLICAS=2
make prod-scale-ml ML_INGEST_REPLICAS=2 ML_FEATURES_REPLICAS=2 ML_MODELS_REPLICAS=2
```

## ML pipeline, artifacts, and tracking

Local host-side execution remains available for quick experiments outside the
Compose runtimes:

```bash
make local-pipeline
make mlflow-local
```

Compose-driven ML execution is orchestrated by Airflow. In both dev and
prod-like runtimes, Airflow submits typed jobs to `job-runner-api`. The runner
routes job requests through `ml-gateway`, which dispatches to the appropriate ML
step service:

```text
Airflow DAG task
  -> job-runner-api
  -> ml-gateway
  -> ml-ingest-* / ml-features-* / ml-models-*
```

Pipeline steps can emit and promote artifact manifests when
`ARTIFACT_MANIFEST_ROOT` or `--artifact-manifest-root` is configured. This keeps
local/DVC runs unchanged by default while allowing Compose runtimes to publish
validated artifact metadata and checksums.

The prediction API serves from promoted prediction manifests. It reads
`predictions/<counter_id>/current.json`, resolves the referenced local payload
through `ARTIFACT_REPOSITORY_ROOT`, verifies the checksum when present, and does
not scan `data/final` to infer the current prediction file.

MLflow environment helpers:

```bash
eval "$(make --no-print-directory env-compose)"
eval "$(make --no-print-directory env-local)"
eval "$(make --no-print-directory env-dagshub)"
```

Compose services use their internal MLflow service names directly. Host-side
helpers are for local Python or DagsHub-oriented workflows.

## Orchestration and monitoring

Airflow orchestrates multi-counter workflows:

- `bike_traffic_init`: historical bootstrap per counter;
- `bike_traffic_daily`: rolling increment after initialization;
- `bike_traffic_orchestrator`: orchestrates configured counters.

In both Compose runtimes, Airflow calls the internal `job-runner-api`, which then
delegates typed `ingest`, `features`, and `models` jobs through `ml-gateway` to
internal ML step FastAPI services. The prediction API is refreshed only after the
model step has produced and promoted prediction artifacts.

Monitoring uses Prometheus, Grafana, Pushgateway, cAdvisor, Alertmanager, and
MailHog. Batch metrics are pushed from ML step services to Pushgateway and then
scraped by Prometheus. Grafana dashboards are provisioned from runtime-specific
`docker/*/grafana/dashboards` folders.

## Current baseline and remaining work

The current local baseline covers:

```text
Airflow DAG task
  -> job-runner-api
  -> ml-gateway
  -> ML step service
  -> promoted prediction manifest
  -> authenticated API refresh
  -> FastAPI serving from the promoted manifest payload
  -> Prometheus and Grafana observability
```

Development and local production-like runtimes are aligned on the functional
path. The remaining differences are deliberate: host visibility, runtime storage,
service hardening, and support-service exposure.

Known remaining work is tracked as global improvement axes rather than as a
phase-specific design target. See
[`docs/remaining-work/global-remaining-work.md`](docs/remaining-work/global-remaining-work.md)
for security hardening, full ETL source chain, object-storage-first artifact
serving, remote deployment, and production operations.

## Collaboration

- Create one branch per story or bugfix.
- Open a pull request against `main`.
- Keep story-specific implementation notes in the pull request body.
- Update the relevant documentation group when a target design becomes current
  state.

Contributors:

- Rémy Canal
- Elias Djouadi
- Koladé Houessou
- Sofia Bouizzoul
