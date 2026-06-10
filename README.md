# 🚲 Cyclist Traffic MLOps Project

[![CI Main](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_main.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)
[![CI Branch](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_branch.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)

Machine learning and MLOps project for daily refreshed bike traffic prediction in
Paris.

The project was developed as part of the April 2025 Machine Learning Engineering
training program. It combines data processing, time-series modelling, API
serving, orchestration, tracking, monitoring, and local production-like runtime
validation.

![Local production-like architecture overview](docs/assets/diagrams/local-prod-architecture-overview.png)

This rendered diagram is an onboarding illustration. The maintained architecture
contract is documented in Mermaid diagrams, service tables, and runtime contract
notes under [`docs/`](docs/).

## Project scope

The MLOps architecture is organized around one business goal:

> an external user can access daily refreshed bike traffic predictions.

The implementation currently provides:

- reproducible local Python tooling with uv;
- DVC-oriented development data and model workspaces;
- FastAPI prediction serving from promoted artifact manifests;
- ML pipeline containers for ingestion, features, and modelling;
- manifest-first artifact handoff for pipeline outputs;
- Airflow orchestration for multi-counter workflows;
- MLflow tracking with PostgreSQL and MinIO;
- Prometheus, Grafana, Pushgateway, Alertmanager, cAdvisor, and MailHog;
- explicit `docker/dev` and `docker/prod` Compose runtimes;
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
├── logs/               <- Development runtime logs
├── models/             <- Development and DVC model artifacts
├── src/                <- API and ML pipeline source code
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
| Remaining work | `docs/remaining-work/` | Future improvement axes that are deliberately outside the validated local production-like baseline. |

Key entrypoints:

- [`docs/current-runtime-and-operations/local-prod-runtime.md`](docs/current-runtime-and-operations/local-prod-runtime.md)
- [`docs/current-runtime-and-operations/ports-and-services.md`](docs/current-runtime-and-operations/ports-and-services.md)
- [`docs/current-runtime-and-operations/repository-structure.md`](docs/current-runtime-and-operations/repository-structure.md)
- [`docs/architecture-references/runtime-communication-matrix.md`](docs/architecture-references/runtime-communication-matrix.md)
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
| Development | `docker/dev/docker-compose.yaml` | `dev-*` | Debugging, broad host visibility, DVC/local workspaces, and current Airflow DockerOperator jobs. |
| Local production-like | `docker/prod/docker-compose.yaml` | `prod-*` | Reduced host exposure, functional networks, non-root custom services, runner-backed ML steps, manifest-first API serving, and acceptance validation. |

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
make prod-clean
```

The default profile is `ptf`, combining MLflow, Airflow, monitoring, and API
services. Use `DEV_PROFILE=api` or `PROD_PROFILE=api` for targeted startup.

## ML pipeline, artifacts, and tracking

Development one-off pipeline containers:

```bash
make dev-mlops-ingest
make dev-mlops-features
make dev-mlops-models
make dev-mlops-pipeline
```

Local host-side execution:

```bash
make local-pipeline
make mlflow-local
```

Pipeline steps can emit and promote artifact manifests when
`ARTIFACT_MANIFEST_ROOT` or `--artifact-manifest-root` is configured. This keeps
local/DVC runs unchanged by default while allowing production-like runtimes to
publish validated artifact metadata and checksums.

The prediction API serves from promoted prediction manifests. It reads
`predictions/<counter_id>/current.json`, resolves the referenced local payload
through `ARTIFACT_REPOSITORY_ROOT`, verifies the checksum when present, and no
longer scans `data/final` to infer the current prediction file.

MLflow environment presets:

```bash
eval "$(make --no-print-directory env-compose)"
eval "$(make --no-print-directory env-local)"
eval "$(make --no-print-directory env-dagshub)"
```

## Orchestration and monitoring

Airflow orchestrates multi-counter workflows:

- `bike_traffic_init`: historical bootstrap per counter;
- `bike_traffic_daily`: rolling increment after initialization;
- `bike_traffic_orchestrator`: orchestrates configured counters.

In the production-like runtime, Airflow calls the internal `job-runner-api`, which
then delegates typed `ingest`, `features`, and `models` jobs to internal ML step
FastAPI services. The API is refreshed only after the model step has produced and
promoted prediction artifacts.

Monitoring uses Prometheus, Grafana, Pushgateway, cAdvisor, Alertmanager, and
MailHog. Batch metrics are pushed from ML step services to Pushgateway and then
scraped by Prometheus. Production-like Grafana dashboards are provisioned from
`docker/prod/grafana/dashboards`.

## Current baseline and remaining work

The local production-like baseline now covers the validated path:

```text
Airflow DAG task
  -> job-runner-api
  -> ml-ingest-prod / ml-features-prod / ml-models-prod
  -> promoted prediction manifest
  -> authenticated API refresh
  -> FastAPI serving from the promoted manifest payload
  -> Prometheus and Grafana observability
```

Known remaining work is tracked as global improvement axes rather than as a
phase-specific design target. See
[`docs/remaining-work/global-remaining-work.md`](docs/remaining-work/global-remaining-work.md)
for security hardening, scale-out execution, full ETL source chain,
object-storage-first artifact serving, remote deployment, and production
operations.

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
