# 🚲 Cyclist Traffic MLOps Project

[![CI Main](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_main.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)
[![CI Branch](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_branch.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)

> A machine learning pipeline to provide bike traffic prediction in Paris.  
> Developed as part of the April 2025 Machine Learning Engineering (MLE) full training program.

## 🧭 Overview

This project implements a complete machine learning and MLOps architecture in three main stages:

### 1. 📐 Data Product Management

- Define business goals
- Scope the data lifecycle

### 2. 📊 Data Science

- Data collection and preprocessing
- Model development and evaluation
- Time series prediction

### 3. ⚙️ MLOps

- Reproducibility and continuous testing
- Containerization with microservices
- Security awareness
- Monitoring and orchestration
- Scalability

> The MLOps architecture we designed focuses on interactions between components
> to achieve our main business case: an external user can access daily refreshed
> bike traffic predictions.
>
> [![MLOps Architecture](references/Architecture_MLOps.drawio.png)](https://drive.google.com/file/d/1aglCRFaxXRVEEEwtE5ePFnW-vjE8CYa-/view?usp=sharing)

## 🧱 GitHub Structure

```text
avr25-mle-trafic-cycliste/
├── LICENSE             <- MIT license
├── README.md           <- This top-level README for developers using this project
├── Makefile            <- Local developer and Docker Compose operation targets
├── .env.template       <- Template for local secrets and runtime variables
├── pyproject.toml      <- Python project, dependency groups, pytest, coverage, and Ruff configuration
├── uv.lock             <- uv lockfile for the local development environment
├── data                <- Data shared with host (read/write)
├── logs                <- Logs shared with host (read/write)
├── models              <- Model artifacts shared with host (read/write)
├── references          <- Data dictionaries, manuals, other explanatory material
├── src/                <- API and ML pipeline source code
├── docker/             <- Container architecture, Airflow DAGs, configs, and scripts
├── docs/               <- Architecture, operations, and dependency documentation
└── tests/              <- Unit and integration tests
```

## ⚙️ Installation

The installation flow is split in two parts:

1. **Pre-clone bootstrap**, before this repository is available locally.
2. **Post-clone setup**, once the Makefile is available and should be preferred.

### 1. Pre-clone bootstrap

These commands are intentionally shown explicitly because the repository and its
Makefile are not available yet.

#### Optional virtual machine creation on Windows

```powershell
Set-Service -Name WSLService -StartupType Automatic
Start-Service -Name WSLService
Get-Service WSLService
wsl --install -d Ubuntu
```

#### Linux bootstrap dependencies

```bash
sudo apt update
sudo apt install --fix-missing
sudo apt install -y python3 python3-pip pipx git
pipx ensurepath
pipx install uv
```

#### GitHub SSH access and repository cloning

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com

git clone git@github.com:zheddhe/avr25-mle-trafic-cycliste.git
cd avr25-mle-trafic-cycliste
```

### 2. Post-clone project setup

From this point, prefer Makefile targets over raw commands.

```bash
make help
make env
make setup
```

`make env` creates `.env` from `.env.template` if it does not already exist.
Replace every `[replace_me]` placeholder in `.env` before running targets that
need secrets or user-specific values.

The `.env` file is intentionally not tracked by Git because it may contain
secrets. Docker Compose reads `.env` automatically from the repository root.
Makefile targets that need repository secrets can load it internally.

Configure local Git identity and DVC credentials with:

```bash
make git-setup
make dvc-setup
```

`make dvc-setup` does not run `dvc init` on this repository because DVC is
already initialized and `.dvc/config` is versioned. It only writes local DVC S3
credentials to `.dvc/config.local`, which is ignored by `.dvc/.gitignore`.

Use this convenience target when both steps are needed:

```bash
make repo_setup
```

The versioned DVC remote is defined in `.dvc/config`:

```ini
[core]
    remote = origin
['remote "origin"']
    url = s3://dvc
    endpointurl = https://dagshub.com/zheddhe/avr25-mle-trafic-cycliste.s3
```

The matching local `.env` variables are:

```bash
DAGSHUB_ACCESS_KEY_ID="[replace_me]"
DAGSHUB_SECRET_ACCESS_KEY="[replace_me]"
```

## 🚀 DevOps setup

This repository is not packaged as a Python distribution. The local Python
environment is managed by uv and is used for developer tooling, static analysis,
tests, Pylance import resolution, and local execution of the application code.

### uv dependency groups

| Group | Purpose |
| :---- | :------ |
| `app` | Local application surface: API, DAGs, ML scripts, MLflow, metrics, and orchestration imports. |
| `test` | Test and lint harness. Includes `app`. |
| `dev` | Full development harness. Includes `test` and DVC tooling. |

### Local validation targets

Prefer these Makefile targets for day-to-day local validation:

```bash
make sync
make lock-check
make lint
make test
make ci
```

`make ci` chains the lock check, Ruff linting, and unit tests while excluding
integration tests. This mirrors the local validation path used by CI without
requiring Docker services.

Useful maintenance targets:

```bash
make clean
make clean_env
```

`make clean` removes local Python caches and test artifacts only. It does not
remove Docker volumes. Use `make clean_env` only when you want to recreate the
uv-managed virtual environment from scratch.

## MLOps setup

> This section covers the project setup as a containerized microservices architecture from an MLOps point of view as illustrated in the schemas below.
> [![Docker Compose Overview](references/Docker_Compose_Overview.drawio.png)](https://drive.google.com/file/d/1-C0uL1whFDYXiqkDn20CK2AUF_-S3Ytp/view?usp=drive_link)
> [![Docker Compose Monitoring](references/Docker_Compose_Monitoring.drawio.png)](https://drive.google.com/file/d/14DbcNiD3w7nrdkPiIymbMpX0-vzXRupW/view?usp=drive_link)
>
> Create and customize your own `.env` file from `.env.template` before starting
> the stack. The file contains Docker Compose variables, Makefile defaults,
> local credentials, and optional remote MLflow/DagsHub settings.

### 1. Docker engine setup

For a first Docker installation on Ubuntu, use the dedicated Makefile target
after cloning the repository:

```bash
make docker-install
```

This installs Docker Engine, the Compose plugin, and adds the current user to
the `docker` group. Open a new shell, or run `newgrp docker`, before using Docker.
Docker Desktop can also be used during development on Windows, macOS, or Linux.

### 2. Docker Compose operation targets

The project is assumed to be cloned when running operational commands. Prefer
Makefile targets for project-level Docker operations:

```bash
make compose-config
make build
make ops
```

`make ops` validates the Compose configuration and starts the default platform
profile. The default profile is `ptf`, which combines MLflow, Airflow,
monitoring, and API services.

Targeted operations remain available when needed:

```bash
make start PROFILE=api
make stop PROFILE=api
make logs SERVICE=api-dev
make rebuild_full
make clean_full
```

`make start` uses `docker compose up -d` so it works from a clean environment
where containers have not been created yet. `make clean_full` is intentionally
destructive: it removes Docker images, volumes, and networks.

Runtime host ports and local URLs are documented in
[`docs/ports-and-services.md`](docs/ports-and-services.md). Runtime
service-to-service communication is documented in
[`docs/runtime-communication-matrix.md`](docs/runtime-communication-matrix.md).
Runtime image, healthcheck, and dependency policies are documented in
[`docs/dependency-strategy.md`](docs/dependency-strategy.md).

For one-off ML pipeline containers, use the dedicated targets:

```bash
make mlops-ingest
make mlops-features
make mlops-models
make mlops-pipeline
```

### 3. 📈 Experience tracker

We use **MLflow** to record **metrics**, **params**, and training/prediction
**artifacts** (scikit-learn pipeline, autoregressive transformer, train/test
splits, predictions, metrics, and hyperparameters).

MLflow runtime variables are managed through three explicit presets in
`.env.template`. Each preset maps to the same runtime target variables so the
shell state is deterministic when switching context.

| Mode | Target |
| ---- | ------ |
| `make env-compose` | Docker Compose MLflow server and MinIO artifact store. |
| `make env-local` | Local backend, with MLflow and AWS variables unset. |
| `make env-dagshub` | Remote DagsHub MLflow tracking, with S3/AWS variables unset. |

For interactive shell use, evaluate the desired preset:

```bash
eval "$(make --no-print-directory env-compose)"
eval "$(make --no-print-directory env-local)"
eval "$(make --no-print-directory env-dagshub)"
```

The MLOps container targets automatically use the Compose preset. Local debug
targets automatically use the local backend preset.

```bash
make ops
make mlops-pipeline
make local-pipeline
make mlflow-local
```

### 4. 🧩 Multi-counter orchestration

Airflow orchestrates the multi-counter ML pipeline. The Docker Compose service
model is the source of truth for Airflow service wiring and runtime identities.

- `airflow-init` imports Airflow variables and connections from
  `./docker/dev/airflow/config/`.
- Runtime Airflow containers mount
  `/opt/airflow/config/bike_dag_config.json` read-only from
  `./docker/dev/airflow/config/bike_dag_config.json`.
- Runtime secrets and IDs are provided through `.env.template`, including
  `AIRFLOW_FERNET_KEY`, `AIRFLOW_API_AUTH_JWT_SECRET`, and
  `AIRFLOW_POSTGRES_PASSWORD`.

DAG roles:

- `bike_traffic_init`: one-shot historical bootstrap per counter.
- `bike_traffic_daily`: rolling increment assuming init has been done.
- `bike_traffic_orchestrator`: triggers `init` then `daily` for each configured
  counter.

### 5. 🧩 Monitoring and alerting

The local monitoring stack uses Prometheus, Grafana, Pushgateway, cAdvisor,
Alertmanager, and MailHog.

- Grafana dashboards use Prometheus metrics to monitor containers, restarts,
  memory, CPU, and ML/API business metrics.
- Pushgateway receives batch metrics from ML pipeline jobs when enabled.
- Alertmanager can route local alerts to MailHog for development validation.

The Pushgateway behavior is controlled by the `DISABLE_METRICS_PUSH` variable in
`.env`:

```bash
# Inhibit push metrics to Pushgateway: 1 = disabled, other values = enabled.
DISABLE_METRICS_PUSH=1
```

## 🤝 Team collaboration

### 1. 📖 External Documentation

- [Data exploration report](https://docs.google.com/spreadsheets/d/1tlDfN-8h9XTJAoKY0zAzmgrJqX90ZAeer48mFxZ_IQg/edit?usp=drive_link)
- [Data processing and modeling report](https://docs.google.com/document/d/1vpRAWaIRX5tjIalEjGLTIjNqwEh1z1kXRZjJA9cgeWo/edit?usp=drive_link)

### 2. 🗺️ GitHub Dashboards

- [Roadmap](https://github.com/users/zheddhe/projects/6/views/2)
- [Current Iteration](https://github.com/users/zheddhe/projects/6/views/3)

### 3. 🔀 Branch Workflow

Based on [jbenet/simple-git-branching-model.md](https://gist.github.com/jbenet/ee6c9ac48068889b0912) and illustrated below:

- Create one branch per story/bugfix and merge via pull requests.
- Tag stable versions ideally after each successful story/bugfix merge.

[![Collaborative branch workflow](references/Branch_Workflow.drawio.png)](https://drive.google.com/file/d/1ctszHKpKDMjhGkC_sdQ3RD8RGAonb967/view?usp=drive_link)

### 4. 🧪 Testing and Continuous Integration

Tests are executed using `pytest`, including:

- ✅ Unit tests for each service separately (`tests/unitary/`)  
- ✅ Cross-service integration tests (`tests/integration/`)

Continuous Integration workflows are handled with GitHub Actions:

- `ci_main.yml`: runs on every push or pull request to the `main` branch  
- `ci_branch.yml`: runs on every push to any other branch

The CI environment uses uv dependency groups directly and runs Ruff before pytest.

### 5. 👥 Contributors

- Rémy Canal – [@remy.canal](mailto:remy.canal@live.fr)  
- Elias Djouadi – [@elias.djouadi](mailto:elias.djouadi@gmail.com)
- Koladé Houessou – [@kolade.houessou](mailto:koladehouessou@gmail.com)
- Sofia Bouizzoul - [@sofia.bouizzoul](mailto:sofia.bouizzoul@gmail.com)
