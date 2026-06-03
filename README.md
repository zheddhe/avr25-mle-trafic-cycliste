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
│   ├── raw             <- Original, immutable data dumps (e.g., external sources)
│   ├── interim         <- Intermediate data derived from raw (goal-specific)
│   ├── processed       <- Processed data (e.g., feature-enriched)
│   └── final           <- Final stage data (e.g., train/test and predictions)
├── logs                <- Logs shared with host (read/write)
│   ├── ml              <- ML pipeline logs
│   ├── api             <- Data API logs
│   ├── scheduler       <- Airflow scheduler logs
│   └── dag[...]        <- Unit DAG run logs
├── models              <- Model artifacts shared with host (read/write)
│   └── ...
├── references          <- Data dictionaries, manuals, other explanatory material
│   └── ...
├── src/                <- All source code used in this project
│   ├── airflow/        <- Airflow orchestration management
│   │   ├── dags        <- Orchestrator DAGs code shared with host
│   │   │   ├── bike_traffic_pipeline_dag.py
│   │   │   └── bike_traffic_orchestrator_dag.py
│   │   │   ├── common
│   │   │   │   └── utils.py
│   │   ├── config      <- Orchestrator config shared with host (read-only)
│   │   │   └── bike_dag_config.json
│   ├── api/            <- FastAPI service (prediction readout)
│   │   └── main.py
│   └── ml/             <- Machine learning pipeline
│       ├── ingest      <- Scripts to ingest initial raw data or daily data
│       │   ├── data_utils.py
│       │   └── import_raw_data.py
│       ├── features    <- Scripts to turn raw data into modeling-ready data
│       │   ├── features_utils.py
│       │   └── build_features.py
│       └── models      <- Train models and compute batch predictions
│           ├── models_utils.py
│           └── train_and_predict.py
├── docker/             <- Container architecture
│   ├── dev/            <- Development setup
│   │   ├── grafana/    <- Config for Grafana dashboards and provisioning
│   │   ├── prometheus/ <- Config for Prometheus targets and general configuration
│   │   ├── airflow/    <- Config and DAGs for Airflow
│   │   ├── mlflow/     <- Custom Docker image for MLflow service
│   │   ├── api/        <- Custom Docker image for API service
│   │   └── ml/         <- Custom Docker images for ML pipeline services
│   └── prod/           <- Production setup
│       └── ...
└── tests/
    ├── unitary/        <- Unit tests
    │   └── ...
    └── integration/    <- Integration tests
        └── ...
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

All commands in this README are provided from a Linux operating system point of view.

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

Once `.env` is populated, configure Git identity and DVC credentials with:

```bash
make repo_setup
```

## 🚀 DevOps setup

> This section covers the project setup as a monolithic architecture from a DevOps point of view.

This repository is not packaged as a Python distribution. The local Python environment
is managed by uv and is used for developer tooling, static analysis, tests, Pylance
import resolution, and local execution of the application code.

### uv dependency groups

| Group | Purpose |
|-------|---------|
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

For a first Docker installation on Ubuntu, install Docker Engine and add your
user to the Docker group. This remains a host-level prerequisite and is not
wrapped in the project Makefile.

```bash
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release -y
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
sudo usermod -aG docker $USER
newgrp docker
```

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

For one-off ML pipeline containers, run the existing Compose services directly:

```bash
docker compose --profile ml up ml-ingest-dev
docker compose --profile ml up ml-features-dev
docker compose --profile ml up ml-models-dev
```

The API is exposed at:

```text
http://localhost:10000/docs
```

### 3. 📈 Experience tracker

We use **MLflow** to record **metrics**, **params**, and training/prediction
**artifacts** (scikit-learn pipeline, autoregressive transformer, train/test
splits, predictions, metrics, and hyperparameters).

The MLflow-related environment variables are centralized in `.env.template` and
should be customized in your local `.env` file.

#### Local MLflow service

The default `.env.template` values target the local Docker Compose MLflow and
MinIO services from inside the Compose network:

```bash
MLFLOW_TRACKING_URI="http://mlflow-server:5000"
MLFLOW_S3_ENDPOINT_URL="http://mlflow-minio:9000"
AWS_ACCESS_KEY_ID="minio"
AWS_SECRET_ACCESS_KEY="[replace_me]"
```

For host-side commands outside Docker Compose, use the exposed local endpoints in
your shell or local `.env` when needed:

```bash
MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:9000"
AWS_ACCESS_KEY_ID="minio"
AWS_SECRET_ACCESS_KEY="[replace_me]"
```

#### DagsHub remote MLflow service

To use the DagsHub remote service instead of the local MLflow container, override
these values in your local `.env` or shell session:

```bash
MLFLOW_TRACKING_URI="https://dagshub.com/zheddhe/avr25-mle-trafic-cycliste.mlflow"
MLFLOW_TRACKING_USERNAME="[replace_me]"
MLFLOW_TRACKING_PASSWORD="[replace_me]"
```

### 4. 🧩 Multi-counter orchestration

- The environment configuration is mounted read-only in the Airflow Init container into `/opt/airflow/config/` (repo source: `./docker/dev/airflow/`). It configures especially:
  - The host repository root, to adjust to your production or development environment.
  - The MLflow server information. By default, this uses the platform technical stack, but it can point to a cloud-hosted service.
  - The images to use for the various containers.
  - The API connection as an admin to refresh predictions.

- The business configuration is mounted read-only in the Airflow containers (Scheduler / WebServer / Worker) into
  `/opt/airflow/config/bike_dag_config.json` (repo source: `./src/airflow/config/`). It configures especially:
  - The list of managed counters extracted from the original dataset.
  - The anchor date used to simulate production startup.
  - The daily increment, which defines which portion of the original dataset is considered to shift the data by one production day.

- `bike_traffic_init`: one-shot historical bootstrap per counter.
  - It short-circuits if the Airflow Variable `bike_init_done__<counter>` equals `"1"`.
  - On success, it sets that variable to `"1"`.

- `bike_traffic_daily`: rolling increment assuming init has been done.
  - Triggered by the parent only. Its `schedule` is `None`.

- `bike_traffic_orchestrator`:
  - Every day, for each configured counter, trigger `init` then `daily`.
  - The `init` run is cheap if already done because it is short-circuited.

### 5. 🧩 Monitoring and alerting

The project has defined:

- Grafana dashboards relying on Prometheus collected metrics to:
  - Monitor the system itself: active containers, restarts, memory usage, and CPU usage.
  - Monitor business metrics for the ML pipeline.
- Alerts that can trigger email notifications when detecting:
  - API service down for a configurable period of time.
  - API service instability, such as restart loops.

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
- Sofia Bouizzoul - [@sofia.bouizzoul](mailto:sofiabouizzoul98@gmail.com)
