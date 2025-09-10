# 🚲 Cyclist Traffic MLOPS Project

[![CI Main](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_main.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)
[![CI Branch](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_branch.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)

> A machine learning pipeline to provide bike traffic prediction in Paris.  
> Developed as part of the April 2025 Machine Learning Engineering (MLE) full training program.

## 🧭 Overview

This project implements a full machine learning and MLOps pipeline in three main stages:

### 1. 📐 Data Product Management

- Define business goals
- Scope the data lifecycle

### 2. 📊 Data Science

- Data collection and preprocessing
- Model development and evaluation
- Time series prediction

### 3. ⚙️ MLOps

- Reproducibility and continuous testing
- Containerization with micro services
- Security awareness
- Monitoring and orchestration
- Scalability

The MLOps architecture we've designed focus on interactions between components in order to achieve our main business case where an external user
access daily refreshed predictions of the biking trafic.

[![MLOps Architecture v2](references/Architecture_MLOps_v2.drawio.png)](https://drive.google.com/file/d/12olpeXpeOF2-UgBSf1h_LhjRVfG8t3KB/view?usp=drive_link)

## 🧭 Project organization

### 1. 📖 External Documentation

- [Data exploration report](https://docs.google.com/spreadsheets/d/1tlDfN-8h9XTJAoKY0zAzmgrJqX90ZAeer48mFxZ_IQg/edit?usp=drive_link)
- [Data processing and modelization report](https://docs.google.com/document/d/1vpRAWaIRX5tjIalEjGLTIjNqwEh1z1kXRZjJA9cgeWo/edit?usp=drive_link)

### 2. 🗺️ GitHub Dashboards

- [Roadmap](https://github.com/users/zheddhe/projects/6/views/2)
- [Current Iteration](https://github.com/users/zheddhe/projects/6/views/3)

### 3. 👥 Branch Workflow

Based on [jbenet/simple-git-branching-model.md](https://gist.github.com/jbenet/ee6c9ac48068889b0912) and illustrated below

- Create branch per story/bugfix and merge them with pull requests afterward
- Tag stable versions ideally after each story/bugfix successfull merge

[![Collaborative branch workflow](references/Branch_Workflow.drawio.png)](https://drive.google.com/file/d/1ctszHKpKDMjhGkC_sdQ3RD8RGAonb967/view?usp=drive_link)

### 4. 📊 MLflow

This project keep a registry of **metrics**, **params** and training and prediction **artefacts**
(sklearn pipeline, auto-regressive transformer, splits train test and prédictions, metrics and hyperparams)
in **MLflow**.

### 5. 🧱 Project Structure

``` text
avr25-mle-trafic-cycliste/
├── LICENSE             <- MIT license
├── README.md           <- This top-level README for developers using this project
├── flake8              <- Linter configuration rules
├── pyproject.toml      <- Python dev project configuration
├── uv.lock             <- UV frozen configuration of the dev env
├── noxfile.py          <- NOX dev session (build/clean)
├── data                <- Data storage
│   ├── raw             <- The original, immutable data dump (e.g. from external sources)
│   ├── interim         <- Intermediate data extracted from raw (e.g. specialized for a goal)
│   ├── processed       <- Processed data that has been transformed (e.g. enriched with feats)
│   └── final           <- Final stage data (e.g. train/test and predictions)
├── logs                <- Logs from training and predicting
│   └──...
├── models              <- Trained and serialized models including their best params and transformers
│   └──...
├── references          <- Data dictionaries, manuals, and all other explanatory materials
│   └──...
├── src/                <- All Source code used in this project
│   ├── api/            <- Service FastAPI (lecture des prédictions)
│   │   └── main.py
│   └── ml/             <- machine learning pipeline
│       ├── data        <- Scripts to collect intial raw data or generate new daily one
│       │   ├── data_utils.py
│       │   └── import_raw_data.py
│       ├── features    <- Scripts to turn raw data into modeling ready data
│       │   ├── features_utils.py
│       │   └── build_features.py
│       └── models      <- Scripts to train models and calculate predictions in batch
│           ├── models_utils.py
│           └── train_and_predict.py
├── docker/             <- container architecture
│   ├── dev/            <- dev architecture
│   │   ├── api/
│   │   │   ├── requirements.txt
│   │   │   └── Dockerfile
│   │   └── ml/
│   │   │   ├── requirements.txt
│   │   │   └── Dockerfile
│   └── prod/           <- production architecture
│       └──...       
└── tests/              <- Unit tests (pytest for src source code)
```

## ⚙️ Installation

### 🔧 Initial Setup (One-time bootstrap)

The build environment initialization requires python, pipx, NOX, UV as a Bootstrap.

```bash
# check python is here (if not install it manually depending on your OS)
python --version 
# install pipx and publish it into the PATH  
python -m pip install --upgrade pip
python -m pip install --user pipx
pipx ensurepath
# install NOX (session manager like a MAKE multi OS) and UV (fast virtual env back end)
pipx install nox uv
```

### 🔧 Repository cloning and DVC setup (One-time init)

Please refer to the Dagshub remote setup actions depending on your preference to collect:

- Git cloning actions (for example)

```bash
git clone https://github.com/zheddhe/avr25-mle-trafic-cycliste.git
```

- DVC setup actions (for example)

```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/zheddhe/avr25-mle-trafic-cycliste.s3

dvc remote modify origin --local access_key_id [...]
dvc remote modify origin --local secret_access_key [...]
```

## 🚀 Day-to-day Usage

```bash
# Rebuild a complete virtual dev env (and trigger flake8 and pytest)
nox -s build

# Activate the virtual env in command line (based on your OS)
# Windows cmd
.nox\build\Scripts\activate.bat 
# Mac/Linux shell
source .nox/build/bin/activate

# [Optional] Clean all project generated file and all virtual envs (build included)
nox -s cleanall

# [Dev without container only] execute the dvc pipeline
dvc repro

# [Dev without container only] launch the data API (find a free port on your system)
uvicorn src.api.main:app --reload --port 10000
# the API will be available at http://localhost:10000/docs

# Configure Dagshub MLflow serveur through environment variable (based on your OS)
# Windows cmd
set MLFLOW_TRACKING_URI=https://dagshub.com/zheddhe/avr25-mle-trafic-cycliste.mlflow 
set MLFLOW_TRACKING_USERNAME=<DagsHub ACCOUNT>
set MLFLOW_TRACKING_PASSWORD=<DagHhub TOKEN (preferrably over a personnal password...)>
# Mac/Linux shell
export MLFLOW_TRACKING_URI=https://dagshub.com/zheddhe/avr25-mle-trafic-cycliste.mlflow
export MLFLOW_TRACKING_USERNAME=<DagsHub ACCOUNT>
export MLFLOW_TRACKING_PASSWORD=<DagsHub TOKEN (preferrably over a personnal password...)>

# Configure Local MLflow dockerized server through environment variables (based on your OS)
# Windows cmd
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
set MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
# Mac/Linux shell
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000

```

In fully dockerized MLOps local environment, we'll switch an .env.local file configuration with the following content:

```text
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
```

Then the docker composition will assemble all dockers with the .env.local context:

```bash
# init and launch all the dockers containers
docker compose --env-file .env.local up -d --force-recreate
```

## 🧪 Testing and Continuous Integration

Tests are executed using `pytest`, including:

- ✅ Unit tests for each modules (in `tests/`)  

CI workflows are handled by GitHub Actions:

- `ci_main.yml`: runs on every push or pull request to the `main` branch  
- `ci_branch.yml`: runs on every push to any other branch

## 👥 Contributors

- Rémy Canal – [@remy.canal](mailto:remy.canal@live.fr)  
- Elias Djouadi – [@elias.djouadi](mailto:elias.djouadi@gmail.com)
- Koladé Houessou – [@kolade.houessou](mailto:koladehouessou@gmail.com)
