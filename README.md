# 🚲 Cyclist Traffic MLOPS Project

[![CI Main](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_main.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)
[![CI Branch](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions/workflows/ci_branch.yml/badge.svg)](https://github.com/zheddhe/avr25-mle-trafic-cycliste/actions)

> A machine learning pipeline to provide bike traffic prediction in Paris.  
> Developed as part of the April 2025 Machine Learning Engineering (MLE) full training program.

---

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

---

## 🧱 Project Structure

``` text
avr25-mle-trafic-cycliste/
├── LICENSE                         # MIT license
├── README.md                       # The top-level README for developers using this project.
├── pyproject.toml                  # The environment context for reproducing a dev environment (with UV)
├── flake8                          # Linter configuration rules
├── data                <- Data storage
│   ├── raw                         # The original, immutable data dump (e.g. from external sources)
│   ├── processed                   # Intermediate data that has been transformed (e.g. enriched)
│   └── final                       # data in final stage (e.g. predictions).
├── logs                <- Logs from training and predicting
│   └──...
├── models              <- Trained and serialized models including their best params and transformers
│   └──...
├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering), the creator's initials, 
│   └──...                 and a short `-` delimited description (e.g. `1.0-jqp-initial-data-exploration`).
├── references          <- Data dictionaries, manuals, and all other explanatory materials.
│   └──...
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         <- Generated graphics and figures to be used in reporting
│       └──...
├── src/                <- All Source code used in this project
│   ├── api/            <- Service FastAPI (lecture des prédictions)
│   │   ├── main.py
│   │   ├── routes/
│   │   │   └── predictions.py
│   │   └── schemas/
│   │       └── prediction.py
│   ├── ml/             <- machine learning pipeline
│   │   ├── data        <- Scripts to collect intial raw data or generate new daily one
│   │   │   ├── utils.py
│   │   │   └── import_raw_data.py
│   │   ├── features    <- Scripts to turn raw data into modeling ready data
│   │   │   ├── utils.py
│   │   │   └── build_features.py
│   │   ├── models      <- Scripts to train models and calculate predictions in batch
│   │   │   ├── utils.py
│   │   │   └── train_and_predict.py
│   └── shared/         <- Shared services
│       └── logger.py
├── docker/             <- container architecture
│   ├── dev/            <- dev architecture
│   │   ├── api/
│   │   │   ├── requirements.txt
│   │   │   └── Dockerfile
│   │   └── ml/
│   │   │   ├── requirements.txt
│   │   │   └── Dockerfile
│   ├── prod/           <- production architecture
│   │   └──...       
├── tests/             <- Unit tests (pytest for src source code)
├── LICENSE                 # MIT license
├── pyproject.toml          # Python project configuration
└── noxfile.py              # NOX session configuration
```

---

## ⚙️ Installation

### 🔧 Initial Setup (One-time bootstrap)

```bash
# The build env initialization requires python, pipx, nox, uv as a bootstrap
python --version # check python is here if not install it manually depending on your OS
python -m pip install --upgrade pip
python -m pip install --user pipx
pipx ensurepath # propagate pipx temporary bootstrap virtual env to PATH if not already done
pipx install nox uv # set up NOX (session manager like a MAKE multi OS) and UV (fast virtual env back end)
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
# Rebuild and complete virtual env for standard streamlit application and notebooks with pytorch (+ trigger test/flake8)
nox -s build

# Activate the virtual env in command line based on your OS (and preferrably add it in your IDE as the interpreter)
.nox\build\Scripts\activate.bat # cmd shell windows only
# or
source .nox/build/bin/activate # cmd shell Mac/Linux only

# Optional: cleanall (project generated file and virtual envs)
nox -s cleanall
```

---

## 🧪 Testing and Continuous Integration

Tests are executed using `pytest`, including:

- ✅ Unit tests for each modules (`trafic/`)  

CI workflows are handled by GitHub Actions:

- `ci_main.yml`: runs on every push or pull request to the `main` branch  
- `ci_branch.yml`: runs on every push to any other branch

---

## 👥 Collaborative branch workflow

Based on [jbenet/simple-git-branching-model.md](https://gist.github.com/jbenet/ee6c9ac48068889b0912) and illustrated below

- Create branch per story/bugfix and merge them with pull requests afterward
- Tag stable versions ideally after each story/bugfix successfull merge

![Collaborative branch workflow](references/collaborative_branch_workflow.drawio.png)

---

## 👥 Contributors

- Rémy Canal – [@remy.canal](mailto:remy.canal@live.fr)  
- Elias Djouadi – [@elias.djouadi](mailto:elias.djouadi@gmail.com)
- Koladé Houessou – [@kolade.houessou](mailto:koladehouessou@gmail.com)
