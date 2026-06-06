SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

.PHONY: help bootstrap docker-install env setup git-setup dvc-setup
.PHONY: repo-setup
.PHONY: env-compose env-local env-dagshub
.PHONY: sync lock-check lint tests ci
.PHONY: dvc-pipeline
.PHONY: local-ingest local-features local-models local-pipeline mlflow-local
.PHONY: sim_api_loop sim_api_down sim_api_req
.PHONY: clean clean_env clean_full

UV ?= uv
DOCKER_COMPOSE ?= docker compose
ENV_FILE ?= .env
PROFILE ?= ptf
DEV_PROFILE ?= $(PROFILE)
PROD_PROFILE ?= ptf
SERVICE ?= api-dev
URL ?= http://localhost:10000
N ?= 100
P_OK ?= 0.80
API_USER ?= user1
API_PASS ?= user1
LIMIT ?= 10
OFFSET ?= 0
COUNTER_IDS ?= Sebastopol_N-S_airflow_day0,Sebastopol_S-N_airflow_day0

DOCKER_KEYRING := /etc/apt/keyrings/docker.gpg
DOCKER_LIST := /etc/apt/sources.list.d/docker.list
DOCKER_GPG_URL := https://download.docker.com/linux/ubuntu/gpg
DOCKER_REPO_URL := https://download.docker.com/linux/ubuntu

MLFLOW_HOST ?= 127.0.0.1
MLFLOW_HOST_PORT ?= 5001
MLFLOW_BACKEND_STORE ?= ./mlruns

log_test = printf '==> [%s] \033[33m%s\033[0m\n' "$$(date --iso-8601=seconds)" "$(1)"
load_env = if [ -f "$(ENV_FILE)" ]; then set -a; source "$(ENV_FILE)"; set +a; fi
check_env = if [ ! -f "$(ENV_FILE)" ]; then echo "Error: $(ENV_FILE) is missing. Run: make env" >&2; exit 1; fi

include docker/dev/Makefile
include docker/prod/Makefile

help: ## Display this help
	@awk ' \
		BEGIN {FS=":.*##"; printf "\nAvailable targets:\n\n"} \
		/^[a-zA-Z0-9_.-]+:.*##/ { \
			printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2 \
		} \
		/^.DEFAULT_GOAL/ {print ""} \
	' $(MAKEFILE_LIST)

bootstrap: ## Install Python bootstrap dependencies
	@$(call log_test,bootstrap)
	sudo apt update
	sudo apt install --fix-missing
	sudo apt install -y python3 python3-pip pipx git
	pipx ensurepath
	pipx install uv

docker-install: ## Install Docker Engine and Compose plugin on Ubuntu
	@$(call log_test,docker-install)
	sudo apt update
	sudo apt install -y ca-certificates curl gnupg lsb-release
	sudo install -m 0755 -d /etc/apt/keyrings
	curl -fsSL $(DOCKER_GPG_URL) | \
		sudo gpg --dearmor --yes -o $(DOCKER_KEYRING)
	echo \
		"deb [arch=$$(dpkg --print-architecture) signed-by=$(DOCKER_KEYRING)] \
		$(DOCKER_REPO_URL) $$(lsb_release -cs) stable" | \
		sudo tee $(DOCKER_LIST) > /dev/null
	sudo apt update
	sudo apt install -y \
		docker-ce \
		docker-ce-cli \
		containerd.io \
		docker-buildx-plugin \
		docker-compose-plugin
	sudo usermod -aG docker "$$USER"
	@echo "==> Docker installed. Open a new shell or run: newgrp docker"

env: ## Create a local .env file from .env.template if missing
	@if [ -f "$(ENV_FILE)" ]; then \
		echo "==> $(ENV_FILE) already exists. Nothing to do."; \
	else \
		cp .env.template "$(ENV_FILE)"; \
		echo "==> Created $(ENV_FILE) from .env.template"; \
		echo "==> Replace [replace_me] placeholders before running secret-based targets."; \
	fi

env-compose: ## Print shell exports for Docker Compose MLflow mode
	@$(check_env)
	@$(load_env); \
	: "$${MLFLOW_TRACKING_URI_COMPOSE:?Missing MLFLOW_TRACKING_URI_COMPOSE}"; \
	: "$${MLFLOW_S3_ENDPOINT_URL_COMPOSE:?Missing MLFLOW_S3_ENDPOINT_URL_COMPOSE}"; \
	: "$${AWS_ACCESS_KEY_ID_COMPOSE:?Missing AWS_ACCESS_KEY_ID_COMPOSE}"; \
	: "$${AWS_SECRET_ACCESS_KEY_COMPOSE:?Missing AWS_SECRET_ACCESS_KEY_COMPOSE}"; \
	: "$${AWS_DEFAULT_REGION_COMPOSE:?Missing AWS_DEFAULT_REGION_COMPOSE}"; \
	printf 'export MLFLOW_TRACKING_URI=%q\n' "$${MLFLOW_TRACKING_URI_COMPOSE}"; \
	printf 'export MLFLOW_S3_ENDPOINT_URL=%q\n' "$${MLFLOW_S3_ENDPOINT_URL_COMPOSE}"; \
	printf 'export AWS_ACCESS_KEY_ID=%q\n' "$${AWS_ACCESS_KEY_ID_COMPOSE}"; \
	printf 'export AWS_SECRET_ACCESS_KEY=%q\n' "$${AWS_SECRET_ACCESS_KEY_COMPOSE}"; \
	printf 'export AWS_DEFAULT_REGION=%q\n' "$${AWS_DEFAULT_REGION_COMPOSE}"; \
	printf 'unset MLFLOW_TRACKING_USERNAME\n'; \
	printf 'unset MLFLOW_TRACKING_PASSWORD\n'

env-local: ## Print shell exports for host-local MLflow backend mode
	@$(check_env)
	@$(load_env); \
	printf 'unset MLFLOW_TRACKING_URI\n'; \
	printf 'unset MLFLOW_S3_ENDPOINT_URL\n'; \
	printf 'unset MLFLOW_TRACKING_USERNAME\n'; \
	printf 'unset MLFLOW_TRACKING_PASSWORD\n'; \
	printf 'unset AWS_ACCESS_KEY_ID\n'; \
	printf 'unset AWS_SECRET_ACCESS_KEY\n'; \
	printf 'unset AWS_DEFAULT_REGION\n'

env-dagshub: ## Print shell exports for DagsHub MLflow mode
	@$(check_env)
	@$(load_env); \
	: "$${MLFLOW_TRACKING_URI_DAGSHUB:?Missing MLFLOW_TRACKING_URI_DAGSHUB}"; \
	: "$${MLFLOW_TRACKING_USERNAME_DAGSHUB:?Missing MLFLOW_TRACKING_USERNAME_DAGSHUB}"; \
	: "$${MLFLOW_TRACKING_PASSWORD_DAGSHUB:?Missing MLFLOW_TRACKING_PASSWORD_DAGSHUB}"; \
	printf 'export MLFLOW_TRACKING_URI=%q\n' "$${MLFLOW_TRACKING_URI_DAGSHUB}"; \
	printf 'export MLFLOW_TRACKING_USERNAME=%q\n' "$${MLFLOW_TRACKING_USERNAME_DAGSHUB}"; \
	printf 'export MLFLOW_TRACKING_PASSWORD=%q\n' "$${MLFLOW_TRACKING_PASSWORD_DAGSHUB}"; \
	printf 'unset MLFLOW_S3_ENDPOINT_URL\n'; \
	printf 'unset AWS_ACCESS_KEY_ID\n'; \
	printf 'unset AWS_SECRET_ACCESS_KEY\n'; \
	printf 'unset AWS_DEFAULT_REGION\n'

setup: env sync dev-compose-config ## Prepare the local project after cloning

git-setup: env ## Configure local Git identity from .env
	@$(call log_test,git-setup)
	@$(load_env)
	@if [ -z "$${GIT_USER:-}" ] || [ "$${GIT_USER}" = "[replace_me]" ]; then \
		echo "Error: GIT_USER is not set in $(ENV_FILE)."; \
		exit 1; \
	fi
	@if [ -z "$${GIT_EMAIL:-}" ] || [ "$${GIT_EMAIL}" = "[replace_me]" ]; then \
		echo "Error: GIT_EMAIL is not set in $(ENV_FILE)."; \
		exit 1; \
	fi
	git config --global user.name "$${GIT_USER}"
	git config --global user.email "$${GIT_EMAIL}"

dvc-setup: env sync ## Configure local DVC credentials in .dvc/config.local
	@$(call log_test,dvc-setup)
	@$(load_env)
	@if [ ! -d ".dvc" ]; then \
		echo "Error: .dvc directory is missing. Run 'dvc init' before configuring remotes."; \
		exit 1; \
	fi
	@if [ -z "$${DAGSHUB_ACCESS_KEY_ID:-}" ] || [ "$${DAGSHUB_ACCESS_KEY_ID}" = "[replace_me]" ]; then \
		echo "Error: DAGSHUB_ACCESS_KEY_ID is not set in $(ENV_FILE)."; \
		exit 1; \
	fi
	@if [ -z "$${DAGSHUB_SECRET_ACCESS_KEY:-}" ] || [ "$${DAGSHUB_SECRET_ACCESS_KEY}" = "[replace_me]" ]; then \
		echo "Error: DAGSHUB_SECRET_ACCESS_KEY is not set in $(ENV_FILE)."; \
		exit 1; \
	fi
	$(UV) run --locked --group dev dvc remote modify origin --local \
		access_key_id "$${DAGSHUB_ACCESS_KEY_ID}"
	$(UV) run --locked --group dev dvc remote modify origin --local \
		secret_access_key "$${DAGSHUB_SECRET_ACCESS_KEY}"

repo-setup: git-setup dvc-setup ## Configure Git identity and local DVC credentials

sync: ## Sync the local uv environment from uv.lock
	@$(call log_test,sync)
	$(UV) sync --locked --group test --group dev

lock-check: ## Check that uv.lock is consistent with pyproject.toml
	@$(call log_test,lock-check)
	$(UV) lock --check

lint: ## Run Ruff checks
	@$(call log_test,lint)
	$(UV) run --locked --group test ruff check .

tests: ## Run integration tests scope
	@$(call log_test,integration-test)
	$(UV) run --locked --group test pytest -m "integration" -v

ci: lock-check lint tests ## Run local CI checks

dvc-pipeline: env ## Run the full DVC pipeline
	@$(call log_test,dvc-pipeline)
	eval "$$($(MAKE) --no-print-directory env-dagshub)"
	$(UV) run --locked --group dev dvc repro

local-ingest: env sync ## Run the ML ingestion step locally without containers
	@$(call log_test,local-ingest)
	@$(load_env)
	eval "$$($(MAKE) --no-print-directory env-local)"
	$(UV) run --locked --group app python -m src.ml.ingest.import_raw_data \
		--raw-path "data/raw/$${RAW_FILE_NAME}" \
		--site "$${SITE}" \
		--orientation "$${ORIENTATION}" \
		--range-start "$${RANGE_START}" \
		--range-end "$${RANGE_END}" \
		--timestamp-col "date_et_heure_de_comptage" \
		--sub-dir "$${SUB_DIR}_local" \
		--interim-name "$${INTERIM_NAME}"

local-features: env sync ## Run the ML feature step locally without containers
	@$(call log_test,local-features)
	@$(load_env)
	eval "$$($(MAKE) --no-print-directory env-local)"
	$(UV) run --locked --group test python -m src.ml.features.build_features \
		--interim-path "data/interim/$${SUB_DIR}_local/$${INTERIM_NAME}" \
		--sub-dir "$${SUB_DIR}_local" \
		--processed-name "$${PROCESSED_NAME}" \
		--timestamp-col "date_et_heure_de_comptage"

local-models: env sync ## Run the ML training step locally without containers
	@$(call log_test,local-models)
	@$(load_env)
	eval "$$($(MAKE) --no-print-directory env-local)"
	$(UV) run --locked --group test python -m src.ml.models.train_and_predict \
		--processed-path "data/processed/$${SUB_DIR}_local/$${PROCESSED_NAME}" \
		--sub-dir "$${SUB_DIR}_local" \
		--target-col "comptage_horaire" \
		--ts-col-utc "date_et_heure_de_comptage_utc" \
		--ts-col-local "date_et_heure_de_comptage_local" \
		--ar "$${AR}" \
		--mm "$${MM}" \
		--roll "$${ROLL}" \
		--test-ratio "$${TEST_RATIO}" \
		--grid-iter "$${GRID_ITER}" \
		--mlflow-uri ""

local-pipeline: local-ingest local-features local-models ## Run the local ML pipeline without containers

mlflow-local: env sync ## Start a host-side MLflow server for local experiments
	@$(call log_test,mlflow-local)
	eval "$$($(MAKE) --no-print-directory env-local)"
	$(UV) run --locked --group app mlflow server \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_HOST_PORT) \
		--backend-store-uri $(MLFLOW_BACKEND_STORE)

sim_api_loop: ## Simulate 10 API stop/start cycles with a 5-second interval
	@$(call log_test,sim_api_loop)
	for i in {1..10}; do \
		echo "[restart $$i] stopping api-dev..."; \
		$(DOCKER_COMPOSE) \
			--env-file $(ENV_FILE) \
			-f docker/dev/docker-compose.yaml \
			-p trafic-cycliste-dev stop api-dev; \
		sleep 2; \
		echo "[restart $$i] starting api-dev..."; \
		$(DOCKER_COMPOSE) \
			--env-file $(ENV_FILE) \
			-f docker/dev/docker-compose.yaml \
			-p trafic-cycliste-dev up -d api-dev; \
		echo "[restart $$i] done, waiting 5s..."; \
		sleep 5; \
	done
	@echo "==> Simulation finished"

sim_api_down: ## Simulate a temporary API outage for 2 minutes
	@$(call log_test,sim_api_down)
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f docker/dev/docker-compose.yaml \
		-p trafic-cycliste-dev stop api-dev
	sleep 120
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f docker/dev/docker-compose.yaml \
		-p trafic-cycliste-dev up -d api-dev

sim_api_req: ## Simulate traffic on /predictions/{counter}
	@$(call log_test,sim_api_req)
	$(UV) run --locked --group test python tests/integration/test_load_api.py \
		--url "$(URL)" \
		--n $(N) \
		--p-ok $(P_OK) \
		--user "$(API_USER)" \
		--password "$(API_PASS)" \
		--limit $(LIMIT) \
		--offset $(OFFSET) \
		$(if $(COUNTER_IDS),--counter-ids "$(COUNTER_IDS)",)

clean: ## Remove local Python caches and test artifacts only
	@$(call log_test,clean)
	rm -rf .nox .pytest_cache .ruff_cache .coverage htmlcov build dist mlruns mlflow.db
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean_env: ## Remove the uv-managed virtual environment
	@$(call log_test,clean_env)
	rm -rf .venv

clean_full: ## Remove development Docker artifacts, including images and volumes
	@$(call log_test,clean_full)
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f docker/dev/docker-compose.yaml \
		-p trafic-cycliste-dev \
		--profile all down -v --rmi all
	docker system prune -f
	docker volume prune -f
