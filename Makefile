SHELL := /bin/bash
.ONESHELL:
# Error handling flags: immediate exit and exit on first pipe error.
.SHELLFLAGS := -eu -o pipefail -c
# Default make target.
.DEFAULT_GOAL := help

.PHONY: help bootstrap env setup repo_setup
.PHONY: sync lock-check lint test ci compose-config
.PHONY: build rebuild_full ops start stop logs
.PHONY: sim_api_loop sim_api_down sim_api_req
.PHONY: clean clean_env clean_full

UV ?= uv
DOCKER_COMPOSE ?= docker compose
ENV_FILE ?= .env
PROFILE ?= ptf
SERVICE ?= api-dev
URL ?= http://localhost:10000
N ?= 100
P_OK ?= 0.80
API_USER ?= user1
API_PASS ?= user1
LIMIT ?= 10
OFFSET ?= 0
COUNTER_IDS ?= Sebastopol_N-S_airflow_day0,Sebastopol_S-N_airflow_day0

log_test = printf '==> [%s] \033[33m%s\033[0m\n' "$$(date --iso-8601=seconds)" "$(1)"
load_env = if [ -f "$(ENV_FILE)" ]; then set -a; source "$(ENV_FILE)"; set +a; fi

help: ## Display this help
	@awk ' \
		BEGIN {FS=":.*##"; printf "\nAvailable targets:\n\n"} \
		/^[a-zA-Z0-9_.-]+:.*##/ { \
			printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2 \
		} \
		/^.DEFAULT_GOAL/ {print ""} \
	' $(MAKEFILE_LIST)

bootstrap: ## Install bootstrap dependencies
	@echo "==> Install python3, pip, and pipx"
	sudo apt update
	sudo apt install --fix-missing
	sudo apt install -y python3 python3-pip pipx
	pipx ensurepath
	@echo "==> Install uv"
	pipx install uv

env: ## Create a local .env file from .env.template if missing
	@if [ -f "$(ENV_FILE)" ]; then \
		echo "==> $(ENV_FILE) already exists. Nothing to do."; \
	else \
		cp .env.template "$(ENV_FILE)"; \
		echo "==> Created $(ENV_FILE) from .env.template"; \
		echo "==> Replace [replace_me] placeholders before running secret-based targets."; \
	fi

setup: env sync compose-config ## Prepare the local project after cloning

repo_setup: env ## Configure Git identity and local DVC S3 credentials
	@$(load_env)
	@if [ -z "$${GIT_USER:-}" ] || [ "$${GIT_USER}" = "[replace_me]" ]; then \
		echo "Error: GIT_USER is not set in $(ENV_FILE)."; \
		exit 1; \
	fi
	@if [ -z "$${GIT_EMAIL:-}" ] || [ "$${GIT_EMAIL}" = "[replace_me]" ]; then \
		echo "Error: GIT_EMAIL is not set in $(ENV_FILE)."; \
		exit 1; \
	fi
	@if [ -z "$${DAGSHUB_KEY:-}" ] || [ "$${DAGSHUB_KEY}" = "[replace_me]" ]; then \
		echo "Error: DAGSHUB_KEY is not set in $(ENV_FILE)."; \
		exit 1; \
	fi
	@echo "==> Set up Git user name and email"
	git config --global user.name "$${GIT_USER}"
	git config --global user.email "$${GIT_EMAIL}"
	@echo "==> Set up local DVC credentials"
	$(UV) run --locked --group dev dvc remote modify origin --local \
		access_key_id "$${DAGSHUB_KEY}"
	$(UV) run --locked --group dev dvc remote modify origin --local \
		secret_access_key "$${DAGSHUB_KEY}"

sync: ## Sync the local uv environment from uv.lock
	@$(call log_test,sync)
	$(UV) sync --locked --group test --group dev

lock-check: ## Check that uv.lock is consistent with pyproject.toml
	@$(call log_test,lock-check)
	$(UV) lock --check

lint: ## Run Ruff checks
	@$(call log_test,lint)
	$(UV) run --locked --group test ruff check .

test: ## Run unit tests while excluding integration tests
	@$(call log_test,test)
	PYTEST_ADDOPTS='-m "not integration"' \
		$(UV) run --locked --group test pytest

ci: lock-check lint test ## Run local CI checks

compose-config: env ## Validate the Docker Compose configuration
	@$(call log_test,compose-config)
	$(DOCKER_COMPOSE) --profile all config >/dev/null

build: env ## Build Docker images for all profiles
	$(DOCKER_COMPOSE) --profile all build

rebuild_full: env ## Rebuild Docker images and restart platform services
	$(DOCKER_COMPOSE) --profile all build
	$(DOCKER_COMPOSE) --profile all down
	$(DOCKER_COMPOSE) --profile ptf up -d

ops: env compose-config start ## Validate and start platform services

start: env ## Start Docker services for the selected profile
	@echo "==> Starting Docker services with profile [$(PROFILE)]"
	$(DOCKER_COMPOSE) --profile $(PROFILE) up -d

stop: ## Stop Docker services for the selected profile
	@echo "==> Stopping Docker services with profile [$(PROFILE)]"
	$(DOCKER_COMPOSE) --profile $(PROFILE) stop

logs: ## Show Docker Compose logs for SERVICE
	$(DOCKER_COMPOSE) logs -f -t $(SERVICE)

sim_api_loop: ## Simulate 10 API stop/start cycles with a 5-second interval
	@echo "==> Simulating API restart failure loop"
	for i in {1..10}; do \
		echo "[restart $$i] stopping api-dev..."; \
		$(DOCKER_COMPOSE) stop api-dev; \
		sleep 2; \
		echo "[restart $$i] starting api-dev..."; \
		$(DOCKER_COMPOSE) up -d api-dev; \
		echo "[restart $$i] done, waiting 5s..."; \
		sleep 5; \
	done
	@echo "==> Simulation finished"

sim_api_down: ## Simulate a temporary API outage for 2 minutes
	@echo "==> Simulating API down for 2 minutes"
	$(DOCKER_COMPOSE) stop api-dev
	sleep 120
	$(DOCKER_COMPOSE) up -d api-dev

sim_api_req: ## Simulate traffic on /predictions/{counter}
	@echo "==> Simulating /predictions/* traffic on $(URL)"
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
	rm -rf .nox .pytest_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean_env: ## Remove the uv-managed virtual environment
	rm -rf .venv

clean_full: ## Remove Docker artifacts, including images, volumes, and networks
	$(DOCKER_COMPOSE) --profile all down -v --rmi all
	docker system prune -f
	docker volume prune -f
