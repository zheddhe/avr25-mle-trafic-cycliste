SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: dev-help dev-compose-config dev-build
.PHONY: dev-start dev-stop dev-down dev-ps dev-logs
.PHONY: dev-mlops-ingest dev-mlops-features dev-mlops-models
.PHONY: dev-mlops-pipeline

DOCKER_COMPOSE ?= docker compose
ENV_FILE ?= .env
DEV_COMPOSE_FILE ?= docker/dev/docker-compose.yaml
DEV_PROFILE ?= ptf
DEV_PROJECT_NAME ?= trafic-cycliste-dev
SERVICE ?= api-dev

dev_log_target = printf '==> [%s] \033[33m%s\033[0m\n' "$$(date --iso-8601=seconds)" "$(1)"
dev_check_env = if [ ! -f "$(ENV_FILE)" ]; then echo "Error: $(ENV_FILE) is missing. Run: make env" >&2; exit 1; fi

dev_compose = $(DOCKER_COMPOSE) \
	--env-file $(ENV_FILE) \
	-f $(DEV_COMPOSE_FILE) \
	-p $(DEV_PROJECT_NAME)

dev-help: ## Display local development runtime targets
	@awk ' \
		BEGIN {FS=":.*##"; printf "\nLocal development targets:\n\n"} \
		/^[a-zA-Z0-9_.-]+:.*##/ { \
			printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2 \
		} \
	' $(MAKEFILE_LIST)

dev-compose-config: ## Validate the local development Compose config
	@$(call dev_log_target,dev-compose-config)
	@$(dev_check_env)
	$(dev_compose) --profile all config >/dev/null

dev-build: ## Build local development Docker images
	@$(call dev_log_target,dev-build)
	@$(dev_check_env)
	$(dev_compose) --profile all build

dev-start: ## Start development services for the selected profile
	@$(call dev_log_target,dev-start DEV_PROFILE=$(DEV_PROFILE))
	@$(dev_check_env)
	$(dev_compose) --profile $(DEV_PROFILE) up -d

dev-stop: ## Stop development services for the selected profile
	@$(call dev_log_target,dev-stop DEV_PROFILE=$(DEV_PROFILE))
	$(dev_compose) --profile $(DEV_PROFILE) stop

dev-down: ## Remove development containers and networks
	@$(call dev_log_target,dev-down)
	$(dev_compose) --profile all down

dev-ps: ## Show development service status
	@$(call dev_log_target,dev-ps)
	$(dev_compose) --profile all ps

dev-logs: ## Follow logs for SERVICE in the development runtime
	@$(call dev_log_target,dev-logs SERVICE=$(SERVICE))
	$(dev_compose) logs -f -t $(SERVICE)

dev-mlops-ingest: ## Run the development ML ingestion container once
	@$(call dev_log_target,dev-mlops-ingest)
	@$(dev_check_env)
	$(dev_compose) --profile ml up ml-ingest-dev

dev-mlops-features: ## Run the development ML feature container once
	@$(call dev_log_target,dev-mlops-features)
	@$(dev_check_env)
	$(dev_compose) --profile ml up ml-features-dev

dev-mlops-models: ## Run the development ML model container once
	@$(call dev_log_target,dev-mlops-models)
	@$(dev_check_env)
	$(dev_compose) --profile ml up ml-models-dev

dev-mlops-pipeline: dev-mlops-ingest dev-mlops-features dev-mlops-models ## Run the full development ML pipeline
