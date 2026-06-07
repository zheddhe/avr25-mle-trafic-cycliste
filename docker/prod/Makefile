SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: prod-help prod-prepare-runtime prod-compose-config prod-build
.PHONY: prod-start prod-stop prod-down prod-ps prod-logs prod-clean

DOCKER_COMPOSE ?= docker compose
ENV_FILE ?= .env
PROD_COMPOSE_FILE ?= docker/prod/docker-compose.yaml
PROD_PROFILE ?= ptf
PROD_PROJECT_NAME ?= bike-traffic-prod
SERVICE ?= api-dev

prod_log_target = printf '==> [%s] \033[33m%s\033[0m\n' "$$(date --iso-8601=seconds)" "$(1)"
prod_check_env = if [ ! -f "$(ENV_FILE)" ]; then echo "Error: $(ENV_FILE) is missing. Run: make env" >&2; exit 1; fi

prod_compose = $(DOCKER_COMPOSE) \
	--env-file $(ENV_FILE) \
	-f $(PROD_COMPOSE_FILE) \
	-p $(PROD_PROJECT_NAME)

prod-help: ## Display local production-like runtime targets
	@awk ' \
		BEGIN {FS=":.*##"; printf "\nLocal production-like targets:\n\n"} \
		/^[a-zA-Z0-9_.-]+:.*##/ { \
			printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2 \
		} \
	' $(MAKEFILE_LIST)

prod-prepare-runtime: ## Create ignored local production-like runtime folders
	@$(call prod_log_target,prod-prepare-runtime)
	mkdir -p \
		docker/prod/runtime/artifacts/manifests \
		docker/prod/runtime/data/raw \
		docker/prod/runtime/data/interim \
		docker/prod/runtime/data/processed \
		docker/prod/runtime/data/final \
		docker/prod/runtime/models \
		docker/prod/runtime/logs/api \
		docker/prod/runtime/logs/airflow \
		docker/prod/runtime/logs/ml

prod-compose-config: prod-prepare-runtime ## Validate production-like Compose config
	@$(call prod_log_target,prod-compose-config)
	@$(prod_check_env)
	$(prod_compose) --profile all config >/dev/null

prod-build: prod-prepare-runtime ## Build local production-like Docker images
	@$(call prod_log_target,prod-build)
	@$(prod_check_env)
	$(prod_compose) --profile all build

prod-start: prod-compose-config ## Start local production-like platform services
	@$(call prod_log_target,prod-start PROD_PROFILE=$(PROD_PROFILE))
	$(prod_compose) --profile $(PROD_PROFILE) up -d

prod-stop: ## Stop local production-like services for the selected profile
	@$(call prod_log_target,prod-stop PROD_PROFILE=$(PROD_PROFILE))
	$(prod_compose) --profile $(PROD_PROFILE) stop

prod-down: ## Remove local production-like containers and networks
	@$(call prod_log_target,prod-down)
	$(prod_compose) --profile all down --remove-orphans

prod-ps: ## Show local production-like service status
	@$(call prod_log_target,prod-ps)
	$(prod_compose) --profile all ps

prod-logs: ## Follow logs for SERVICE in the local production-like runtime
	@$(call prod_log_target,prod-logs SERVICE=$(SERVICE))
	$(prod_compose) logs -f -t $(SERVICE)

prod-clean: ## Remove production-like containers, networks, and volumes
	@$(call prod_log_target,prod-clean)
	@$(prod_check_env)
	$(prod_compose) --profile all down -v --remove-orphans
