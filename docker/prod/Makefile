SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

.PHONY: help prod-compose-config prod-build prod-ops prod-stop prod-down
.PHONY: prod-ps prod-logs

DOCKER_COMPOSE ?= docker compose
ENV_FILE ?= .env
PROD_COMPOSE_FILE ?= docker/prod/docker-compose.yaml
PROD_PROFILE ?= ptf
SERVICE ?= api-dev

log_target = printf '==> [%s] \033[33m%s\033[0m\n' "$$(date --iso-8601=seconds)" "$(1)"
check_env = if [ ! -f "$(ENV_FILE)" ]; then echo "Error: $(ENV_FILE) is missing. Run: make env" >&2; exit 1; fi

help: ## Display local production-like runtime targets
	@awk ' \
		BEGIN {FS=":.*##"; printf "\nLocal production-like targets:\n\n"} \
		/^[a-zA-Z0-9_.-]+:.*##/ { \
			printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2 \
		} \
	' $(MAKEFILE_LIST)

prod-compose-config: ## Validate the local production-like Compose config
	@$(call log_target,prod-compose-config)
	@$(check_env)
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f $(PROD_COMPOSE_FILE) \
		--profile all \
		config >/dev/null

prod-build: ## Build local production-like Docker images
	@$(call log_target,prod-build)
	@$(check_env)
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f $(PROD_COMPOSE_FILE) \
		--profile all \
		build

prod-ops: prod-compose-config ## Start local production-like platform services
	@$(call log_target,prod-ops PROD_PROFILE=$(PROD_PROFILE))
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f $(PROD_COMPOSE_FILE) \
		--profile $(PROD_PROFILE) \
		up -d

prod-stop: ## Stop local production-like services for the selected profile
	@$(call log_target,prod-stop PROD_PROFILE=$(PROD_PROFILE))
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f $(PROD_COMPOSE_FILE) \
		--profile $(PROD_PROFILE) \
		stop

prod-down: ## Remove local production-like containers and networks
	@$(call log_target,prod-down)
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f $(PROD_COMPOSE_FILE) \
		--profile all \
		down

prod-ps: ## Show local production-like service status
	@$(call log_target,prod-ps)
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f $(PROD_COMPOSE_FILE) \
		--profile all \
		ps

prod-logs: ## Follow logs for SERVICE in the local production-like runtime
	@$(call log_target,prod-logs SERVICE=$(SERVICE))
	$(DOCKER_COMPOSE) \
		--env-file $(ENV_FILE) \
		-f $(PROD_COMPOSE_FILE) \
		logs -f -t $(SERVICE)
