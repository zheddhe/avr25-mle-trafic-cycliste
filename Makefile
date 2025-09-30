SHELL := /bin/bash
.ONESHELL:
# flags de gestion du comportement sur erreur (sortie immediate et sur premiere erreur en pipe)
.SHELLFLAGS := -eu -o pipefail -c
# cible make par defaut : help
.DEFAULT_GOAL := help

bootstrap: ## Initialise les dependances bootstrap nécessaires
	@echo "==> Install python3/pip/pipx"
	sudo apt update
	sudo apt install --fix-missing
	sudo apt install -y python3 python3-pip pipx
	pipx ensurepath
	@echo "==> Install nox/uv"
	pipx install nox uv

repo_setup: ## Configure le repo DVC S3 (dagshub) et les credentials github
	@echo "==> Set up GIT user name and email using ${GIT_USER:-change_me} and ${GIT_EMAIL:-change_me@mail.com}"
	git config --global user.name ${GIT_USER:-change_me}
	git config --global user.email ${GIT_EMAIL:-change_me@mail.com}
	@echo "==> Set up local DVC secrets using ${DAGSHUB_KEY} and ${DAGSHUB_KEY}"
	dvc remote modify origin --local access_key_id ${DAGSHUB_KEY}
	dvc remote modify origin --local secret_access_key ${DAGSHUB_KEY}

rebuild_full: ## Recrée les images docker et relance les services complètement
	docker compose --profile all build
	docker compose --profile all down
	docker compose --profile all up -d

PROFILE ?= all

start: ## Démarre les services docker du profil choisi (PROFILE=all/mlflow/airflow/monitoring/api)
	@echo "==> Starting docker services with profile [$(PROFILE)]"
	docker compose --profile $(PROFILE) start

stop: ## Stoppe les services docker du profil choisi (PROFILE=all/mlflow/airflow/monitoring/api)
	@echo "==> Stopping docker services with profile [$(PROFILE)]"
	docker compose --profile $(PROFILE) stop

sim_api_reboot: ## simule un arrêt relance de l'API 10 fois a intervalle de 5s
	@echo "==> Simulating API restart failure loop (10 restarts, 5s interval)"
	for i in {1..10}; do \
		echo "[restart $$i] stopping api-dev..."; \
		docker compose stop api-dev; \
		sleep 2; \
		echo "[restart $$i] starting api-dev..."; \
		docker compose start api-dev; \
		echo "[restart $$i] done, waiting 5s..."; \
		sleep 5; \
	done
	@echo "==> Simulation finished"

sim_api_down: ## simule un arrêt temporaire de l'API pendant 2 minutes
	@echo "==> Simulating API down for 2 min"
	docker compose stop api-dev
	sleep 120
	docker compose start api-dev

clean_full: ## Nettoie les artefacts (images/volumes/networks)
	docker compose --profile all down -v --rmi all && docker system prune -f

help: ## Affiche cette aide
	@awk 'BEGIN{FS=":.*##"; printf "\nTargets disponibles:\n\n"} /^[a-zA-Z0-9_.-]+:.*##/{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2} /^.DEFAULT_GOAL/{print ""} ' $(MAKEFILE_LIST)
