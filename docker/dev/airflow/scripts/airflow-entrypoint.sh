#!/bin/bash
set -e

echo "[Custom Entrypoint] UID=${AIRFLOW_UID}, GID=${AIRFLOW_GID}"

# Créer le groupe docker si absent
if ! getent group docker >/dev/null; then
  groupadd -g "${AIRFLOW_GID}" docker
fi

# Créer le user airflow s’il n’existe pas
if ! id -u airflow >/dev/null 2>&1; then
  useradd -u "${AIRFLOW_UID}" -g "${AIRFLOW_GID}" -m airflow
fi

# Ajouter l’utilisateur au groupe docker
usermod -aG docker airflow

echo "[Custom Entrypoint] User groups for airflow:"
id airflow || true

echo "[Custom Entrypoint] Socket permissions:"
ls -l /var/run/docker.sock || true

# 🟢 Appel de l’entrypoint officiel Airflow sous l’utilisateur airflow
exec gosu airflow /entrypoint "$@"