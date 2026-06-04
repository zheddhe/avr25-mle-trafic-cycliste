#!/usr/bin/env bash
set -euo pipefail

runtime_uid="${AIRFLOW_UID:-50000}"
docker_gid="${AIRFLOW_GID:-0}"

echo "[Custom Entrypoint] UID=${runtime_uid}, GID=${docker_gid}"

if ! id -u airflow >/dev/null 2>&1; then
  useradd --uid "${runtime_uid}" --gid 0 --home-dir /home/airflow airflow
elif [ "$(id -u airflow)" != "${runtime_uid}" ]; then
  usermod --uid "${runtime_uid}" airflow
fi

docker_group="$(getent group "${docker_gid}" | cut -d: -f1 || true)"
if [ -z "${docker_group}" ]; then
  docker_group="docker"
  groupadd --gid "${docker_gid}" "${docker_group}"
fi

usermod --append --groups "${docker_group}" airflow

if [ -d /home/airflow ]; then
  chown -R "${runtime_uid}:0" /home/airflow
fi

echo "[Custom Entrypoint] User groups for airflow:"
id airflow || true

echo "[Custom Entrypoint] Socket permissions:"
ls -l /var/run/docker.sock || true

if command -v runuser >/dev/null 2>&1; then
  exec runuser --user airflow -- /entrypoint "$@"
fi

if command -v su >/dev/null 2>&1; then
  exec su --shell /bin/bash airflow -c 'exec "$@"' -- /entrypoint "$@"
fi

echo "[Custom Entrypoint] Neither runuser nor su is available" >&2
exit 127
