#!/usr/bin/env bash
set -euo pipefail

echo "Running Airflow database migration"
airflow db migrate

echo "Refreshing Airflow api_dev connection"
airflow connections delete api_dev || true
airflow connections import /opt/airflow/config/connections.json

echo "Creating Airflow sequential_counters pool"
airflow pools set \
  sequential_counters \
  1 \
  "Orchestrator: trigger child DAGs sequentially"

echo "Airflow initialization completed"
