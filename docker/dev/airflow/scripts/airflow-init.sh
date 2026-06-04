#!/usr/bin/env bash
set -euo pipefail

: "${_AIRFLOW_WWW_USER_USERNAME:?Missing Airflow username}"
: "${_AIRFLOW_WWW_USER_PASSWORD:?Missing Airflow password}"

echo "Running Airflow database migration"
airflow db migrate

echo "Creating Airflow admin user"
airflow users create \
  --username "${_AIRFLOW_WWW_USER_USERNAME}" \
  --password "${_AIRFLOW_WWW_USER_PASSWORD}" \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com

echo "Importing Airflow variables"
airflow variables import /opt/airflow/config/variables.json

echo "Refreshing Airflow api_dev connection"
airflow connections delete api_dev || true
airflow connections import /opt/airflow/config/connections.json

echo "Creating Airflow sequential_counters pool"
airflow pools set \
  sequential_counters \
  2 \
  "Orchestrator: trigger child DAGs sequentially"

echo "Airflow initialization completed"
