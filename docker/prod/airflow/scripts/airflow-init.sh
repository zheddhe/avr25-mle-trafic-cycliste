#!/usr/bin/env bash
set -euo pipefail

echo "Running Airflow database migration"
airflow db migrate

echo "Cleaning obsolete production-like Airflow variables"
for variable_name in \
  docker_image_ingest \
  docker_image_features \
  docker_image_models \
  docker_network \
  host_repo_root \
  container_repo_root \
  airflow_repo_root \
  bike_dag_config \
  default_counter_id \
  mlflow_tracking_uri \
  mlflow_s3_endpoint_url \
  aws_access_key_id \
  aws_secret_access_key \
  aws_default_region \
  airflow_uid \
  airflow_gid \
  pushgateway_addr \
  disable_metrics_push \
  tz; do
  airflow variables delete "${variable_name}" || true
done

echo "Refreshing Airflow api_prod connection"
airflow connections delete api_dev || true
airflow connections delete api_prod || true
airflow connections import /opt/airflow/config/connections.json

echo "Creating Airflow sequential_counters pool"
airflow pools set \
  sequential_counters \
  1 \
  "Orchestrator: trigger child DAGs sequentially"

echo "Airflow initialization completed"
