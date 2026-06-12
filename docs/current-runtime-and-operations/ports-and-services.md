# Runtime ports and service exposure

This document describes local Docker Compose host port conventions. It is a local
runtime convention only and does not represent a production security model.

Docker services communicate through their internal container ports and runtime
networks. The variables below only control ports published on the host machine.

## Port ranges

`.env.template` separates dev and prod-like host port ranges so both runtimes can
run at the same time.

| Range | Runtime | Responsibility |
| ----- | ------- | -------------- |
| `10000-10499` | Prod-like | Business applications and application APIs. |
| `10500-10999` | Dev | Business applications and application APIs. |
| `12000-12499` | Prod-like | Orchestration and workflow tools. |
| `12500-12999` | Dev | Orchestration and workflow tools. |
| `13000-13499` | Prod-like | Tracking and artifact systems. |
| `13500-13999` | Dev | Tracking and artifact systems. |
| `14000-14499` | Prod-like | Observability and alerting. |
| `14500-14999` | Dev | Observability and alerting. |
| `15000-15999` | Dev | Development-only helpers and debug tools. |

## Runtime scope

The development runtime deliberately exposes more local UIs and debug endpoints
than the local production-like runtime.

| Runtime | Compose file | Exposure model |
| ------- | ------------ | -------------- |
| Development | `docker/dev/docker-compose.yaml` | Full visible local runtime for debugging and demos. |
| Local production-like | `docker/prod/docker-compose.yaml` | Reduced host exposure for operator-facing services only. |

## Development host-exposed services

| Variable | Default | Service | Internal port | Reason |
| -------- | ------- | ------- | ------------- | ------ |
| `API_HOST_PORT_DEV` | `10500` | `api-dev` | `10000` | FastAPI prediction API and OpenAPI docs. |
| `AIRFLOW_HOST_PORT_DEV` | `12580` | `airflow-api-server` | `8080` | Airflow UI and API for local DAG operations. |
| `MLFLOW_HOST_PORT_DEV` | `13501` | `mlflow-server` | `5000` | MLflow tracking UI and host-side MLflow clients. |
| `MINIO_CONSOLE_HOST_PORT_DEV` | `13502` | `mlflow-minio` | `9001` | MinIO browser console for local artifact debugging. |
| `PROMETHEUS_HOST_PORT_DEV` | `14590` | `monitoring-prometheus` | `9090` | Prometheus UI and query debugging. |
| `PUSHGATEWAY_HOST_PORT_DEV` | `14591` | `monitoring-pushgateway` | `9091` | Pushgateway UI and local metrics debugging. |
| `CADVISOR_HOST_PORT_DEV` | `14520` | `monitoring-cadvisor` | `8080` | Container-level runtime metrics debugging. |
| `GRAFANA_HOST_PORT_DEV` | `14530` | `monitoring-grafana` | `3000` | Grafana dashboards. |
| `ALERTMANAGER_HOST_PORT_DEV` | `14593` | `monitoring-alertmanager` | `9093` | Alertmanager UI and alert routing debugging. |
| `MAILHOG_SMTP_HOST_PORT_DEV` | `15025` | `monitoring-mailhog` | `1025` | Local SMTP capture endpoint. |
| `MAILHOG_UI_HOST_PORT_DEV` | `15080` | `monitoring-mailhog` | `8025` | MailHog web UI. |

## Local production-like host-exposed services

| Variable | Default | Service | Internal port | Reason |
| -------- | ------- | ------- | ------------- | ------ |
| `API_HOST_PORT_PROD` | `10000` | `api-prod` | `10000` | FastAPI prediction API and OpenAPI docs. |
| `AIRFLOW_HOST_PORT_PROD` | `12080` | `airflow-api-server` | `8080` | Airflow UI and API for local DAG operations. |
| `MLFLOW_HOST_PORT_PROD` | `13001` | `mlflow-server` | `5000` | MLflow tracking UI and host-side MLflow clients. |
| `GRAFANA_HOST_PORT_PROD` | `14030` | `monitoring-grafana` | `3000` | Grafana dashboards. |

## Local URLs

With the default `.env.template` values, the main local URLs are:

| Service | Development URL | Production-like URL |
| ------- | --------------- | ------------------- |
| FastAPI docs | `http://localhost:10500/docs` | `http://localhost:10000/docs` |
| Airflow UI | `http://localhost:12580` | `http://localhost:12080` |
| MLflow UI | `http://localhost:13501` | `http://localhost:13001` |
| Grafana UI | `http://localhost:14530` | `http://localhost:14030` |
| MinIO console | `http://localhost:13502` | Internal-only. |
| Prometheus UI | `http://localhost:14590` | Internal-only. |
| Pushgateway UI | `http://localhost:14591` | Internal-only. |
| cAdvisor UI | `http://localhost:14520` | Internal-only. |
| Alertmanager UI | `http://localhost:14593` | Internal-only. |
| MailHog UI | `http://localhost:15080` | Internal-only. |

## Internal-only services

The following service families are intentionally not published on host ports:

| Service or family | Runtime | Internal port | Reason |
| ----------------- | ------- | ------------- | ------ |
| `ml-ingest-dev` | Dev | `10081` | Internal FastAPI ML step service for ingestion jobs. |
| `ml-features-dev` | Dev | `10082` | Internal FastAPI ML step service for feature jobs. |
| `ml-models-dev` | Dev | `10083` | Internal FastAPI ML step service for model jobs. |
| `ml-ingest-prod` | Prod-like | `10081` | Internal FastAPI ML step service for production-like ingestion jobs. |
| `ml-features-prod` | Prod-like | `10082` | Internal FastAPI ML step service for production-like feature jobs. |
| `ml-models-prod` | Prod-like | `10083` | Internal FastAPI ML step service for production-like model jobs. |
| `ml-gateway` | Dev and prod-like | `10090` | Internal Nginx gateway between runner API and ML step service replicas. |
| `job-runner-api` | Dev and prod-like | `10080` | Internal typed job submission and status boundary. |
| `mlflow-postgres` | Dev and prod-like | `5432` | MLflow internal metadata database. |
| `mlflow-minio` | Prod-like | `9000`, `9001` | Internal artifact backend and console in prod-like runtime. |
| `mlflow-minio` API | Dev | `9000` | Internal artifact API; only the console is host-exposed in dev. |
| `mlflow-mc-init` | Dev and prod-like | n/a | One-off MinIO initialization helper. |
| `airflow-postgres` | Dev and prod-like | `5432` | Airflow internal metadata database. |
| `airflow-redis` | Dev and prod-like | `6379` | Airflow Celery broker. |
| `airflow-init` | Dev and prod-like | n/a | One-off Airflow initialization helper. |
| `airflow-scheduler` | Dev and prod-like | n/a | Internal Airflow scheduler process. |
| `airflow-dag-processor` | Dev and prod-like | n/a | Internal Airflow DAG parsing process. |
| `airflow-worker` | Dev and prod-like | n/a | Internal Airflow Celery worker process. |
| `airflow-triggerer` | Dev and prod-like | n/a | Internal Airflow async trigger process. |
| Prometheus, Pushgateway, Alertmanager, cAdvisor, MailHog | Prod-like | service-specific | Internal support services; host UI kept dev-only except Grafana. |

Manifest-first API serving is documented in
[`local-prod-runtime.md`](local-prod-runtime.md). The API reads promoted manifests
and does not infer payloads by scanning runtime folders.

## Validation

The expected configuration and image build validation commands are:

```bash
make dev-compose-config
make prod-compose-config
make dev-build
make prod-build
```

Expected exposure checks:

- dev can expose API, Airflow, MLflow, Grafana, MinIO console, Prometheus,
  Pushgateway, cAdvisor, Alertmanager, and MailHog;
- prod-like exposes only API, Airflow, MLflow, and Grafana;
- `job-runner-api`, `ml-gateway`, and ML step services stay internal-only in both
  runtimes;
- prod-like MinIO and observability support services stay internal-only except
  Grafana.
