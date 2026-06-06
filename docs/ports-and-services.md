# Runtime ports and service exposure

This document describes local Docker Compose host port conventions. It is a
local runtime convention only and does not represent a production security model.

Docker services communicate through their internal container ports and Docker
networks. The variables below only control ports published on the host machine.

## Port ranges

| Range | Responsibility |
| ----- | -------------- |
| `10000-10999` | Business applications and application APIs |
| `11000-11999` | Local business debug endpoints, if needed later |
| `12000-12999` | Orchestration and workflow tools |
| `13000-13999` | Tracking and artifact systems |
| `14000-14999` | Observability and alerting |
| `15000-15999` | Development-only helpers and debug tools |

## Runtime scope

The development runtime deliberately exposes more local UIs and debug endpoints
than the local production-like runtime.

| Runtime | Compose file | Exposure model |
| ------- | ------------ | -------------- |
| Development | `docker/dev/docker-compose.yaml` | Broad local UI and debug exposure. |
| Compatibility dev entrypoint | `docker-compose.yaml` | Same development exposure from the repository root. |
| Local production-like | `docker/prod/docker-compose.yaml` | Reduced host exposure for operator-facing services only. |

## Development host-exposed services

| Variable | Default | Service | Internal port | Reason |
| -------- | ------- | ------- | ------------- | ------ |
| `API_HOST_PORT` | `10000` | `api-dev` | `10000` | FastAPI prediction API and OpenAPI docs |
| `AIRFLOW_HOST_PORT` | `12080` | `airflow-api-server` | `8080` | Airflow UI and API for local DAG operations |
| `AIRFLOW_FLOWER_HOST_PORT` | `12555` | `airflow-flower` | `5555` | Celery worker monitoring during local development |
| `MINIO_API_HOST_PORT` | `13000` | `mlflow-minio` | `9000` | S3-compatible artifact endpoint for host-side MLflow clients |
| `MLFLOW_HOST_PORT` | `13001` | `mlflow-server` | `5000` | MLflow tracking UI and host-side MLflow clients |
| `MINIO_CONSOLE_HOST_PORT` | `13002` | `mlflow-minio` | `9001` | MinIO browser console for local artifact debugging |
| `PROMETHEUS_HOST_PORT` | `14090` | `monitoring-prometheus` | `9090` | Prometheus UI and query debugging |
| `PUSHGATEWAY_HOST_PORT` | `14091` | `monitoring-pushgateway` | `9091` | Pushgateway UI and local metrics debugging |
| `ALERTMANAGER_HOST_PORT` | `14093` | `monitoring-alertmanager` | `9093` | Alertmanager UI and alert routing debugging |
| `CADVISOR_HOST_PORT` | `14200` | `monitoring-cadvisor` | `8080` | Container-level runtime metrics debugging |
| `GRAFANA_HOST_PORT` | `14300` | `monitoring-grafana` | `3000` | Grafana dashboards |
| `MAILHOG_SMTP_HOST_PORT` | `15025` | `monitoring-mailhog` | `1025` | Local SMTP capture endpoint |
| `MAILHOG_UI_HOST_PORT` | `15080` | `monitoring-mailhog` | `8025` | MailHog web UI |

## Local production-like host-exposed services

| Variable | Default | Service | Internal port | Reason |
| -------- | ------- | ------- | ------------- | ------ |
| `API_HOST_PORT` | `10000` | `api-dev` | `10000` | FastAPI prediction API and OpenAPI docs |
| `AIRFLOW_HOST_PORT` | `12080` | `airflow-api-server` | `8080` | Airflow UI and API for local DAG operations |
| `MLFLOW_HOST_PORT` | `13001` | `mlflow-server` | `5000` | MLflow tracking UI and host-side MLflow clients |
| `GRAFANA_HOST_PORT` | `14300` | `monitoring-grafana` | `3000` | Grafana dashboards |

## Local URLs

With the default `.env.template` values, the main local URLs are:

| Service | URL | Runtime |
| ------- | --- | ------- |
| FastAPI docs | `http://localhost:10000/docs` | Dev and prod-like |
| Airflow UI | `http://localhost:12080` | Dev and prod-like |
| Flower UI | `http://localhost:12555` | Dev only |
| MinIO API | `http://localhost:13000` | Dev only |
| MLflow UI | `http://localhost:13001` | Dev and prod-like |
| MinIO console | `http://localhost:13002` | Dev only |
| Prometheus UI | `http://localhost:14090` | Dev only |
| Pushgateway UI | `http://localhost:14091` | Dev only |
| Alertmanager UI | `http://localhost:14093` | Dev only |
| cAdvisor UI | `http://localhost:14200` | Dev only |
| Grafana UI | `http://localhost:14300` | Dev and prod-like |
| MailHog UI | `http://localhost:15080` | Dev only |

## Internal-only services

The following service families are intentionally not published on host ports:

| Service or family | Runtime | Reason |
| ----------------- | ------- | ------ |
| `ml-ingest-*` | Dev and prod-like | One-off ML pipeline container triggered by Compose or orchestration |
| `ml-features-*` | Dev and prod-like | One-off ML pipeline container triggered by Compose or orchestration |
| `ml-models-*` | Dev and prod-like | One-off ML pipeline container triggered by Compose or orchestration |
| `mlflow-postgres` | Dev and prod-like | MLflow internal metadata database |
| `mlflow-mc-init` | Dev and prod-like | One-off MinIO initialization helper |
| `airflow-postgres` | Dev and prod-like | Airflow internal metadata database |
| `airflow-redis` | Dev and prod-like | Airflow Celery broker |
| `airflow-init` | Dev and prod-like | One-off Airflow initialization helper |
| `airflow-scheduler` | Dev and prod-like | Internal Airflow scheduler process |
| `airflow-dag-processor` | Dev and prod-like | Internal Airflow DAG parsing process |
| `airflow-worker` | Dev and prod-like | Internal Airflow Celery worker process |
| `airflow-triggerer` | Dev and prod-like | Internal Airflow async trigger process |
| MinIO API and console | Prod-like | Internal artifact backend; host UI kept dev-only |
| Prometheus, Pushgateway, Alertmanager, cAdvisor, MailHog | Prod-like | Internal support services; host UI kept dev-only except Grafana |

`airflow-redis` used to publish `6379` on the host. It is now internal-only
because no documented local workflow requires direct host access to the broker.

## Validation

The expected configuration validation commands are:

```bash
make dev-compose-config
make prod-compose-config
```
