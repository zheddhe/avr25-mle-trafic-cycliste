# Runtime ports and service exposure

This document describes the local Docker Compose host port strategy. It is a
local development convention only. It does not represent a production security
model.

Docker services communicate through their internal container ports and Docker
networks. The variables below only control ports published on the host machine.

## Port ranges

| Range | Responsibility |
|-------|----------------|
| `10000-10999` | Business applications and application APIs |
| `11000-11999` | Local business debug endpoints, if needed later |
| `12000-12999` | Orchestration and workflow tools |
| `13000-13999` | Tracking and artifact systems |
| `14000-14999` | Observability and alerting |
| `15000-15999` | Development-only helpers and debug tools |

## Host-exposed services

| Variable | Default | Service | Internal port | Reason |
|----------|---------|---------|---------------|--------|
| `API_HOST_PORT` | `10000` | `api-dev` | `10000` | FastAPI prediction API and OpenAPI docs |
| `AIRFLOW_HOST_PORT` | `12080` | `airflow-webserver` | `8080` | Airflow UI for local DAG operations |
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

## Local URLs

With the default `.env.template` values, the main local URLs are:

| Service | URL |
|---------|-----|
| FastAPI docs | `http://localhost:10000/docs` |
| Airflow UI | `http://localhost:12080` |
| Flower UI | `http://localhost:12555` |
| MinIO API | `http://localhost:13000` |
| MLflow UI | `http://localhost:13001` |
| MinIO console | `http://localhost:13002` |
| Prometheus UI | `http://localhost:14090` |
| Pushgateway UI | `http://localhost:14091` |
| Alertmanager UI | `http://localhost:14093` |
| cAdvisor UI | `http://localhost:14200` |
| Grafana UI | `http://localhost:14300` |
| MailHog UI | `http://localhost:15080` |

## Internal-only services

The following services are intentionally not published on host ports:

| Service | Reason |
|---------|--------|
| `ml-ingest-dev` | One-off ML pipeline container triggered by Compose or Airflow |
| `ml-features-dev` | One-off ML pipeline container triggered by Compose or Airflow |
| `ml-models-dev` | One-off ML pipeline container triggered by Compose or Airflow |
| `mlflow-postgres` | MLflow internal metadata database |
| `mlflow-mc-init` | One-off MinIO initialization helper |
| `airflow-postgres` | Airflow internal metadata database |
| `airflow-redis` | Airflow Celery broker, used only inside `airflow_net` |
| `airflow-init` | One-off Airflow initialization helper |
| `airflow-scheduler` | Internal Airflow scheduler process |
| `airflow-worker` | Internal Airflow Celery worker process |

`airflow-redis` used to publish `6379` on the host. It is now internal-only
because no documented local workflow requires direct host access to the broker.

## Validation

The expected configuration validation command is:

```bash
make compose-config
```

It runs:

```bash
docker compose --profile all config
```
