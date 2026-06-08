# Runtime communication matrix

This document describes the implemented local Docker Compose communication model.
It is a current-state runtime architecture reference.

The current runtime has two Compose views:

| Runtime | Compose file | Communication model |
| ------- | ------------ | ------------------- |
| Development | `docker/dev/docker-compose.yaml` | Broad local integration, DockerOperator execution, root `data/logs/models` mounts. |
| Local production-like | `docker/prod/docker-compose.yaml` | Functional networks, reduced host exposure, no Docker socket in Airflow, isolated `docker/prod/runtime` workspaces, and an internal runner API boundary. |

Host port ranges and local URLs are documented in
[`../current-runtime-and-operations/ports-and-services.md`](../current-runtime-and-operations/ports-and-services.md).
Runtime ownership and current exceptions are documented in
[`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md).

## Development communication model

The development runtime keeps broad local visibility and current Airflow
DockerOperator jobs.

| Boundary | Main services | Purpose |
| -------- | ------------- | ------- |
| `airflow_net` | Airflow services, Airflow metadata DB, Redis, ML jobs | Airflow control plane and DockerOperator job context. |
| `mlflow_net` | MLflow server, MLflow PostgreSQL, MinIO, model jobs | ML tracking and artifact logging. |
| `mlops_net` | API, monitoring, Pushgateway, MailHog, cAdvisor, selected cross-stack services | Local integration and debugging. |
| Root mounts | `data`, `models`, `logs` | Development, DVC, and host-visible artifacts. |
| Docker socket | development `airflow-worker` only | Local DockerOperator execution. |

The Docker socket path is a deliberate development exception. It must not be
reintroduced in `docker/prod`.

## Production-like communication model

The production-like runtime uses functional networks implemented in
[`local-prod-network-topology.md`](local-prod-network-topology.md).

| Boundary | Main services | Purpose |
| -------- | ------------- | ------- |
| `orchestration_net` | Airflow API, scheduler, DAG processor, triggerer, worker, Airflow PostgreSQL, Redis | Airflow control plane and metadata. |
| `pipeline_runtime_net` | Airflow worker, API, job runner API, ML jobs, Pushgateway | Runtime handoff between orchestration, business jobs, refresh, and batch metrics. |
| `tracking_client_net` | ML jobs, MLflow server | MLflow client calls. |
| `tracking_backend_net` | MLflow server, MLflow PostgreSQL, MinIO, MC init | Private tracking metadata and artifact backend. |
| `observability_net` | Prometheus, Grafana, Alertmanager, cAdvisor, Pushgateway, API metrics | Scrapes, dashboards, and local alerts. |
| `dev_support_net` | MailHog, Alertmanager, Airflow email clients | Local support services. |
| Runtime mounts | `docker/prod/runtime/data`, `docker/prod/runtime/models`, `docker/prod/runtime/logs`, `docker/prod/runtime/artifacts` | Production-like generated data, models, logs, and artifact manifests. |

## Current service-to-service paths

| Source | Target | Runtime | Mechanism | Reason |
| ------ | ------ | ------- | --------- | ------ |
| Airflow services | Airflow PostgreSQL | Dev and prod-like | Compose DNS, PostgreSQL | Metadata database and result backend. |
| Airflow services | Airflow Redis | Dev and prod-like | Compose DNS, Redis | Celery broker. |
| Airflow services | Airflow API server | Prod-like | Compose DNS, HTTP | Internal Airflow execution API. |
| Airflow DAG tasks | ML job containers | Dev only | Docker socket / DockerOperator | Current local development ML execution. |
| Airflow DAG tasks | `api-dev` | Dev and prod-like | HTTP on internal network | Refresh API after successful DAG runs. |
| ML jobs | Pushgateway | Dev and prod-like | HTTP | Batch metric push when enabled. |
| ML model jobs | MLflow server | Dev and prod-like | HTTP | Run, parameter, metric, model, and artifact logging. |
| MLflow server | MLflow PostgreSQL | Dev and prod-like | PostgreSQL | MLflow backend store. |
| MLflow server | MinIO | Dev and prod-like | S3/HTTP | MLflow artifact store. |
| Prometheus | API, Pushgateway, cAdvisor | Dev and prod-like | HTTP scrape | Metrics collection. |
| Grafana | Prometheus | Dev and prod-like | HTTP | Provisioned datasource. |
| Alertmanager | MailHog | Dev and prod-like | SMTP | Local alert capture. |
| API | Prediction artifacts | Dev and prod-like | Filesystem | Current prediction serving input. |
| `job-runner-api` | In-memory state | Prod-like | Process memory | Local job status persistence for accepted typed requests. |

`job-runner-api` exposes an internal HTTP API on `pipeline_runtime_net`. The
current implementation accepts and stores typed job requests without executing ML
workloads.

## Host exposure summary

Development intentionally exposes more local UIs. The production-like runtime
publishes only operator-facing services by default.

| Service family | Development exposure | Production-like exposure |
| -------------- | -------------------- | ------------------------ |
| FastAPI prediction API | Host exposed | Host exposed. |
| Job runner API | Not present | Internal-only. |
| Airflow API/UI | Host exposed | Host exposed. |
| MLflow UI/API | Host exposed | Host exposed. |
| Grafana | Host exposed | Host exposed. |
| MinIO API/console | Host exposed | Internal-only. |
| Prometheus | Host exposed | Internal-only. |
| Pushgateway | Host exposed | Internal-only. |
| Alertmanager | Host exposed | Internal-only. |
| cAdvisor | Host exposed | Internal-only. |
| MailHog | Host exposed | Internal-only. |
| PostgreSQL and Redis | Internal-only | Internal-only. |

See [`../current-runtime-and-operations/ports-and-services.md`](../current-runtime-and-operations/ports-and-services.md)
for exact ports and URLs.

## Shared mount ownership

| Mount | Runtime | Current role |
| ----- | ------- | ------------ |
| Root `data` | Dev | Raw, interim, processed, final data for DVC/local workflows. |
| Root `models` | Dev | Development model artifacts. |
| Root `logs` | Dev | Development logs. |
| `docker/prod/runtime/data` | Prod-like | Generated production-like data. |
| `docker/prod/runtime/models` | Prod-like | Generated production-like model artifacts. |
| `docker/prod/runtime/logs` | Prod-like | Production-like service, batch, and runner API logs. |
| `docker/prod/runtime/artifacts` | Prod-like | Manifest-first handoff root with run-scoped manifests and promoted `current.json` files. |
| Root raw CSV | Prod-like | Read-only business source input. |

`job-runner-api` mounts only `docker/prod/runtime/logs/job-runner`. It does not
mount data, model, artifact, Docker socket, or Docker runtime paths.

## Operational guardrails

- Do not add Docker socket mounts to production-like Airflow services.
- Do not copy the broad `mlops_net` development model into `docker/prod`.
- Prefer explicit functional networks over pairwise networks unless sensitive
  state or privileged control surfaces require isolation.
- Keep host exposure in `../current-runtime-and-operations/ports-and-services.md`
  synchronized with Compose.
- Keep typed job contracts in `src/pipeline/contracts/` independent from FastAPI,
  Airflow, Docker, and concrete runner implementation code.
