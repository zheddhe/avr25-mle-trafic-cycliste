# Local production-like network topology

This document describes the implemented `docker/prod` functional network topology
after Phase 7. It started as the Phase 7 design artifact for issue #55 and now
acts as the reference for current production-like Compose service placement and
for Phase 8 service additions.

`docker/dev` intentionally keeps its broader development-oriented network model.
`docker/prod` uses bounded functional domains instead of the broad development
`mlops_net`.

## Documentation role

| Document | Role |
| -------- | ---- |
| [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md) | Operational runtime guide and workspace ownership. |
| [`runtime-communication-matrix.md`](runtime-communication-matrix.md) | Current service traffic and Phase 8 additions. |
| [`runtime-security-boundaries.md`](runtime-security-boundaries.md) | Runtime identities and security boundaries. |
| [`../next-phase-design/artifact-handoff-strategy.md`](../next-phase-design/artifact-handoff-strategy.md) | Phase 8 artifact handoff contract. |
| [`../next-phase-design/airflow-job-runner-strategy.md`](../next-phase-design/airflow-job-runner-strategy.md) | Phase 8 runner execution target. |

## Design principle

The production-like runtime does not create one network per service pair. It uses
bounded functional domains and allows a few edge services to join several domains
when they are explicit gateways.

A network is justified when at least one of these statements is true:

- it protects stateful backend services such as PostgreSQL, Redis, or MinIO;
- it represents a stable many-to-one communication pattern such as monitoring;
- it separates local-development support services from production-like runtime services;
- it carries privileged or controlled job-execution concerns such as the future
  runner API and typed workers.

## Implemented `docker/prod` networks

| Network | Responsibility | Current members or expected members |
| ------- | -------------- | ----------------------------------- |
| `orchestration_net` | Airflow control plane, metadata DB, broker, and internal Airflow execution API. | Airflow API, scheduler, DAG processor, triggerer, worker, init, PostgreSQL, Redis. |
| `pipeline_runtime_net` | Runtime control and data-pipeline handoff between orchestration, API refresh, batch jobs, and batch metric writes. | Airflow worker, API, ML jobs, Pushgateway, future `job-runner-api`. |
| `tracking_client_net` | MLflow client API calls from ML workloads. | ML jobs and `mlflow-server`. |
| `tracking_backend_net` | Private MLflow metadata and artifact backends. | `mlflow-server`, `mlflow-postgres`, `mlflow-minio`, `mlflow-mc-init`. |
| `observability_net` | Monitoring, dashboard, alert routing, and scrape access to selected metric endpoints. | Prometheus, Grafana, Alertmanager, cAdvisor, Pushgateway, API metrics endpoint, selected exporters. |
| `dev_support_net` | Local support services that should not be part of the production-like core. | MailHog, Airflow services that send local email, Alertmanager in local email mode. |

`artifact_handoff_net` is not implemented. The current Phase 8 handoff strategy
uses manifests that reference local runtime paths and optional MinIO object URIs.
Introduce a dedicated artifact network only if a future object-store or promotion
service needs a separate boundary.

## Gateway services

A few services deliberately join multiple networks. These services are not
accidental broad-network members; they bridge bounded domains.

| Service | Networks | Gateway role |
| ------- | -------- | ------------ |
| `airflow-worker` | `orchestration_net`, `pipeline_runtime_net`, `dev_support_net` | Runs orchestration tasks and, after Phase 8, submits work to the runner API. |
| future `job-runner-api` | `pipeline_runtime_net`, optional `observability_net` | Accepts typed job submissions and exposes runner health/metrics internally. |
| future typed ML workers | `pipeline_runtime_net`, plus `tracking_client_net` for model workers | Execute allowed business jobs without exposing Docker runtime control to Airflow. |
| `api-dev` | `pipeline_runtime_net`, `observability_net` | Receives refresh calls and exposes metrics. Host publication remains local ingress. |
| `mlflow-server` | `tracking_client_net`, `tracking_backend_net`, `observability_net` | Accepts MLflow client calls and owns backend access to PostgreSQL and MinIO. |
| `monitoring-pushgateway` | `pipeline_runtime_net`, `observability_net` | Receives batch metrics writes and exposes them for Prometheus scrape. |
| `monitoring-alertmanager` | `observability_net`, `dev_support_net` | Receives Prometheus alerts and sends local development email. |

This gateway pattern replaces `mlops_net` in the production-like runtime:
cross-domain access is explicit and attached only to services that need it.

## Current dev network review

The development runtime still uses:

| Network | Role in `docker/dev` |
| ------- | -------------------- |
| `airflow_net` | Airflow control plane plus DockerOperator execution context. |
| `mlflow_net` | MLflow tracking and artifact plane. |
| `mlops_net` | Broad local integration network for API, monitoring, Pushgateway, MailHog, cAdvisor, and cross-stack development access. |

This model remains valid for development because it supports broad host
visibility, DockerOperator jobs, and local debugging. It should not be copied
into `docker/prod`.

## Required service-name dependencies

| Source service | Target service | DNS name | Port | Network | Reason |
| -------------- | -------------- | -------- | ---- | ------- | ------ |
| `monitoring-prometheus` | `api-dev` | `api-dev` | `10000` | `observability_net` | FastAPI `/metrics` scrape. |
| `monitoring-prometheus` | `monitoring-cadvisor` | `monitoring-cadvisor` | `8080` | `observability_net` | Container metric scrape. |
| `monitoring-prometheus` | `monitoring-pushgateway` | `monitoring-pushgateway` | `9091` | `observability_net` | Batch metric scrape. |
| `monitoring-grafana` | `monitoring-prometheus` | `monitoring-prometheus` | `9090` | `observability_net` | Provisioned datasource. |
| `monitoring-alertmanager` | `monitoring-mailhog` | `monitoring-mailhog` | `1025` | `dev_support_net` | Local alert email capture. |
| Airflow services | `airflow-postgres` | `airflow-postgres` | `5432` | `orchestration_net` | Airflow metadata DB and result backend. |
| Airflow services | `airflow-redis` | `airflow-redis` | `6379` | `orchestration_net` | Celery broker. |
| Airflow services | `airflow-api-server` | `airflow-api-server` | `8080` | `orchestration_net` | Internal Airflow execution API. |
| Airflow DAG tasks | `api-dev` | `api-dev` | `10000` | `pipeline_runtime_net` | Authenticated API refresh after successful DAG runs. |
| ML jobs | `monitoring-pushgateway` | `monitoring-pushgateway` | `9091` | `pipeline_runtime_net` | Push batch job metrics. |
| ML jobs | `mlflow-server` | `mlflow-server` | `5000` | `tracking_client_net` | Log runs, metrics, params, and artifacts. |
| `mlflow-server` | `mlflow-postgres` | `mlflow-postgres` | `5432` | `tracking_backend_net` | MLflow backend store. |
| `mlflow-server` | `mlflow-minio` | `mlflow-minio` | `9000` | `tracking_backend_net` | MLflow artifact store. |
| `mlflow-mc-init` | `mlflow-minio` | `mlflow-minio` | `9000` | `tracking_backend_net` | Bootstrap the MLflow bucket. |

## Phase 8 additions

Phase 8 should place new services according to these rules:

- `job-runner-api` belongs on `pipeline_runtime_net` and should not be published
  to the host by default.
- Runner health or metrics may be attached to `observability_net` if Prometheus
  needs to scrape them.
- Typed ML workers should attach only to the networks required for their stage.
- Model workers may attach to `tracking_client_net`; ingest and feature workers
  should not receive MLflow or MinIO access unless they need it.
- Artifact handoff should follow
  [`../next-phase-design/artifact-handoff-strategy.md`](../next-phase-design/artifact-handoff-strategy.md)
  before introducing any new network.

## Services that should not share a broad network

The following services should not be colocated on a single broad production-like
network:

- Airflow PostgreSQL and Redis should not share a platform network with API,
  Grafana, Prometheus, MLflow, or MailHog.
- MLflow PostgreSQL should not share a platform network with Airflow, API,
  monitoring, or local email services.
- MinIO should stay inside the tracking backend boundary unless it becomes an
  explicit artifact handoff boundary with scoped credentials.
- MailHog should remain local-support-only.
- `api-dev` should not share Airflow metadata, Redis, MLflow backend, or MinIO
  backend networks.
- Docker socket access should not be introduced in the production-like runtime.

## Topology sketch

```mermaid
flowchart LR
    airflow[Airflow services] --> airflow_db[(airflow-postgres)]
    airflow --> airflow_redis[(airflow-redis)]
    airflow --> runner[Future job-runner-api]
    runner --> api[api-dev]
    runner --> jobs[Typed ML workers]

    jobs --> mlflow[mlflow-server]
    mlflow --> mlflow_db[(mlflow-postgres)]
    mlflow --> minio[(mlflow-minio)]

    jobs --> pushgateway[monitoring-pushgateway]
    pushgateway --> prometheus[monitoring-prometheus]
    api --> prometheus
    prometheus --> grafana[monitoring-grafana]
    prometheus --> alertmanager[monitoring-alertmanager]
    alertmanager --> mailhog[monitoring-mailhog]
```

## Validation

Network validation is expected through the runtime configuration and future smoke
checks:

```bash
make prod-compose-config
make prod-start
make prod-ps
```

A dedicated production-like smoke target is tracked by Phase 8 and should verify
runner reachability, API metrics scraping, artifact manifest availability, and the
absence of Docker socket mounts in production-like Airflow services.
