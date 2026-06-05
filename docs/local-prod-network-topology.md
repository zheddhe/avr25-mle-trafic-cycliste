# Local production-like network topology

This document is the Phase 7 design artifact for issue #55. It derives a
target local production-like Docker Compose topology from the current
service-name dependencies.

The current `docker/dev` runtime is intentionally unchanged by this story. The
target network set below is a proposal for a future `docker/prod` or
`docker/local-prod` Compose implementation.

## Scope and inputs

The design reviews these runtime sources:

| Source | Use in this design |
| ------ | ------------------ |
| `docker-compose.yaml` | Current services, networks, mounts, and host exposure. |
| `.env.template` | Runtime names, ports, credentials, and local defaults. |
| `docs/runtime-communication-matrix.md` | Current service traffic and host exposure. |
| `docs/runtime-security-boundaries.md` | Runtime identities and boundary intent. |
| `docs/repository-structure.md` | `docker/dev` stability and future `docker/prod` split. |
| `docs/ports-and-services.md` | Host URLs and internal-only service ports. |
| `docker/dev/prometheus/prometheus.yml` | Prometheus scrape and alert routing targets. |
| `docker/dev/grafana/provisioning/datasources/datasource.yaml` | Grafana to Prometheus service-name dependency. |
| `docker/dev/airflow/config/connections.json` | Airflow `api_dev` HTTP target. |
| `docker/dev/airflow/config/variables.json` | DockerOperator images, network, MLflow, MinIO, and Pushgateway values. |

The goal is network design only. It does not implement secrets, ingress,
Kubernetes, worker execution replacement, or a Compose refactor.

## Current network review

### `airflow_net`

`airflow_net` is the Airflow control-plane network. It correctly groups the
Airflow API server, scheduler, DAG processor, triggerer, worker, Flower,
Airflow PostgreSQL, Redis, initialization task, and Airflow-managed jobs.

Target decision:

- keep Airflow metadata and broker services internal to an orchestration
  boundary;
- keep Airflow UI/API access as an explicit host or ingress concern, not a
  shared platform network concern;
- remove broad Airflow-to-platform attachment in the target model unless a
  documented DAG dependency needs it;
- treat `airflow-worker` Docker socket access as a local-dev exception until a
  worker-pool or job-runner story replaces it.

`api-dev` does not need to share the Airflow metadata network only to receive
`POST /admin/refresh`. That dependency should move to a serving operations
boundary.

### `mlflow_net`

`mlflow_net` is the ML tracking and artifact boundary. It correctly groups
`mlflow-server`, `mlflow-postgres`, `mlflow-minio`, the MinIO bootstrap helper,
and ML workloads that log tracking data or artifacts.

Target decision:

- keep `mlflow-postgres` private to tracking backend services;
- keep `mlflow-minio` private to tracking/artifact services by default;
- attach ML workloads through an explicit tracking client boundary;
- avoid exposing tracking backend services on a broad integration network;
- expose local MLflow and MinIO UIs only through deliberate local host or dev
  ingress paths.

### `mlops_net`

`mlops_net` is currently a broad local integration network. It is useful for
development because it groups API, monitoring, Pushgateway, MailHog, cAdvisor,
and selected cross-stack services.

Target decision:

- do not keep `mlops_net` as a broad local production-like network;
- split it into functional networks with clear names;
- keep the name only as a legacy `docker/dev` concept, or rename it to
  `dev_integration_net` if a future cleanup wants to make its purpose explicit;
- do not migrate the name into `docker/prod` unless it is narrowed to a single
  responsibility.

## Required service-name dependencies

| Source service | Target service | DNS name | Port | Protocol | Reason | Proposed network |
| -------------- | -------------- | -------- | ---- | -------- | ------ | ---------------- |
| `monitoring-prometheus` | `monitoring-prometheus` | `monitoring-prometheus` | `9090` | HTTP | Self scrape and readiness checks. | `monitoring_scrape_net` |
| `monitoring-prometheus` | `monitoring-cadvisor` | `monitoring-cadvisor` | `8080` | HTTP | Container metric scrape. | `monitoring_scrape_net` |
| `monitoring-prometheus` | `api-dev` | `api-dev` | `10000` | HTTP | FastAPI `/metrics` scrape. | `monitoring_scrape_net` |
| `monitoring-prometheus` | `monitoring-pushgateway` | `monitoring-pushgateway` | `9091` | HTTP | Batch metric scrape. | `monitoring_scrape_net` |
| `monitoring-prometheus` | `monitoring-alertmanager` | `monitoring-alertmanager` | `9093` | HTTP | Alert routing target. | `alerting_net` |
| `monitoring-grafana` | `monitoring-prometheus` | `monitoring-prometheus` | `9090` | HTTP | Provisioned datasource. | `monitoring_query_net` |
| `monitoring-alertmanager` | `monitoring-mailhog` | `monitoring-mailhog` | `1025` | SMTP | Local alert email test route. | `email_dev_net` |
| Airflow services | `airflow-postgres` | `airflow-postgres` | `5432` | PostgreSQL | Airflow metadata DB and result backend. | `orchestration_net` |
| Airflow services | `airflow-redis` | `airflow-redis` | `6379` | Redis | Celery broker. | `orchestration_net` |
| Airflow services | `airflow-api-server` | `airflow-api-server` | `8080` | HTTP | Internal Airflow execution API. | `orchestration_net` |
| Airflow DAG tasks | `api-dev` | `api-dev` | `10000` | HTTP | Refresh API data after successful DAG runs. | `serving_ops_net` |
| Airflow DAG tasks | ML job containers | Docker API | N/A | Docker API | Create ingestion, features, and model jobs. | `job_execution_net` |
| Airflow services | `monitoring-mailhog` | `monitoring-mailhog` | `1025` | SMTP | Local Airflow email capture. | `email_dev_net` |
| ML jobs | `monitoring-pushgateway` | `monitoring-pushgateway` | `9091` | HTTP | Push batch job metrics. | `batch_metrics_net` |
| ML jobs | `mlflow-server` | `mlflow-server` | `5000` | HTTP | Log runs, metrics, params, and artifacts. | `tracking_client_net` |
| `mlflow-server` | `mlflow-postgres` | `mlflow-postgres` | `5432` | PostgreSQL | MLflow backend store. | `tracking_backend_net` |
| `mlflow-server` | `mlflow-minio` | `mlflow-minio` | `9000` | HTTP/S3 | MLflow artifact store. | `tracking_backend_net` |
| `mlflow-mc-init` | `mlflow-minio` | `mlflow-minio` | `9000` | HTTP/S3 | Bootstrap the MLflow bucket. | `tracking_backend_net` |
| API and ML jobs | promoted datasets | No DNS | N/A | Filesystem or S3 | Read/write released prediction artifacts. | `artifact_handoff_net` |

The Docker API entry is a logical dependency. It is not a Compose DNS
dependency, but it must stay visible in the migration plan because the current
Airflow worker creates ML workload containers through the local Docker socket.

## Target Compose network set

| Network | Responsibility | Expected members |
| ------- | -------------- | ---------------- |
| `orchestration_net` | Airflow control plane, metadata DB, and broker. | Airflow API, scheduler, DAG processor, triggerer, worker, init, Flower, PostgreSQL, Redis. |
| `job_execution_net` | Airflow-created ML job execution and handoff. | Airflow worker or target job runner, `ml-ingest-*`, `ml-features-*`, `ml-models-*`. |
| `serving_net` | User-facing API runtime. | `api-dev` or future API service. |
| `serving_ops_net` | Operational API actions from orchestration. | Airflow DAG task identity and `api-dev`. |
| `tracking_client_net` | MLflow client calls from ML workloads. | ML jobs and `mlflow-server`. |
| `tracking_backend_net` | MLflow private metadata and artifact backend. | `mlflow-server`, `mlflow-postgres`, `mlflow-minio`, `mlflow-mc-init`. |
| `artifact_handoff_net` | Optional future data release or object-store handoff. | ML jobs, API, and storage service if promoted from host mounts. |
| `batch_metrics_net` | Batch metric pushes. | ML jobs and `monitoring-pushgateway`. |
| `monitoring_scrape_net` | Prometheus scrape access to metric endpoints. | Prometheus, API metrics endpoint, cAdvisor, Pushgateway, selected exporters. |
| `monitoring_query_net` | Dashboard queries against Prometheus. | Grafana and Prometheus. |
| `alerting_net` | Prometheus to Alertmanager routing. | Prometheus and Alertmanager. |
| `email_dev_net` | Local SMTP capture for development only. | Alertmanager, Airflow services that send mail, MailHog. |

`monitoring_query_net` may be folded into `monitoring_scrape_net` if the local
Compose file remains small. Keeping it separate is cleaner when Grafana should
query Prometheus without being colocated with scrape targets.

## Why monitoring needs a shared scrape boundary

Prometheus is intentionally a many-to-one scraper. Its checked-in configuration
uses service names for several targets, and future exporters will likely follow
the same pattern.

A pairwise network per Prometheus target would create operational noise:

- every new exporter would require a new Compose network;
- Prometheus would need one extra attachment per target;
- target removals would require network lifecycle cleanup;
- dashboard and alert validation would become harder to reason about.

A shared `monitoring_scrape_net` is the better local production-like boundary
when the network is limited to scrape endpoints and selected exporters. It
keeps service discovery maintainable while avoiding broad application,
database, broker, and artifact-store colocation.

## Pairwise network policy

The target design avoids pairwise networks by default. A pairwise network is
justified only when the target contains sensitive backend state or a privileged
runtime surface.

Current strong candidates for narrow or pairwise-like isolation are:

- Airflow metadata DB and Redis inside `orchestration_net`;
- MLflow PostgreSQL and MinIO inside `tracking_backend_net`;
- Docker socket or replacement job runner inside `job_execution_net`;
- local SMTP capture isolated in `email_dev_net`.

Monitoring does not qualify for pairwise links in this design because it has a
stable many-target scrape pattern.

## Services that should not share a broad network

The following services should not be colocated on a single broad network in a
local production-like runtime:

- `airflow-postgres` and `airflow-redis` should not share a platform network
  with API, Grafana, Prometheus, MLflow, or MailHog.
- `mlflow-postgres` should not share a platform network with Airflow, API,
  monitoring, or local email services.
- `mlflow-minio` should not share a broad network unless it becomes an explicit
  project object-store boundary with scoped credentials.
- `monitoring-cadvisor` should not share business API, orchestration metadata,
  or tracking backend networks because it observes runtime internals.
- `monitoring-mailhog` should remain dev-only and isolated from production-like
  serving and tracking networks.
- `api-dev` should not share Airflow metadata, Redis, MLflow backend, or MinIO
  backend networks.
- `airflow-worker` Docker socket access should not be combined with a broad
  integration network.

## Target topology sketch

```mermaid
flowchart LR
    airflow[Airflow services] --> airflow_db[(airflow-postgres)]
    airflow --> airflow_redis[(airflow-redis)]
    airflow --> api[api-dev]
    airflow --> jobs[ML jobs]

    jobs --> mlflow[mlflow-server]
    mlflow --> mlflow_db[(mlflow-postgres)]
    mlflow --> minio[(mlflow-minio)]

    jobs --> pushgateway[monitoring-pushgateway]
    prometheus[monitoring-prometheus] --> api
    prometheus --> pushgateway
    prometheus --> cadvisor[monitoring-cadvisor]
    prometheus --> alertmanager[monitoring-alertmanager]
    grafana[monitoring-grafana] --> prometheus
    alertmanager --> mailhog[monitoring-mailhog]
```

This sketch shows functional dependencies only. It is not an implementation
diff for `docker/dev`.

## Migration sequence

1. Keep `docker/dev` unchanged and treat this document as the target contract.
2. Create a separate `docker/prod` or `docker/local-prod` Compose entrypoint.
3. Add the target network names without removing existing service definitions.
4. Move monitoring first: define `monitoring_scrape_net`,
   `monitoring_query_net`, `alerting_net`, and `email_dev_net`.
5. Move MLflow backend services to `tracking_backend_net`; attach ML jobs only
   through `tracking_client_net`.
6. Move Airflow metadata and broker services to `orchestration_net`; expose API
   refresh through `serving_ops_net` instead of `airflow_net`.
7. Define `batch_metrics_net` for Pushgateway writes and keep Prometheus scrape
   access separate.
8. Define the artifact handoff contract before removing broad `data`, `models`,
   or `logs` bind mounts.
9. Replace `mlops_net` with functional networks in the target runtime.
10. Update architecture diagrams and operator documentation after validation.

## Rollback criteria

Rollback to the previous Compose topology or temporarily reattach the legacy
integration network if any of these conditions occur:

- `make compose-config` or `docker compose config` fails;
- Prometheus cannot resolve or scrape expected targets;
- Grafana cannot resolve or query Prometheus;
- Alertmanager cannot receive alerts from Prometheus;
- MailHog no longer receives local Airflow or Alertmanager email;
- Airflow cannot reach PostgreSQL, Redis, the internal API server, or the API
  refresh endpoint;
- Airflow-created ML jobs cannot start or cannot join required networks;
- ML jobs cannot log to MLflow or push metrics to Pushgateway;
- `mlflow-server` cannot reach PostgreSQL or MinIO;
- `api-dev` cannot read promoted prediction artifacts or expose `/metrics`.

## Validation commands

The documentation-only validation for this story remains:

```bash
make compose-config
```

Future implementation stories should also validate service-name resolution and
runtime readiness with commands like:

```bash
docker compose config
docker compose --profile ptf up -d
docker compose ps
docker compose exec monitoring-prometheus \
    wget -qO- http://api-dev:10000/metrics
docker compose exec monitoring-prometheus \
    wget -qO- http://monitoring-alertmanager:9093/-/ready
docker compose exec monitoring-grafana \
    wget -qO- http://monitoring-prometheus:9090/-/ready
docker compose exec airflow-worker getent hosts api-dev
docker compose exec airflow-worker getent hosts monitoring-pushgateway
docker compose exec mlflow-server getent hosts mlflow-postgres
docker compose exec mlflow-server getent hosts mlflow-minio
```

## Out-of-scope items

This design does not:

- refactor the current `docker/dev` runtime;
- implement `docker/prod`;
- replace Airflow worker execution;
- implement secrets or production ingress;
- remove host bind mounts;
- change monitoring, alerting, MLflow, MinIO, API, or Airflow service config.
