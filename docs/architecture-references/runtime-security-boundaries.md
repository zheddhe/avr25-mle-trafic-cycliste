# Runtime security boundaries and identities

This document describes the current local runtime security boundaries and service
identities.

It is not a production security model. It is a local MLOps boundary reference
that explains which privileges are acceptable for development and which
boundaries are implemented in `docker/prod`.

## Scope and source documents

| Document | Role |
| -------- | ---- |
| [`runtime-communication-matrix.md`](runtime-communication-matrix.md) | Current service traffic and network paths. |
| [`local-prod-network-topology.md`](local-prod-network-topology.md) | Implemented functional network topology for `docker/prod`. |
| [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md) | Runtime usage, workspaces, and current exceptions. |
| [`../current-runtime-and-operations/ports-and-services.md`](../current-runtime-and-operations/ports-and-services.md) | Host exposure inventory. |
| [`../next-phase-design/artifact-handoff-strategy.md`](../next-phase-design/artifact-handoff-strategy.md) | Manifest-first artifact promotion contract and open handoff gaps. |
| [`../next-phase-design/airflow-job-runner-strategy.md`](../next-phase-design/airflow-job-runner-strategy.md) | Runner execution coordination and remaining validation gaps. |

Communication paths justify permissions, not the opposite. A container, network,
credential, volume, or elevated runtime capability should exist because a
documented service dependency needs it.

## Current boundary summary

| Boundary | Development runtime | Production-like runtime |
| -------- | ------------------- | ----------------------- |
| API serving | Host-exposed `api-dev`; reads dev data workspace. | Host-exposed `api-dev`; reads production-like runtime prediction workspace. |
| Runner API | Not present. | Internal-only `job-runner-api`; accepts typed job requests, delegates to internal ML step services, and stores statuses in memory. |
| Orchestration | Airflow uses DockerOperator and Docker socket. | Airflow does not mount Docker socket. |
| Tracking | MLflow, PostgreSQL, and MinIO are visible for local debugging. | MLflow UI/API is host-exposed; PostgreSQL and MinIO stay internal. |
| Monitoring | Prometheus, Pushgateway, Alertmanager, cAdvisor, Grafana, MailHog UIs are exposed for local debugging. | Grafana is host-exposed; supporting monitoring services stay internal. |
| Data and artifacts | Root `data`, `models`, and `logs` are writable dev/DVC workspaces. | `docker/prod/runtime` owns generated runtime data, models, logs, and artifacts. |
| Job execution | Airflow worker can control the Docker daemon through Docker socket. | Airflow submits typed jobs to `job-runner-api`; concrete execution happens inside internal ML step services. |

## Runtime identities

| Runtime identity | Current model | Boundary notes |
| ---------------- | ------------- | -------------- |
| Host operator | Local user running `make` and `docker compose` with Docker group access. | Docker group membership is high privilege and should not be confused with application identity. |
| `api-dev` | Custom dev/prod images; prod-like image runs as non-root app user. | Reads prediction artifacts and exposes HTTP on port `10000`. |
| `job-runner-api` | Custom prod-like image; runs as non-root app user. | Exposes internal HTTP on port `10080`, keeps in-memory status, delegates through HTTP, and has no Docker socket mount. |
| `ml-ingest-*` | Custom dev/prod images; prod-like image runs as non-root app user. | Reads raw data and writes interim data/logs. Prod-like service exposes internal HTTP on port `10081`. |
| `ml-features-*` | Custom dev/prod images; prod-like image runs as non-root app user. | Reads interim data and writes processed data/logs. Prod-like service exposes internal HTTP on port `10082`. |
| `ml-models-*` | Custom dev/prod images; prod-like image runs as non-root app user. | Reads processed data, writes forecasts/models/logs, logs evidence to MLflow, and exposes internal HTTP on port `10083` in prod-like mode. |
| Airflow services | Upstream Airflow image and supported runtime user model. | Development worker has Docker socket exception; production-like worker does not. |
| `mlflow-server` | Upstream MLflow image. | Owns access to MLflow PostgreSQL and MinIO backend. |
| PostgreSQL, Redis, MinIO | Upstream images with private volumes. | Internal stateful services; do not publish DB/broker ports to host. |
| Prometheus/Grafana/Alertmanager | Upstream monitoring images. | Grafana is operator-facing; Prometheus and Alertmanager are internal in prod-like runtime. |
| cAdvisor | Privileged runtime metrics collector. | Local observability exception requiring host/runtime access. |
| MailHog | Local SMTP capture service. | Development helper; internal-only in prod-like runtime. |

## Docker socket boundary

`/var/run/docker.sock` is a privileged local-development boundary.

A container with write access to the Docker socket can ask the host Docker daemon
to create containers, mount host paths, join networks, and access secrets or data
available to the Docker daemon. In practice, this is close to host-level control.

Current usage:

- development `airflow-worker` uses the socket to create ML workload containers
  from Airflow DAG tasks;
- cAdvisor uses runtime mounts and the Docker socket to collect local container
  metrics.

Production-like rule:

- do not add Docker socket mounts to Airflow services in `docker/prod`;
- do not replace the missing socket with an untyped shell-execution API;
- keep job execution behind typed contracts and controlled service boundaries.

`job-runner-api` does not import the Docker SDK, does not mount the Docker socket,
and does not expose a generic shell command interface. It delegates validated job
requests to internal ML step services over `pipeline_runtime_net`.

## Host exposure boundary

Development exposes broad local UIs for debugging. The production-like runtime
publishes only operator-facing services by default:

- FastAPI prediction API;
- Airflow API/UI;
- MLflow UI/API;
- Grafana.

`job-runner-api`, ML step services, MinIO, Prometheus, Pushgateway, Alertmanager,
MailHog, cAdvisor, Redis, and PostgreSQL remain internal in `docker/prod`.

Exact port assignments are documented in
[`../current-runtime-and-operations/ports-and-services.md`](../current-runtime-and-operations/ports-and-services.md).

## Volume and artifact boundary

| Mount or volume | Runtime | Boundary expectation |
| --------------- | ------- | -------------------- |
| Root `data` | Dev | DVC and local development data workspace. |
| Root `models` | Dev | DVC and local development model workspace. |
| Root `logs` | Dev | Development service and batch logs. |
| Root raw CSV | Prod-like | Read-only source input into ingestion. |
| `docker/prod/runtime/data` | Prod-like | Generated runtime data owned by ML step services. |
| `docker/prod/runtime/models` | Prod-like | Runtime model artifacts owned by `ml-models-prod`. |
| `docker/prod/runtime/logs` | Prod-like | Runtime service, job, and runner API logs. |
| `docker/prod/runtime/artifacts` | Prod-like | Manifest-first artifact handoff root owned by ML step services. |
| Service-owned Docker volumes | Dev and prod-like | Stateful backends for PostgreSQL, Redis, MinIO, Prometheus, Grafana, and Alertmanager. |

The runner API currently mounts only its service log path. It has no read or write
access to ML data, model artifacts, promoted manifests, Docker runtime paths,
tracking backends, or stateful backend volumes. Concrete ML step services own the
runtime data, model, log, artifact, and optional tracking mounts they need.

## Credential and secret review

`.env.template` separates local developer settings, DVC/DagsHub remote
credentials, MLflow target variables, MinIO artifact credentials, Airflow secrets,
Grafana admin credentials, API demo credentials, monitoring settings, and time
zone configuration.

Local-only defaults and placeholders that must not be promoted to production:

- Airflow simple auth defaults;
- API demo defaults;
- Airflow development connection defaults;
- Airflow development variables with local MinIO values;
- Grafana local admin placeholders;
- MailHog unauthenticated SMTP/UI;
- Prometheus, Pushgateway, Alertmanager, and cAdvisor local UIs without auth.

## Non-root and capability rules

Implemented in `docker/prod` for custom API, runner API, and ML images:

- non-root application user;
- default UID/GID compatible with common local bind mounts;
- `cap_drop: ALL`;
- `no-new-privileges:true`.

Do not assume upstream infrastructure images can safely be overridden to non-root
without service-specific validation.

## Validation expectations

For current runtime boundaries:

```bash
make dev-compose-config
make prod-compose-config
make prod-start
make prod-ps
```

Current production-like boundary validation should prove:

- Airflow services do not mount `/var/run/docker.sock`;
- `job-runner-api` is internal-only;
- ML step services are internal-only;
- `job-runner-api` has no Docker SDK import, Docker socket mount, or broad runtime
  data/model/artifact mounts;
- custom prod-like API, runner API, and ML images keep their non-root settings;
- generated runtime outputs stay under `docker/prod/runtime`.
