# Runtime communication matrix

This document describes the implemented local Docker Compose communication model.
It is a current-state runtime architecture reference.

The current runtime has two Compose views:

| Runtime | Compose file | Communication model |
| ------- | ------------ | ------------------- |
| Development | `docker/dev/docker-compose.yaml` | Broad local integration, DockerOperator execution, root `data/logs/models` mounts. |
| Local production-like | `docker/prod/docker-compose.yaml` | Functional networks, reduced host exposure, no Docker socket in Airflow, isolated `docker/prod/runtime` workspaces, an internal typed ML step runner API, internal ML step services, and manifest-first API serving. |

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
| `pipeline_runtime_net` | Airflow worker, API, job runner API, ML step services, Pushgateway | Runtime handoff between orchestration, typed job execution, API refresh, and batch metrics. |
| `tracking_client_net` | `ml-models-prod`, MLflow server | MLflow client calls from model workloads. |
| `tracking_backend_net` | MLflow server, MLflow PostgreSQL, MinIO, MC init | Private tracking metadata and artifact backend. |
| `observability_net` | Prometheus, Grafana, Alertmanager, cAdvisor, Pushgateway, API metrics | Scrapes, dashboards, and local alerts. |
| `dev_support_net` | MailHog, Alertmanager, Airflow email clients | Local support services. |
| Runtime mounts | `docker/prod/runtime/data`, `docker/prod/runtime/models`, `docker/prod/runtime/logs`, `docker/prod/runtime/artifacts` | Production-like generated data, models, logs, payloads, and promoted artifact manifests. |

## Current service-to-service paths

| Source | Target | Runtime | Mechanism | Reason |
| ------ | ------ | ------- | --------- | ------ |
| Airflow services | Airflow PostgreSQL | Dev and prod-like | Compose DNS, PostgreSQL | Metadata database and result backend. |
| Airflow services | Airflow Redis | Dev and prod-like | Compose DNS, Redis | Celery broker. |
| Airflow services | Airflow API server | Prod-like | Compose DNS, HTTP | Internal Airflow execution API. |
| Airflow DAG tasks | ML job containers | Dev only | Docker socket / DockerOperator | Current local development ML execution. |
| Airflow DAG tasks | `job-runner-api` | Prod-like | HTTP on internal network | Submit typed ML step jobs and read runner status. |
| Airflow DAG tasks | API | Dev and prod-like | HTTP on internal network | Refresh API after successful DAG runs. |
| `job-runner-api` | `ml-ingest-prod` | Prod-like | HTTP on `pipeline_runtime_net` | Execute one validated ingestion job request. |
| `job-runner-api` | `ml-features-prod` | Prod-like | HTTP on `pipeline_runtime_net` | Execute one validated feature job request. |
| `job-runner-api` | `ml-models-prod` | Prod-like | HTTP on `pipeline_runtime_net` | Execute one validated model job request. |
| ML step services | Runtime artifact manifests | Dev and prod-like | Filesystem mount | Write and promote artifact manifests under the configured manifest root. |
| API | Promoted prediction manifests | Dev and prod-like | Filesystem mount | Read `predictions/<counter_id>/current.json` and verify local payload evidence. |
| API | Prediction payload CSV | Dev and prod-like | Filesystem mount | Load the manifest-referenced local prediction payload. |
| ML step services | Pushgateway | Dev and prod-like | HTTP | Batch metric push when enabled. |
| `ml-models-prod` | MLflow server | Prod-like | HTTP | Run, parameter, metric, model, and artifact logging. |
| Development model jobs | MLflow server | Dev | HTTP | Run, parameter, metric, model, and artifact logging. |
| MLflow server | MLflow PostgreSQL | Dev and prod-like | PostgreSQL | MLflow backend store. |
| MLflow server | MinIO | Dev and prod-like | S3/HTTP | MLflow artifact store. |
| Prometheus | API, Pushgateway, cAdvisor | Dev and prod-like | HTTP scrape | Metrics collection. |
| Grafana | Prometheus | Dev and prod-like | HTTP | Provisioned datasource. |
| Alertmanager | MailHog | Dev and prod-like | SMTP | Local alert capture. |
| `job-runner-api` | In-memory state | Prod-like | Process memory | Local job status persistence for typed step attempts. |

`job-runner-api` exposes an internal HTTP API on `pipeline_runtime_net`. The
current implementation accepts one typed ML step request, records status in
memory, delegates execution synchronously to the matching internal ML step
service, and returns structured result or error evidence.

## Manifest handoff path

The implemented production-like handoff is explicit:

```text
Airflow DAG task
  -> job-runner-api
  -> ml-ingest-prod / ml-features-prod / ml-models-prod
  -> docker/prod/runtime/artifacts/manifests/<artifact_type>/<counter_id>/current.json
  -> authenticated API /admin/refresh
  -> API serving from the manifest-referenced local prediction payload
```

The API does not scan `data/final` for the newest `y_full.csv`. It reads the
promoted prediction manifest, checks that the storage backend is local, resolves
`storage.local_path` from the configured repository root, verifies the checksum
when available, and then loads the referenced CSV.

## Runner execution boundary

The active runner contract is intentionally narrow:

- accepted job types are `ingest`, `features`, and `models`;
- active request and status contracts live under `src/ml/jobs`;
- concrete prod-like execution happens through internal FastAPI ML step services;
- a full-pipeline runtime job is not exposed by the runner API;
- Airflow remains responsible for ordering ingest, features, and model tasks;
- the runner does not use Docker socket access, Docker SDK, Kubernetes, or a
  distributed worker queue.

This boundary keeps step execution observable without making the runner a second
pipeline orchestrator. Future durable queues or remote job backends must preserve
this typed step-level API boundary unless a new story changes the architecture.

## Host exposure summary

Development intentionally exposes more local UIs. The production-like runtime
publishes only operator-facing services by default.

| Service family | Development exposure | Production-like exposure |
| -------------- | -------------------- | ------------------------ |
| FastAPI prediction API | Host exposed | Host exposed. |
| Job runner API | Not present | Internal-only. |
| ML step services | One-off job containers | Internal-only FastAPI services. |
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
| `docker/dev/runtime/artifacts` | Dev | Development manifest-first handoff root. |
| `docker/prod/runtime/data` | Prod-like | Generated production-like data owned by ML step services and read by the API through manifest local paths. |
| `docker/prod/runtime/models` | Prod-like | Generated production-like model artifacts owned by `ml-models-prod`. |
| `docker/prod/runtime/logs` | Prod-like | Production-like service, batch, and runner API logs. |
| `docker/prod/runtime/artifacts` | Prod-like | Manifest-first handoff root with run-scoped manifests and promoted `current.json` files. |
| Root raw CSV | Prod-like | Read-only business source input for `ml-ingest-prod`. |

`job-runner-api` currently mounts only `docker/prod/runtime/logs/job-runner` in
Compose. It delegates concrete execution to internal ML step services, which own
data, model, log, artifact, and optional tracking access. Any future runner mount
widening must be documented with the runtime impact.

## Operational guardrails

- Do not add Docker socket mounts to production-like Airflow services.
- Do not copy the broad `mlops_net` development model into `docker/prod`.
- Do not reintroduce implicit latest-file or folder scanning in API serving.
- Prefer explicit functional networks over pairwise networks unless sensitive
  state or privileged control surfaces require isolation.
- Keep host exposure in `../current-runtime-and-operations/ports-and-services.md`
  synchronized with Compose.
- Keep typed ML job contracts in `src/ml/jobs` independent from FastAPI, Airflow,
  Docker, and concrete runner implementation code.
- Keep `job-runner-api` free from broad data, model, artifact, and tracking
  mounts unless a future story documents the boundary change.
