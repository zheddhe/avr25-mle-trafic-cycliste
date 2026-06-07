# Runtime communication matrix

This document describes local Docker Compose communication after Phases 6 and 7,
with Phase 8 artifact and job contracts now partially implemented. It is a
runtime architecture reference, not a production security model.

The current runtime has two Compose views:

| Runtime | Compose file | Communication model |
| ------- | ------------ | ------------------- |
| Development | `docker/dev/docker-compose.yaml` | Broad local integration, DockerOperator execution, root `data/logs/models` mounts. |
| Local production-like | `docker/prod/docker-compose.yaml` | Functional networks, reduced host exposure, no Docker socket in Airflow, isolated `docker/prod/runtime` workspaces. |

Host port ranges and local URLs are documented in
[`../current-runtime-and-operations/ports-and-services.md`](../current-runtime-and-operations/ports-and-services.md).
Runtime ownership and remaining exceptions are documented in
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
| `pipeline_runtime_net` | Airflow worker, API, ML jobs, Pushgateway, future runner API | Runtime handoff between orchestration, business jobs, refresh, and batch metrics. |
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
| Airflow DAG tasks | ML job containers | Dev only | Docker socket / DockerOperator | Current local development ML execution. |
| Airflow DAG tasks | `api-dev` | Dev and prod-like | HTTP on internal network | Refresh API after successful DAG runs. |
| ML jobs | Pushgateway | Dev and prod-like | HTTP | Batch metric push when enabled. |
| ML model jobs | MLflow server | Dev and prod-like | HTTP | Run, parameter, metric, model, and artifact logging. |
| MLflow server | MLflow PostgreSQL | Dev and prod-like | PostgreSQL | MLflow backend store. |
| MLflow server | MinIO | Dev and prod-like | S3/HTTP | MLflow artifact store. |
| Prometheus | API, Pushgateway, cAdvisor | Dev and prod-like | HTTP scrape | Metrics collection. |
| Grafana | Prometheus | Dev and prod-like | HTTP | Provisioned datasource. |
| Alertmanager | MailHog | Dev and prod-like | SMTP | Local alert capture. |
| API | Prediction artifacts | Dev and prod-like | Filesystem | Current prediction serving input. Phase 8 moves this to manifest-aware serving. |

## Host exposure summary

Development intentionally exposes more local UIs. The production-like runtime
publishes only operator-facing services by default.

| Service family | Development exposure | Production-like exposure |
| -------------- | -------------------- | ------------------------ |
| FastAPI | Host exposed | Host exposed |
| Airflow API/UI | Host exposed | Host exposed |
| MLflow UI/API | Host exposed | Host exposed |
| Grafana | Host exposed | Host exposed |
| MinIO API/console | Host exposed | Internal-only |
| Prometheus | Host exposed | Internal-only |
| Pushgateway | Host exposed | Internal-only |
| Alertmanager | Host exposed | Internal-only |
| cAdvisor | Host exposed | Internal-only |
| MailHog | Host exposed | Internal-only |
| PostgreSQL and Redis | Internal-only | Internal-only |

See [`../current-runtime-and-operations/ports-and-services.md`](../current-runtime-and-operations/ports-and-services.md)
for exact ports and URLs.

## Shared mount coupling

| Mount | Runtime | Current role | Phase 8 direction |
| ----- | ------- | ------------ | ----------------- |
| Root `data` | Dev | Raw, interim, processed, final data for DVC/local workflows. | Remains dev/DVC-owned. |
| Root `models` | Dev | Development model artifacts. | Remains dev/DVC-owned. |
| Root `logs` | Dev | Development logs. | Remains dev-owned. |
| `docker/prod/runtime/data` | Prod-like | Generated production-like data. | Referenced by artifact manifests. |
| `docker/prod/runtime/models` | Prod-like | Generated production-like model artifacts. | Referenced by artifact manifests. |
| `docker/prod/runtime/logs` | Prod-like | Production-like service and job logs. | Correlate with runner `job_id` and `run_id`. |
| `docker/prod/runtime/artifacts` | Prod-like | Manifest-first handoff root. | Owns promoted `current.json` and run-scoped manifests. |
| Root raw CSV | Prod-like | Read-only business source input. | Remains the only required root/DVC input. |

## Phase 8 coverage and status

Phase 8 introduces explicit contracts instead of adding broad communication paths.
The remaining plan is tracked centrally in
[`../next-phase-design/artifact-handoff-strategy.md`](../next-phase-design/artifact-handoff-strategy.md).

| Addition | Status | Expected communication | Reference |
| -------- | ------ | ---------------------- | --------- |
| Artifact manifest schemas | Implemented | Shared Python contract used by ML, runner, and API integrations. | [`../next-phase-design/artifact-manifest-models.md`](../next-phase-design/artifact-manifest-models.md) |
| Artifact manifest store | Implemented | Local helpers write run manifests and replace `current.json`. | [`../next-phase-design/artifact-manifest-store.md`](../next-phase-design/artifact-manifest-store.md) |
| ML manifest emission | Implemented for local manifests | Ingest, features, and model jobs can emit promoted local manifests under `docker/prod/runtime/artifacts`. | #66 |
| Job contracts | Implemented | Airflow, runner, and typed workers exchange Pydantic job payloads and statuses. | [`../next-phase-design/airflow-job-runner-strategy.md`](../next-phase-design/airflow-job-runner-strategy.md) |
| `job-runner-api` | Remaining | Airflow submits jobs through `pipeline_runtime_net`; service is internal-only. | #68 |
| Runner execution | Remaining | Runner executes typed jobs and returns manifest references. | #69 |
| Prod Airflow DAG | Remaining | Airflow observes runner job states instead of creating containers. | #70 |
| Artifact-aware API | Remaining | API serves the promoted artifact described by `current.json`. | #71 |
| Smoke validation | Remaining | Automated checks verify runner, Airflow, artifact, API, and monitoring connectivity. | #72 |

The current implementation therefore covers the artifact and contract foundation,
but not yet the end-to-end production-like execution path. `docker/prod` is
safer because Airflow has no Docker socket mount, but the replacement runner path
still has to be implemented before it is fully executable through Airflow.

## Known Phase 8 validation debt

Phase 8 still needs a dedicated validation story for realistic, production-like
coverage. The minimum debt to track is:

- broader unit coverage for manifest errors, runner states, API artifact loading,
  and configuration failures;
- integration fixtures derived from the real raw-data schema instead of only
  synthetic minimal examples;
- at least one smoke-sized dataset that exercises ingest, features, model,
  manifest promotion, API load, and monitoring checks;
- explicit documentation of which fixture size is safe for CI and which one is
  intended for local production-like validation.

## Rules for future changes

- Do not add Docker socket mounts to production-like Airflow services.
- Do not copy the broad `mlops_net` development model into `docker/prod`.
- Prefer explicit functional networks over pairwise networks unless sensitive
  state or privileged control surfaces require isolation.
- Keep host exposure in `../current-runtime-and-operations/ports-and-services.md`
  synchronized with Compose.
- Keep artifact handoff changes aligned with
  [`../next-phase-design/artifact-handoff-strategy.md`](../next-phase-design/artifact-handoff-strategy.md).
