# Runtime communication matrix

This document describes the current local Docker Compose communication model for
Phase 6. It is a review artifact for the local MLOps runtime, not a production
security model.

`docker-compose.yaml` is the source of truth for the runtime state. The diagrams
in `references/Architecture_MLOps.drawio.png`,
`references/Docker_Compose_Overview.drawio.png`, and
`references/Docker_Compose_Monitoring.drawio.png` remain useful communication
artifacts for the intended architecture and network responsibilities, but they
may lag behind recent Compose changes.

Host port ranges and local URLs are documented in
[`docs/ports-and-services.md`](ports-and-services.md). This document focuses on
service-to-service communication, implicit coupling through shared mounts, and
local-development boundaries that should be reviewed before a production-like
split.

## Network responsibilities

| Network | Responsibility | Production-like concern |
| ------- | -------------- | ----------------------- |
| `airflow_net` | Airflow control plane: API server, scheduler, DAG processor, triggerer, worker, Flower, Airflow PostgreSQL, Redis, and Airflow-managed tasks that need DAG-orchestration context. | Should stay limited to orchestration and Airflow dependencies. Business API access from this network should remain justified. |
| `mlflow_net` | MLflow tracking and artifact plane: MLflow server, MLflow PostgreSQL, MinIO, MinIO initialization helper, and ML jobs that log runs or artifacts. | Should remain the default private boundary for tracking metadata and artifacts. |
| `mlops_net` | Local platform integration plane: FastAPI, monitoring, Pushgateway, MailHog, cAdvisor, and services that need cross-stack integration. | Broad shared network. Useful locally, but a future production design should split monitoring, serving, and orchestration boundaries more strictly. |

## Multiple-network attachments

| Service | Networks | Current justification | Cleanup candidate? |
| ------- | -------- | --------------------- | ------------------ |
| `ml-ingest-dev` | `airflow_net`, `mlops_net` | Can be triggered by Airflow and can push metrics to Pushgateway on `mlops_net`. | Keep for now while Airflow launches jobs through the local Docker socket and metrics target `monitoring-pushgateway`. |
| `ml-features-dev` | `airflow_net`, `mlops_net` | Same batch orchestration and metrics-push pattern as ingestion. | Keep for now. |
| `ml-models-dev` | `mlflow_net`, `airflow_net`, `mlops_net` | Needs MLflow/MinIO access, Airflow orchestration context, and Pushgateway metrics. | Keep for now; this is the clearest intentional cross-plane service. |
| `api-dev` | `airflow_net`, `mlops_net` | Exposes FastAPI prediction and admin refresh endpoints; Airflow refresh currently targets the API after successful DAG runs. | Candidate for future cleanup if Airflow can reach the API through `mlops_net` only. Direct `airflow_net` access is not obviously required by the current Compose file. |
| Airflow common services | `airflow_net`, `mlops_net` | Airflow services inherit `mlops_net` to reach platform services such as API, Pushgateway, MLflow, MinIO, or MailHog depending on runtime tasks. | Candidate for narrower per-service attachments later. Do not change blindly because DAG behavior depends on local DockerOperator and HTTP refresh flows. |
| `airflow-worker` | `airflow_net`, `mlops_net` | Executes Celery tasks and runs local DockerOperator workloads through `/var/run/docker.sock`. | Keep for local development only; Docker socket access is privileged and not production-like. |
| `airflow-flower` | `airflow_net` | Celery monitoring only; no cross-stack dependency documented. | No cleanup needed. |
| `mlflow-minio` | `mlflow_net`, `mlops_net` | Private artifact store for MLflow, also host-exposed for local debugging and host-side clients. `mlops_net` attachment is not required for MLflow server access because both services share `mlflow_net`. | Candidate for future cleanup; prefer `mlflow_net` plus explicit host exposure unless a concrete `mlops_net` caller is retained. |
| `mlflow-server` | `mlflow_net`, `mlops_net` | Tracks ML runs and exposes UI/API to host; `mlops_net` allows cross-stack calls from ML/API/monitoring services. | Candidate for future cleanup if all MLflow clients stay on `mlflow_net` or use host exposure deliberately. |

## Host-exposed versus internal-only services

| Service | Internal port | Host port variable | Exposed to host? | Dev-only? | Reason |
| ------- | ------------- | ------------------ | ---------------- | --------- | ------ |
| `api-dev` | `10000` | `API_HOST_PORT` | Yes | No, local serving surface | FastAPI prediction API, OpenAPI docs, and admin refresh endpoint. |
| `airflow-api-server` | `8080` | `AIRFLOW_HOST_PORT` | Yes | Local operations | Airflow UI and API for DAG validation and manual runs. |
| `airflow-flower` | `5555` | `AIRFLOW_FLOWER_HOST_PORT` | Yes | Yes | Celery worker monitoring. |
| `mlflow-minio` | `9000` | `MINIO_API_HOST_PORT` | Yes | Local artifact debugging | S3-compatible artifact endpoint for host-side clients. |
| `mlflow-minio` | `9001` | `MINIO_CONSOLE_HOST_PORT` | Yes | Yes | MinIO browser console. |
| `mlflow-server` | `5000` | `MLFLOW_HOST_PORT` | Yes | Local tracking UI/API | MLflow tracking, registry, and run inspection. |
| `monitoring-prometheus` | `9090` | `PROMETHEUS_HOST_PORT` | Yes | Local observability | Prometheus UI and queries. |
| `monitoring-pushgateway` | `9091` | `PUSHGATEWAY_HOST_PORT` | Yes | Local observability | Batch metrics debugging. |
| `monitoring-alertmanager` | `9093` | `ALERTMANAGER_HOST_PORT` | Yes | Local observability | Alert routing and debug UI. |
| `monitoring-cadvisor` | `8080` | `CADVISOR_HOST_PORT` | Yes | Local observability | Container metrics inspection. |
| `monitoring-grafana` | `3000` | `GRAFANA_HOST_PORT` | Yes | Local observability | Dashboards. |
| `monitoring-mailhog` | `1025` | `MAILHOG_SMTP_HOST_PORT` | Yes | Yes | Local SMTP capture endpoint. |
| `monitoring-mailhog` | `8025` | `MAILHOG_UI_HOST_PORT` | Yes | Yes | MailHog web UI. |
| `airflow-postgres` | `5432` | None | No | No | Airflow metadata database. |
| `airflow-redis` | `6379` | None | No | No | Airflow Celery broker. |
| `mlflow-postgres` | `5432` | None | No | No | MLflow metadata database. |
| `mlflow-mc-init` | None | None | No | Yes | One-off MinIO bucket initialization helper. |
| `airflow-init` | None | None | No | Yes | One-off Airflow initialization helper. |
| `ml-ingest-dev` | None | None | No | Batch job | One-off or Airflow-triggered ingestion container. |
| `ml-features-dev` | None | None | No | Batch job | One-off or Airflow-triggered feature container. |
| `ml-models-dev` | None | None | No | Batch job | One-off or Airflow-triggered training and prediction container. |

## Communication matrix

| Source service | Target service | Network | Internal port | Host-exposed port | Protocol | Authentication or credential boundary | Reason | Exposed to host? | Dev-only? | Production-like concern |
| -------------- | -------------- | ------- | ------------- | ----------------- | -------- | ------------------------------------- | ------ | ---------------- | --------- | ----------------------- |
| Host browser or API client | `api-dev` | Host port mapping | `10000` | `${API_HOST_PORT:-10000}` | HTTP | FastAPI application auth; local `.env` user defaults | Prediction API, OpenAPI docs, admin refresh validation | Yes | Local access path | Replace default credentials and define ingress/auth boundaries for production. |
| Host browser or operator | `airflow-api-server` | Host port mapping | `8080` | `${AIRFLOW_HOST_PORT:-12080}` | HTTP | Airflow simple auth manager; local admin defaults | Airflow UI/API for DAG operations | Yes | Local operations | Simple auth is not a production-grade identity boundary. |
| Host browser or operator | `airflow-flower` | Host port mapping | `5555` | `${AIRFLOW_FLOWER_HOST_PORT:-12555}` | HTTP | No explicit Compose auth boundary | Celery worker monitoring | Yes | Yes | Should not be exposed without auth in production-like runtime. |
| Host browser or ML client | `mlflow-server` | Host port mapping | `5000` | `${MLFLOW_HOST_PORT:-13001}` | HTTP | MLflow allowed-hosts middleware; no Compose auth | MLflow tracking UI and host-side clients | Yes | Local tracking | Add identity and network restrictions before production exposure. |
| Host S3 client or browser | `mlflow-minio` | Host port mapping | `9000` / `9001` | `${MINIO_API_HOST_PORT:-13000}` / `${MINIO_CONSOLE_HOST_PORT:-13002}` | HTTP/S3 | MinIO root credentials from `.env` | Artifact store API and console debugging | Yes | Local artifact debugging | Root credentials and console exposure must be hardened or removed. |
| Host browser or operator | `monitoring-prometheus` | Host port mapping | `9090` | `${PROMETHEUS_HOST_PORT:-14090}` | HTTP | No explicit Compose auth boundary | Prometheus UI and query debugging | Yes | Local observability | Restrict UI and admin APIs in production-like runtime. |
| Host browser or operator | `monitoring-grafana` | Host port mapping | `3000` | `${GRAFANA_HOST_PORT:-14300}` | HTTP | Grafana admin credentials from `.env` | Dashboards | Yes | Local observability | Rotate credentials and define SSO/role model later. |
| Host browser or operator | `monitoring-cadvisor` | Host port mapping | `8080` | `${CADVISOR_HOST_PORT:-14200}` | HTTP | No explicit Compose auth boundary | Container metrics debugging | Yes | Local observability | Exposes runtime metadata; should not be public. |
| Host browser or operator | `monitoring-pushgateway` | Host port mapping | `9091` | `${PUSHGATEWAY_HOST_PORT:-14091}` | HTTP | No explicit Compose auth boundary | Inspect pushed batch metrics | Yes | Local observability | Restrict writes and lifecycle management in production-like runtime. |
| Host browser or operator | `monitoring-alertmanager` | Host port mapping | `9093` | `${ALERTMANAGER_HOST_PORT:-14093}` | HTTP | No explicit Compose auth boundary | Alert routing inspection | Yes | Local observability | Restrict UI and API access. |
| Host SMTP client or browser | `monitoring-mailhog` | Host port mapping | `1025` / `8025` | `${MAILHOG_SMTP_HOST_PORT:-15025}` / `${MAILHOG_UI_HOST_PORT:-15080}` | SMTP/HTTP | No real mail delivery; local capture only | Validate email routing locally | Yes | Yes | Replace with real SMTP provider and secrets in production. |
| `airflow-api-server`, `airflow-scheduler`, `airflow-dag-processor`, `airflow-triggerer`, `airflow-worker` | `airflow-postgres` | `airflow_net` | `5432` | None | PostgreSQL | `AIRFLOW_POSTGRES_*` credentials | Airflow metadata database and Celery result backend | No | No | Keep private; rotate credentials and use managed DB later if needed. |
| `airflow-scheduler`, `airflow-worker`, `airflow-flower` | `airflow-redis` | `airflow_net` | `6379` | None | Redis | No Redis password in current broker URL | Celery broker | No | No | Add broker auth or managed broker for production-like runtime. |
| `airflow-api-server`, `airflow-scheduler`, `airflow-dag-processor`, `airflow-triggerer`, `airflow-worker` | `airflow-api-server` | `airflow_net` | `8080` | `${AIRFLOW_HOST_PORT:-12080}` | HTTP | `AIRFLOW_API_AUTH_JWT_SECRET` for Airflow execution API | Internal Airflow execution API and UI/API service | Partly | No | Keep internal execution API off public networks. |
| `airflow-worker` | Docker daemon on host | Bind mount, not Docker network | `/var/run/docker.sock` | None | Unix socket / Docker API | Host Docker group/root-equivalent boundary | Local DockerOperator workload creation for ML jobs | No | Yes | Privileged local-development boundary; not a production orchestration model. |
| Airflow DAG tasks | `ml-ingest-dev` container | Docker daemon then `mlops_net` / `airflow_net` | N/A | None | Docker API | Docker socket and container env variables | Trigger ingestion as part of init/daily DAGs | No | Local orchestration | Prefer worker queue, KubernetesPodOperator, or managed job runtime later. |
| Airflow DAG tasks | `ml-features-dev` container | Docker daemon then `mlops_net` / `airflow_net` | N/A | None | Docker API | Docker socket and container env variables | Trigger feature engineering after ingestion | No | Local orchestration | Same privileged Docker socket concern. |
| Airflow DAG tasks | `ml-models-dev` container | Docker daemon then `mlflow_net` / `mlops_net` / `airflow_net` | N/A | None | Docker API | Docker socket, MLflow/MinIO credentials, env variables | Trigger training and prediction after features | No | Local orchestration | Same privileged Docker socket concern. |
| Airflow DAG tasks | `api-dev` | `mlops_net` or `airflow_net` | `10000` | `${API_HOST_PORT:-10000}` | HTTP | API Basic Auth / admin role | `POST /admin/refresh` after successful init or daily run | Yes | No | `api-dev` probably does not need `airflow_net`; validate before cleanup. |
| `api-dev` | Shared `./data` mount | Volume bind mount | N/A | N/A | Filesystem | Host filesystem permissions | Read final prediction data from `DATA_FINAL_ROOT` | No | Local runtime | Replace shared mutable filesystem with artifact contract or object storage. |
| `ml-ingest-dev` | Shared `./data`, `./logs` mounts | Volume bind mount | N/A | N/A | Filesystem | Host filesystem permissions | Read raw data, write interim data and logs | No | Batch job | Shared ownership can hide data contracts and race conditions. |
| `ml-features-dev` | Shared `./data`, `./logs` mounts | Volume bind mount | N/A | N/A | Filesystem | Host filesystem permissions | Read interim data, write processed features and logs | No | Batch job | Same shared mutable filesystem concern. |
| `ml-models-dev` | Shared `./data`, `./models`, `./logs` mounts | Volume bind mount | N/A | N/A | Filesystem | Host filesystem permissions | Read processed data, write predictions, model artifacts, and logs | No | Batch job | Promote explicit artifact storage and ownership boundaries later. |
| `ml-models-dev` | `mlflow-server` | `mlflow_net` | `5000` | `${MLFLOW_HOST_PORT:-13001}` | HTTP | `MLFLOW_TRACKING_URI`; no explicit MLflow auth in Compose mode | Log runs, params, metrics, models, and artifacts | Partly | No | Add MLflow auth/registry governance for production-like runtime. |
| `ml-models-dev` | `mlflow-minio` | `mlflow_net` | `9000` | `${MINIO_API_HOST_PORT:-13000}` | HTTP/S3 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL` | Store MLflow artifacts | Partly | No | Avoid root credentials and restrict bucket policies. |
| `mlflow-server` | `mlflow-postgres` | `mlflow_net` | `5432` | None | PostgreSQL | `MLFLOW_POSTGRES_*` credentials | MLflow backend store | No | No | Keep private; use managed DB or least-privilege credentials later. |
| `mlflow-server` | `mlflow-minio` | `mlflow_net` | `9000` | `${MINIO_API_HOST_PORT:-13000}` | HTTP/S3 | MinIO credentials from `.env` | MLflow artifact storage backend | Partly | No | Keep artifact store private except deliberate local host exposure. |
| `mlflow-mc-init` | `mlflow-minio` | `mlflow_net` | `9000` | `${MINIO_API_HOST_PORT:-13000}` | HTTP/S3 | MinIO root credentials | Create and configure the `mlflow` bucket | Partly | Yes | One-off init with elevated credentials should not be long-lived. |
| `ml-ingest-dev`, `ml-features-dev`, `ml-models-dev` | `monitoring-pushgateway` | `mlops_net` | `9091` | `${PUSHGATEWAY_HOST_PORT:-14091}` | HTTP | Controlled by `DISABLE_METRICS_PUSH`; no auth | Push batch metrics when enabled | Yes | Local metrics path | Add write controls and job lifecycle cleanup later. |
| `api-dev` | `monitoring-prometheus` scrape target | `mlops_net` | `10000` | `${API_HOST_PORT:-10000}` | HTTP `/metrics` | No scrape auth | API technical and business metrics | Yes | No | Decide whether metrics stay on serving network or separate monitoring network. |
| `monitoring-prometheus` | `api-dev` | `mlops_net` | `10000` | `${API_HOST_PORT:-10000}` | HTTP `/metrics` | No scrape auth | Scrape API metrics | Yes | No | Same monitoring-network separation concern. |
| `monitoring-prometheus` | `monitoring-cadvisor` | `mlops_net` | `8080` | `${CADVISOR_HOST_PORT:-14200}` | HTTP | No scrape auth | Scrape host/container metrics | Yes | Local observability | cAdvisor is privileged and mounts host runtime paths. |
| `monitoring-prometheus` | `monitoring-pushgateway` | `mlops_net` | `9091` | `${PUSHGATEWAY_HOST_PORT:-14091}` | HTTP | No scrape auth | Scrape pushed batch metrics | Yes | Local observability | Review metric retention and stale series cleanup later. |
| `monitoring-prometheus` | `monitoring-alertmanager` | `mlops_net` | `9093` | `${ALERTMANAGER_HOST_PORT:-14093}` | HTTP | No explicit Compose auth | Send firing alerts to Alertmanager | Yes | Local observability | Restrict alert APIs and notification routes. |
| `monitoring-grafana` | `monitoring-prometheus` | `mlops_net` | `9090` | `${PROMETHEUS_HOST_PORT:-14090}` | HTTP | Grafana datasource config; no Prometheus auth | Query metrics for dashboards | Yes | Local observability | Add datasource secrets and network isolation later. |
| `monitoring-alertmanager` | `monitoring-mailhog` | `mlops_net` | `1025` | `${MAILHOG_SMTP_HOST_PORT:-15025}` | SMTP | MailHog capture only; TLS disabled | Validate alert emails locally | Yes | Yes | Replace with authenticated SMTP and TLS in production. |
| Airflow services | `monitoring-mailhog` | `mlops_net` | `1025` | `${MAILHOG_SMTP_HOST_PORT:-15025}` | SMTP | MailHog capture only; TLS disabled | Capture Airflow emails locally | Yes | Yes | Replace with authenticated SMTP and TLS in production. |
| `monitoring-cadvisor` | Host Docker/runtime paths | Bind mounts | N/A | N/A | Filesystem / Docker socket | Privileged container and host mounts | Collect container and host metrics | Yes | Local observability | Privileged host access must be redesigned for production. |

## Shared mounts and implicit ownership coupling

| Mount | Used by | Coupling created | Production-like concern |
| ----- | ------- | ---------------- | ----------------------- |
| `./data:/app/data` | ML jobs, `api-dev` | ML jobs write datasets and predictions that the API reads directly. | Define a versioned data/artifact contract before separating runtime roles. |
| `./data:/opt/airflow/data` | Airflow services | DAGs and Airflow tasks can inspect or coordinate data files. | Avoid implicit orchestration decisions based on mutable shared files where possible. |
| `./logs:/app/logs` | ML jobs, `api-dev` | Runtime logs are shared with the host and possibly other containers. | Centralize logs through logging infrastructure later. |
| `./logs:/opt/airflow/logs` | Airflow services | Airflow logs are host-visible and shared by Airflow components. | Expected locally; production should use remote log storage. |
| `./models:/app/models` | `ml-models-dev` | Model files are written to a host-mounted directory. | Prefer MLflow registry/artifact storage as the primary model handoff. |
| `./docker/dev/airflow/dags:/opt/airflow/dags` | Airflow services | DAG code is live-mounted from the repository. | Good for local development; production should deploy immutable DAG artifacts. |
| `./docker/dev/airflow/config/bike_dag_config.json:/opt/airflow/config/bike_dag_config.json:ro` | Airflow services | Business DAG configuration is read directly from the repository. | Keep configuration versioned, but separate environment-specific values later. |
| `./docker/dev/airflow/config/:/opt/airflow/config/` | `airflow-init` | Airflow variables and connections are imported from repository files. | Do not keep real secrets in repository-managed config files. |
| `/var/run/docker.sock:/var/run/docker.sock` | `airflow-worker`, `monitoring-cadvisor` | Gives privileged access to host Docker runtime. | Local-development boundary only; not production-like. |
| `./docker/dev/prometheus:/etc/prometheus:ro` | `monitoring-prometheus` | Prometheus scrape and alert rules are repository-mounted. | Expected locally; production should deploy validated config artifacts. |
| `./docker/dev/grafana/provisioning:/etc/grafana/provisioning:ro` | `monitoring-grafana` | Grafana datasources and dashboards are repository-provisioned. | Expected locally; secure datasource secrets later. |
| `./docker/dev/grafana/dashboards:/var/lib/grafana/dashboards` | `monitoring-grafana` | Dashboard JSON files are live-mounted. | Useful locally; package dashboards as immutable artifacts later. |
| `./docker/dev/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro` | `monitoring-alertmanager` | Alert routing is repository-mounted. | Replace MailHog test routes and secure notification secrets later. |
| `mlflow-postgres-db` | `mlflow-postgres` | Persistent MLflow metadata. | Private service volume; backup and migration strategy required later. |
| `mlflow-minio-data` | `mlflow-minio` | Persistent MLflow artifacts. | Private artifact store; backup, lifecycle, and credential rotation needed later. |
| `airflow-postgres-db` | `airflow-postgres` | Persistent Airflow metadata. | Private service volume; backup and migration strategy required later. |
| `airflow-redis-data` | `airflow-redis` | Persistent Redis broker data. | Review persistence needs and broker durability later. |
| `monitoring-prometheus-data` | `monitoring-prometheus` | Persistent metrics time-series data. | Retention and storage planning required for production-like monitoring. |
| `monitoring-grafana-data` | `monitoring-grafana` | Persistent Grafana state. | Backup and provisioning strategy required later. |
| `monitoring-alertmanager-data` | `monitoring-alertmanager` | Persistent alertmanager state. | Review silences and notification state management. |

## Explicit cleanup candidates for later stories

- Validate whether `api-dev` needs `airflow_net`. Current evidence suggests Airflow
  only needs HTTP access to the API for `/admin/refresh`, which can likely stay on
  `mlops_net`.
- Validate whether `mlflow-server` and `mlflow-minio` need `mlops_net`. Their core
  communication is already covered by `mlflow_net`; host exposure covers local UI
  and host-side clients.
- Split Airflow services more narrowly if DAG behavior allows it. The inherited
  `x-airflow-common` network list is convenient locally but broad.
- Replace Airflow worker Docker socket usage with a production-like job runtime
  such as a controlled worker image, a remote Docker boundary, or Kubernetes
  orchestration.
- Move secrets out of default local files and into a dedicated secret-management
  boundary as part of the runtime identity and permissions story.

## Validation

No functional runtime change is introduced by this document. The expected Compose
validation command remains:

```bash
make compose-config
```
