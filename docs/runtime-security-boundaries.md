# Runtime security boundaries and identities

This document defines the current and target security boundaries for the local
Docker Compose runtime. It is a Phase 6 design artifact for issue #48.

It intentionally does not implement production hardening. The current
development runtime stays unchanged. Follow-up stories can use this document to
design a stricter local production-like runtime.

## Scope and source documents

The current communication model is documented in
[`docs/runtime-communication-matrix.md`](runtime-communication-matrix.md).

Host port ranges and local URLs are documented in
[`docs/ports-and-services.md`](ports-and-services.md).

Communication paths justify permissions, not the opposite. A container, network,
credential, volume, or elevated runtime capability should exist because a
documented service dependency needs it. Historical convenience or local debug
shortcuts should be marked as local-development exceptions.

Audited runtime sources:

| Source | Security information audited |
| ------ | ---------------------------- |
| `docker-compose.yaml` | Services, networks, ports, users, mounts, profiles, healthchecks, and credentials consumed by Compose. |
| `.env.template` | Local developer values, placeholders, image variables, host ports, MLflow, MinIO, Airflow, Grafana, API demo, and DVC/DagsHub credentials. |
| `docker/dev/prometheus/prometheus.yml` | Prometheus scrape targets and Alertmanager target. |
| `docker/dev/grafana/provisioning/datasources/datasource.yaml` | Grafana datasource dependency on Prometheus. |
| `docker/dev/airflow/config/connections.json` | Airflow `api_dev` HTTP connection to the FastAPI service. |
| `docker/dev/airflow/config/variables.json` | Airflow DockerOperator images, network, MLflow, MinIO, Pushgateway, UID/GID, and repository path variables. |
| `docker/dev/airflow/config/bike_dag_config.json` | Multi-counter DAG business configuration. |
| `docker/dev/airflow/scripts/airflow-init.sh` | Import path for Airflow variables, connections, and pool setup. |
| `README.md` | Operator-facing runtime, Airflow, MLflow, and monitoring workflow documentation. |

## Current security model summary

The current runtime is a local development topology. It optimizes visibility,
debuggability, host access, and fast iteration.

Current boundaries:

- `airflow_net` groups the Airflow control plane, metadata database, Redis broker,
  and Airflow-managed tasks.
- `mlflow_net` groups MLflow tracking metadata, MinIO artifacts, MLflow server,
  and ML workloads that log runs or artifacts through MLflow.
- `mlops_net` is a broad local integration network for API, monitoring,
  Pushgateway, MailHog, cAdvisor, and cross-stack service discovery.
- Host ports expose the API, Airflow UI, Flower, MLflow, MinIO, Prometheus,
  Pushgateway, Alertmanager, cAdvisor, Grafana, and MailHog for local operations.
- Shared host mounts expose `data`, `logs`, `models`, Airflow DAGs, Airflow
  config, Prometheus config, Grafana config, Alertmanager config, and selected
  Docker runtime paths.
- The custom API and ML containers are good candidates for explicit non-root
  runtime users in a later implementation story.
- `airflow-worker` is exceptional: it currently runs with root entrypoint
  privileges and mounts `/var/run/docker.sock` to launch local ML workload
  containers.

## Service-name dependency audit

The following dependencies are required by checked-in configuration files. They
are the primary justification for service discovery and network membership.

| Source | Target service name | Port | Protocol | Config source | Security boundary |
| ------ | ------------------- | ---- | -------- | ------------- | ----------------- |
| `monitoring-prometheus` | `monitoring-prometheus` | `9090` | HTTP | `docker/dev/prometheus/prometheus.yml` self scrape | Monitoring scrape boundary |
| `monitoring-prometheus` | `monitoring-cadvisor` | `8080` | HTTP | `docker/dev/prometheus/prometheus.yml` `cadvisor` job | Monitoring scrape boundary |
| `monitoring-prometheus` | `api-dev` | `10000` | HTTP `/metrics` | `docker/dev/prometheus/prometheus.yml` `api` job | Monitoring scrape to serving boundary |
| `monitoring-prometheus` | `monitoring-pushgateway` | `9091` | HTTP | `docker/dev/prometheus/prometheus.yml` `pushgateway` job | Monitoring scrape boundary |
| `monitoring-prometheus` | `monitoring-alertmanager` | `9093` | HTTP | `docker/dev/prometheus/prometheus.yml` alertmanager target | Alert routing boundary |
| `monitoring-grafana` | `monitoring-prometheus` | `9090` | HTTP | Grafana datasource provisioning | Dashboard query boundary |
| Airflow DAG tasks | `api-dev` | `10000` | HTTP | Airflow `api_dev` connection | API refresh boundary |
| Airflow DAG tasks | `ml-ingest-dev` | N/A | Docker API | Airflow variables and DockerOperator workflow | Local orchestration boundary |
| Airflow DAG tasks | `ml-features-dev` | N/A | Docker API | Airflow variables and DockerOperator workflow | Local orchestration boundary |
| Airflow DAG tasks | `ml-models-dev` | N/A | Docker API | Airflow variables and DockerOperator workflow | Local orchestration boundary |
| `ml-models-dev` | `mlflow-server` | `5000` | HTTP | `MLFLOW_TRACKING_URI` and Airflow variables | Tracking and logical artifact logging boundary |
| `mlflow-server` | `mlflow-postgres` | `5432` | PostgreSQL | `MLFLOW_BACKEND_STORE_URI` | Tracking metadata boundary |
| `mlflow-server` | `mlflow-minio` | `9000` | HTTP/S3 | `MLFLOW_S3_ENDPOINT_URL` | Artifact backend boundary |
| `mlflow-mc-init` | `mlflow-minio` | `9000` | HTTP/S3 | MinIO client alias setup | Artifact bootstrap boundary |
| Airflow services | `airflow-postgres` | `5432` | PostgreSQL | Airflow SQLAlchemy and result backend URIs | Orchestration metadata boundary |
| Airflow services | `airflow-redis` | `6379` | Redis | Airflow Celery broker URL | Orchestration broker boundary |
| Airflow services | `airflow-api-server` | `8080` | HTTP | Airflow execution API URL | Orchestration API boundary |
| Airflow services | `monitoring-mailhog` | `1025` | SMTP | Airflow SMTP config | Local email-dev boundary |
| `monitoring-alertmanager` | `monitoring-mailhog` | `1025` | SMTP | Alertmanager local route | Local email-dev boundary |
| ML jobs | `monitoring-pushgateway` | `9091` | HTTP | `PUSHGATEWAY_ADDR` | Batch metrics boundary |

Current ML model jobs should be read as logical MLflow clients. They request
tracking and artifact logging through `mlflow-server`; `mlflow-minio` is the
artifact backend owned by the MLflow runtime. The current document should not
promote a direct `ml-models-dev` data-lake dependency on MinIO.

A future local production-like design may intentionally expose MinIO, or another
object store, as a broader storage boundary for bronze, silver, and gold data:
raw or interim data, processed features, final prediction outputs, and model
artifacts. That is a target handoff strategy for Phase 7, not a current Compose
runtime dependency.

## Runtime identities

| Runtime identity | Current user model | Required permissions | Target strategy or follow-up |
| ---------------- | ------------------ | -------------------- | ---------------------------- |
| Host operator | Local user running `make` and `docker compose` | Docker group access, repository read/write, `.env` ownership, DVC setup when needed | Keep as local developer identity; document that Docker group membership is high privilege. |
| `api-dev` | Custom image, no explicit Compose `user` | Read final prediction data, write logs, serve HTTP on `10000` | Add non-root user in the API image or Compose override in a follow-up. |
| `ml-ingest-dev` | Custom image, no explicit Compose `user` | Read raw data, write interim data and logs, optionally push metrics | Add non-root user and writable mount ownership strategy in a follow-up. |
| `ml-features-dev` | Custom image, no explicit Compose `user` | Read interim data, write processed data and logs, optionally push metrics | Add non-root user and explicit data handoff ownership in a follow-up. |
| `ml-models-dev` | Custom image, no explicit Compose `user` | Read processed data, write final data, models, logs, and MLflow runs or artifacts through MLflow | Add non-root user; prefer MLflow tracking and artifact APIs before reducing host mounts. |
| `mlflow-postgres` | Upstream PostgreSQL image | Own private PostgreSQL volume | Keep private on `mlflow_net`; later use least-privilege managed database credentials. |
| `mlflow-minio` | Upstream MinIO image | Own MinIO data volume and expose S3/console locally | Keep as MLflow artifact backend now; evaluate scoped users or policies if it becomes a broader object store. |
| `mlflow-mc-init` | Upstream MinIO client helper | Bootstrap bucket using MinIO root credentials | Keep local bootstrap-only; move to scoped provisioning identity later. |
| `mlflow-server` | Upstream MLflow image | Reach PostgreSQL and MinIO, expose MLflow HTTP API/UI | Add explicit auth and service identity before production-like exposure. |
| `airflow-api-server` | Upstream Airflow image with `AIRFLOW_UID:0` group | Airflow metadata DB, Redis, DAG/config/log/data mounts, HTTP UI/API | Keep upstream user model for dev; avoid broad network access later. |
| `airflow-scheduler` | Upstream Airflow image with `AIRFLOW_UID:0` group | Metadata DB, Redis, DAG/config/log/data mounts, scheduling | Keep as orchestration identity; narrow non-required cross-stack access later. |
| `airflow-dag-processor` | Upstream Airflow image with `AIRFLOW_UID:0` group | Metadata DB, DAG/config/log/data mounts | Keep read/parse-focused; narrow cross-stack access later. |
| `airflow-triggerer` | Upstream Airflow image with `AIRFLOW_UID:0` group | Metadata DB, DAG/config/log/data mounts | Keep internal to orchestration boundary. |
| `airflow-worker` | Explicit `user: "0:0"` with root entrypoint | Celery worker, DAG/config/log/data mounts, Docker socket access, Docker group handling | Privileged local-development exception. Replace with issue #56 worker-pool or job-runner model. |
| `airflow-init` | Upstream Airflow image through common settings | Run DB migration, import variables/connections, create pool | Keep initialization-only; do not store real secrets in repository-managed config files. |
| `airflow-flower` | Upstream Airflow image through common settings | Read Celery state through broker, expose Flower UI locally | Local debug only; do not expose without auth in production-like runtime. |
| `monitoring-prometheus` | Upstream Prometheus image | Read repository-mounted config, scrape service names, persist TSDB volume | Keep as monitoring identity; define scrape auth and target scope later. |
| `monitoring-grafana` | Upstream Grafana image | Read provisioning config and dashboards, persist Grafana volume | Use local admin placeholder now; define admin/SSO/role model later. |
| `monitoring-cadvisor` | Upstream cAdvisor image, `privileged: true` | Host/runtime filesystem and Docker socket reads | Local observability exception. Replace with production-grade node metrics model later. |
| `monitoring-pushgateway` | Upstream Pushgateway image | Receive batch metrics writes and expose scrape endpoint | Add write controls and lifecycle cleanup later. |
| `monitoring-alertmanager` | Upstream Alertmanager image | Read local route config, send SMTP to MailHog, persist state | Local alert testing only; production mail credentials belong to the real notification target. |
| `monitoring-mailhog` | Upstream MailHog image | Receive local SMTP and expose local UI | Dev-only mail capture service; no local credentials are expected. Do not include it in production runtime. |

## Functional boundary model to prepare Phase 7

This section proposes boundaries only. It does not change Compose networks.

| Boundary | Required services or names | Required ports | Ingress | Egress | Credentials or auth | Mode |
| -------- | -------------------------- | -------------- | ------- | ------ | ------------------- | ---- |
| Serving/API boundary | `api-dev` | `10000` | Host clients, Airflow refresh, Prometheus scrape | Shared final data, logs, metrics | API demo credentials in local Makefile defaults and Airflow `api_dev` connection | Production-like concept, local demo defaults |
| Orchestration boundary | `airflow-api-server`, `airflow-scheduler`, `airflow-dag-processor`, `airflow-triggerer`, `airflow-worker`, `airflow-init`, `airflow-postgres`, `airflow-redis`, `airflow-flower` | `8080`, `5432`, `6379`, `5555` | Host UI/API for Airflow and Flower | API refresh, local Docker socket, ML jobs, MailHog, Pushgateway as configured | Airflow fernet key, JWT secret, PostgreSQL password, simple auth defaults | Mixed: production-like control plane with local-dev shortcuts |
| Tracking/artifact boundary | `mlflow-server`, `mlflow-postgres`, `mlflow-minio`, `mlflow-mc-init`, ML workloads as MLflow clients | `5000`, `5432`, `9000`, `9001` | Host MLflow/MinIO UI/API, MLflow client calls from ML workloads | PostgreSQL backend, MinIO artifact backend | MLflow placeholders, PostgreSQL password, MinIO root credentials, AWS variables | Production-like concept, local placeholders and defaults |
| Monitoring/scrape boundary | `monitoring-prometheus`, `monitoring-grafana`, `monitoring-cadvisor`, `monitoring-pushgateway`, `api-dev` metrics endpoint | `9090`, `3000`, `8080`, `9091`, `10000` | Host dashboards/UI, Prometheus scrapes | Scrape targets and datasource queries | Grafana admin credentials; no scrape auth currently | Local observability now, production-like target later |
| Alerting/email-dev boundary | `monitoring-alertmanager`, `monitoring-mailhog`, Airflow SMTP client | `9093`, `1025`, `8025` | Host Alertmanager/MailHog UI | SMTP to MailHog | No MailHog credentials by design; real SMTP secrets would belong to Airflow or Alertmanager in production | Local-dev only |
| Data/artifact handoff boundary | Current host `data`, `logs`, `models`; target object-store bronze/silver/gold datasets and MLflow artifacts | Filesystem today; S3-style API if promoted later | ML jobs, API, Airflow, host operator | Reads/writes shared bind mounts today; object storage in target design | Host filesystem permissions today; object-store credentials only if target storage is implemented | Local shared filesystem now; production-like contract later |
| Job execution boundary | `airflow-worker`, Docker daemon, `ml-ingest-dev`, `ml-features-dev`, `ml-models-dev` | Docker socket, service networks selected by DAG variables | Airflow worker | Docker daemon creates containers | Docker socket root-equivalent access, container env variables | Local-dev only; must be replaced before local-prod target |

The target Phase 7 network design should avoid pairwise network explosion. Broad
functional networks are acceptable when they represent a service discovery and
policy boundary. They should not be used to hide unclear privileges.

The data/artifact handoff target can evolve toward a data lake style contract.
In that model, object storage may hold bronze raw or interim datasets, silver
processed features, gold final predictions, and model or MLflow artifacts. This
would make MinIO, or an equivalent S3-compatible service, a deliberate storage
service with explicit credentials, bucket policies, lifecycle, and ownership.
That is distinct from the current local bind-mount based `data` handoff.

## Volume ownership and mount expectations

| Mount or volume | Current users | Ownership expectation | Production-like concern |
| --------------- | ------------- | --------------------- | ----------------------- |
| `./data:/app/data` | ML jobs and `api-dev` | Host developer owns the directory; containers need read/write for ML jobs and read for API final data | Replace implicit filesystem coupling with explicit data or artifact contracts. |
| `./data:/opt/airflow/data` | Airflow services | Airflow can inspect data for DAG coordination | Avoid DAG logic depending on mutable shared files where possible. |
| `./logs:/app/logs` | ML jobs and `api-dev` | Containers need write access and host needs read access | Move runtime logs to logging infrastructure later. |
| `./logs:/opt/airflow/logs` | Airflow services | Airflow components need shared log write access | Use remote log storage in production-like runtime. |
| `./models:/app/models` | `ml-models-dev` | Training job writes model artifacts | Prefer MLflow artifact APIs or registry as primary model handoff. |
| Airflow DAG mount | Airflow services | Repository DAG files are live-mounted read/write from host perspective | Local-dev convenience; deploy immutable DAG artifacts later. |
| Airflow config mounts | Airflow services and `airflow-init` | Versioned config is mounted into Airflow; init imports variables and connections | Keep real secrets out of repository-managed config. |
| Prometheus config mount | `monitoring-prometheus` | Read-only repository config | Package validated config artifacts later. |
| Grafana provisioning and dashboards | `monitoring-grafana` | Read-only provisioning, dashboard mount for local iteration | Secure datasource secrets and package dashboards later. |
| Alertmanager config mount | `monitoring-alertmanager` | Read-only local route config | Replace MailHog route with secure notification config later. |
| `/var/run/docker.sock` | `airflow-worker`, `monitoring-cadvisor` | Grants Docker API access from containers | Root-equivalent local-development boundary. |
| `mlflow-postgres-db` | `mlflow-postgres` | Private service-owned volume | Add backup, migration, and credential-rotation strategy. |
| `mlflow-minio-data` | `mlflow-minio` | Private MLflow artifact-store volume | If promoted to broader object storage, add bucket policy, data lifecycle, and ownership rules. |
| `airflow-postgres-db` | `airflow-postgres` | Private service-owned volume | Add backup and migration strategy. |
| `airflow-redis-data` | `airflow-redis` | Broker persistence volume | Review durability and password needs. |
| Monitoring state volumes | Prometheus, Grafana, Alertmanager | Private service-owned volumes | Add retention, backup, and provisioning strategy. |

## Containers that should target non-root execution

The following containers can reasonably target non-root execution after this
documentation story:

- `api-dev`;
- `ml-ingest-dev`;
- `ml-features-dev`;
- `ml-models-dev`.

Target strategy:

1. Add or verify a non-root user in each custom Dockerfile.
2. Give that user explicit read/write access to only the required runtime paths.
3. Align host bind mount ownership for `data`, `logs`, and `models`.
4. Keep service ports above `1024`, which is already true for `api-dev`.
5. Validate with `make compose-config`, targeted service startup, and ML pipeline
   smoke tests.

This should be handled as an implementation story because changing runtime users
can break bind mount writes and local Airflow-triggered jobs.

## Root or elevated runtime exceptions

| Service | Elevated behavior | Current justification | Follow-up |
| ------- | ----------------- | --------------------- | --------- |
| `airflow-worker` | Runs as `0:0`, uses root entrypoint, mounts Docker socket | Local DockerOperator execution needs socket group handling and container creation | Replace with issue #56 worker-pool or job-runner model. |
| `monitoring-cadvisor` | `privileged: true`, host/runtime mounts, Docker socket | Local container and host metrics collection | Replace with production-grade node/container metrics model or restrict host access. |
| Host operator | Docker group access | Required to run local Compose and build images | Document as high privilege; avoid treating it as application identity. |
| Upstream infrastructure images | Image-defined users or root defaults may vary | Local development baseline uses upstream images | Audit per image before local-prod implementation. |

## Docker socket boundary

`/var/run/docker.sock` is a privileged local-development boundary.

A container with write access to the Docker socket can ask the host Docker daemon
to create containers, mount host paths, join networks, and access secrets or data
available to the Docker daemon. In practice, this is close to host-level control.

Current usage:

- `airflow-worker` uses the socket to create ML workload containers from Airflow
  DAG tasks.
- `monitoring-cadvisor` uses runtime mounts and the Docker socket to collect
  local container metrics.

This is not a production-like orchestration model. Production-like orchestration
should move toward a controlled job execution boundary, for example:

- pre-started non-root worker pool;
- queue-based job runner;
- controlled internal job API;
- restricted container-runtime proxy as a temporary transition;
- Kubernetes Jobs or CronJobs later.

Follow-up: issue #56 should design the Airflow job runner or worker-pool model.

## Credential and secret review

`.env.template` already separates local developer settings, DVC/DagsHub remote
credentials, MLflow target variables, MinIO artifact credentials, Airflow secrets,
Grafana admin credentials, API demo credentials, monitoring settings, and time
zone configuration.

Credential classes:

| Class | Current location | Current status | Required action |
| ----- | ---------------- | -------------- | --------------- |
| Local developer Git identity | `.env.template` `GIT_USER`, `GIT_EMAIL` | Placeholder values | Keep local-only. |
| DVC/DagsHub credentials | `.env.template` `DAGSHUB_*` | Placeholder values | Keep out of Git; stored locally through `.dvc/config.local`. |
| MLflow remote credentials | `.env.template` `MLFLOW_TRACKING_USERNAME_*`, `MLFLOW_TRACKING_PASSWORD_*` | Placeholder or empty by mode | Keep mode separation; do not mix DagsHub and Compose credentials. |
| MLflow Compose tracking | `.env.template` `MLFLOW_TRACKING_URI_COMPOSE` | Internal service URL | Keep as service-name dependency. |
| MinIO artifact credentials | `.env.template` `MINIO_ROOT_*`, `AWS_*` | Passwords use placeholders, user defaults to local `minio` | Keep local defaults for dev; create scoped policies if MinIO becomes a broader object store. |
| Airflow secrets | `.env.template` `AIRFLOW_FERNET_KEY`, `AIRFLOW_API_AUTH_JWT_SECRET`, `AIRFLOW_POSTGRES_PASSWORD` | Placeholder values | Keep required placeholders; generate locally. |
| Airflow simple auth | `docker-compose.yaml` `admin:admin` | Local development default | Mark local-dev only; replace before production-like runtime. |
| Airflow connection to API | `docker/dev/airflow/config/connections.json` | Hardcoded `remy` / `remy` | Local-only default; move to `.env`, Airflow secret backend, or generated config. |
| Airflow MLflow/MinIO variables | `docker/dev/airflow/config/variables.json` | Hardcoded `minio` / `minio123` | Local-only default; align with `.env` or generated Airflow variables later. |
| Airflow host repository path | `docker/dev/airflow/config/variables.json` | Hardcoded local host path | Local developer-specific value; move to `.env` or generated config. |
| Grafana admin credentials | `.env.template` `GRAFANA_ADMIN_*` | Password placeholder, local user default | Keep placeholder; rotate and externalize later. |
| API demo credentials | `.env.template` `API_USER`, `API_PASS` | Local demo default `user1` / `user1` | Mark local demo only; replace before production-like runtime. |
| MailHog SMTP | Compose and Airflow SMTP settings | No auth by design, local capture only | Keep credential-free in local dev; do not run MailHog in production. |
| Prometheus and Pushgateway | Compose and Prometheus config | No scrape or write auth in local dev | Add controls only for a production-like runtime that exposes these paths. |

Local-only defaults and placeholders that must not be promoted to production:

- Airflow simple auth `admin:admin`;
- API demo defaults `user1:user1`;
- Airflow `api_dev` connection `remy:remy`;
- Airflow variables `minio:minio123`;
- Grafana local admin user when paired with a local placeholder password;
- MailHog unauthenticated SMTP and UI, which is expected for local mail capture;
- Prometheus, Pushgateway, Alertmanager, and cAdvisor local UIs without Compose
  authentication.

These values are compatibility shortcuts for the current local `docker/dev`
stack. They are not production secrets and do not imply that every local helper
needs credentials. For example, MailHog remains a local-only capture service;
production email security belongs to the real SMTP provider and to the services
that send email, such as Airflow or Alertmanager.

## Phase 7 follow-up map

This document directly feeds Phase 7 design and implementation stories:

| Follow-up | Input from this document |
| --------- | ------------------------ |
| Issue #55 local production-like network topology | Functional boundaries, service-name dependency audit, ingress and egress requirements. |
| Issue #56 Airflow job runner worker pool | Docker socket risk, `airflow-worker` exception, job execution boundary. |
| Issue #57 local production-like Compose runtime | Non-root target strategy, credential classes, volume ownership expectations, local-dev exceptions. |
| Issue #49 repository structure | Runtime asset ownership, dev versus local-prod documentation expectations, config and DAG placement. |

## Validation

This story introduces documentation only. It should not change Docker Compose
behavior, runtime users, networks, secrets, or service startup order.

Expected validation remains:

```bash
make compose-config
```

Additional review checklist:

- `docs/runtime-communication-matrix.md` remains the current-state reference.
- This document remains a target-boundary and identity design reference.
- No Docker socket removal, network refactor, secret manager, or `docker/prod`
  runtime is introduced by this story.
