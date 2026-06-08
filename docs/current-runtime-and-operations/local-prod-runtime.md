# Local Compose runtimes

This document describes the implemented local Docker Compose runtime model on
`main`. It is a current-state operational guide.

The development runtime optimizes for debugging, host visibility, live-mounted
runtime assets, and direct inspection of local service UIs. The production-like
runtime optimizes for stricter local operation, reduced host exposure, functional
networks, non-root custom application images, and isolated runtime workspaces.

## When to use each runtime

| Runtime | Entry point | Primary use |
| ------- | ----------- | ----------- |
| Development | `docker/dev/docker-compose.yaml` | Debugging, local demos, broad host visibility, DockerOperator-based Airflow ML jobs. |
| Local production-like | `docker/prod/docker-compose.yaml` | Network and exposure validation, isolated runtime workspaces, internal runner API checks, and monitoring smoke checks. |

Use `docker/dev` when iterating on DAGs, ML CLI containers, root-level DVC data,
logs, or model outputs.

Use `docker/prod` when validating implemented service boundaries, reduced host
exposure, non-root application containers, internal service discovery, runner API
behavior, and isolated runtime workspaces.

Do not use a root-level Compose file. Runtime commands must go through the
runtime-specific Make targets or an explicit `docker compose -f` command.

## Related documentation

Start from [`../README.md`](../README.md) for the full documentation map and the
rules that separate current runtime docs, architecture references, and active
design notes.

| Document | Role |
| -------- | ---- |
| [`repository-structure.md`](repository-structure.md) | Repository ownership, DVC boundaries, and runtime workspace ownership. |
| [`ports-and-services.md`](ports-and-services.md) | Implemented host exposure and internal-only services. |
| [`../architecture-references/runtime-communication-matrix.md`](../architecture-references/runtime-communication-matrix.md) | Implemented service traffic, runner boundary, networks, and mounts. |
| [`../architecture-references/runtime-security-boundaries.md`](../architecture-references/runtime-security-boundaries.md) | Runtime identities, Docker socket risk, and security boundaries. |
| [`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md) | Implemented production-like network topology. |

## Operational commands

The root Makefile includes the dedicated runtime Makefiles:

- `docker/dev/Makefile` for local development Compose operations;
- `docker/prod/Makefile` for local production-like Compose operations.

Explicit runtime commands are preferred for cross-checks:

```bash
make dev-compose-config
make dev-start
make dev-ps

make prod-compose-config
make prod-start
make prod-ps
```

Runtime-scoped cleanups:

```bash
make dev-clean
make prod-clean
```

## Runtime workspace strategy

The root `data`, `models`, and `logs` folders are development, DVC, and
host-local workspaces. They are used by the development runtime and by local
Python/DVC commands.

The local production-like runtime writes to ignored runtime workspaces under
`docker/prod/runtime`:

| Path | Purpose |
| ---- | ------- |
| `docker/prod/runtime/data` | Production-like generated data workspace. |
| `docker/prod/runtime/models` | Production-like model workspace. |
| `docker/prod/runtime/logs` | Production-like service and batch log workspace. |
| `docker/prod/runtime/artifacts` | Manifest-first artifact handoff root. |

Only the required business source CSV is mounted from the root development/DVC
workspace into the production-like ingestion service as read-only input:

```text
data/raw/comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv
```

This keeps DVC ownership local to the root workspace while allowing `docker/prod`
to run without writing into root `data`, `models`, or `logs`.

## Production-like service boundary

The production-like runtime contains:

- custom non-root API, runner API, and ML images;
- Airflow services without `/var/run/docker.sock` mounts;
- MLflow, MinIO, PostgreSQL, Redis, and monitoring support services;
- internal `ml-ingest-prod`, `ml-features-prod`, and `ml-models-prod` FastAPI
  services for concrete ML step execution;
- an internal `job-runner-api` service for typed ML step submission, status reads,
  and failure mapping.

`job-runner-api` is a FastAPI service listening on container port `10080`. It
exposes `/health`, `/jobs`, and `/jobs/{job_id}` on internal Docker networks. It
keeps job status in process memory and delegates one allow-listed typed ML step at
a time to the matching internal ML service through Compose DNS.

The active runner contract accepts only `ingest`, `features`, and `models` jobs
from `src/ml/jobs`. It does not expose a pipeline-wide runtime job. Submitted
jobs move through `queued`, `running`, and a terminal `succeeded` or `failed`
state. Successful jobs return output path, metrics, and optional artifact
manifest evidence. Controlled execution failures return structured job errors.

The runner implementation is intentionally synchronous and in-memory for the
local production-like runtime. It is not a durable queue, distributed scheduler,
Docker SDK wrapper, shell command runner, or Kubernetes job controller.

`docker/prod` Airflow services can start without Docker socket access. The
production-like DAG under `docker/prod/airflow/dags` submits `ingest`, then
`features`, then `models` jobs to `job-runner-api`, and only triggers API refresh
after the model step has succeeded.

## Network topology

The `docker/prod` Compose file implements the functional network design from
[`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md).

| Network | Main responsibility |
| ------- | ------------------- |
| `orchestration_net` | Airflow control plane, metadata DB, and broker. |
| `pipeline_runtime_net` | API refresh, batch job handoff, runner API access, and Pushgateway writes. |
| `tracking_client_net` | MLflow client calls from model workloads. |
| `tracking_backend_net` | MLflow PostgreSQL and MinIO backend isolation. |
| `observability_net` | Prometheus scrapes, Grafana datasource access, and alert routing. |
| `dev_support_net` | Local SMTP capture for Airflow and Alertmanager. |

The broad development `mlops_net` remains available in `docker/dev` because the
current Airflow DockerOperator path depends on the development network model. It
is not used by `docker/prod`.

## Host exposure

The local production-like runtime publishes only operator-facing services needed
for local validation by default:

| Service | Reason |
| ------- | ------ |
| `api-dev` | Local prediction API, OpenAPI docs, and smoke tests. |
| `airflow-api-server` | Local DAG inspection and orchestration UI. |
| `mlflow-server` | Local tracking inspection while the model registry path matures. |
| `monitoring-grafana` | Local dashboard entrypoint. |

`job-runner-api`, ML step services, MinIO, Prometheus, Pushgateway,
Alertmanager, MailHog, cAdvisor, Redis, and PostgreSQL services stay internal in
`docker/prod`.

## Mount strategy

| Mount | Runtime | Status | Reason |
| ----- | ------- | ------ | ------ |
| Root `data`, `models`, `logs` | Dev | Writable | Debug visibility, local DVC, and current DockerOperator jobs. |
| Root source CSV | Prod | Read-only | Required business input for ingestion. |
| `docker/prod/runtime/data` | Prod | Writable | Production-like generated data workspace. |
| `docker/prod/runtime/models` | Prod | Writable | Production-like model workspace. |
| `docker/prod/runtime/logs` | Prod | Writable | Production-like service and batch log workspace. |
| `docker/prod/runtime/artifacts` | Prod | Writable by ML jobs | Manifest-first handoff root. |
| `docker/prod/runtime/logs/job-runner` | Prod | Writable by `job-runner-api` | Runner API service logs. |
| Airflow DAG/config files | Prod | Read-only | DAG and config placement is explicit for this runtime. |
| Monitoring configs | Prod | Read-only | Prometheus, Alertmanager, and Grafana provisioning remain versioned assets. |

The runner API service mounts only its log directory in Compose. The concrete ML
services own runtime data, model, log, artifact, and optional MLflow access. The
runner therefore keeps its control-plane boundary narrow and does not need Docker
or broad artifact workspace mounts.

## Runtime identities and exceptions

Custom API, runner API, and ML containers run as a non-root application user in
`docker/prod`. They also drop Linux capabilities and use `no-new-privileges`.
The prod Dockerfiles default this user to UID/GID `1000` to keep local
bind-mount writes compatible with common developer hosts.

Documented exceptions:

- Airflow uses the upstream Airflow image and its supported runtime user model.
- cAdvisor remains privileged because local container metrics require access to
  host and Docker runtime paths.
- Infrastructure images such as PostgreSQL, Redis, MinIO, Prometheus, Grafana,
  and Alertmanager keep their upstream image users unless a service-specific
  validation proves a safer override.

## Secrets and placeholders

`docker/dev` and `docker/prod` read `.env` from the repository root. Do not
commit a populated `.env` file.

The production-like Airflow config uses placeholder credentials where values are
not safe to commit. Replace placeholders locally before running authenticated
flows.

## Validation checklist

Minimum validation:

```bash
make dev-compose-config
make prod-compose-config
```

Additional local smoke checks after startup:

```bash
make dev-start
make prod-start
make prod-ps
```

Runner API checks inside the production-like Compose network:

```bash
docker compose \
    --env-file .env \
    -f docker/prod/docker-compose.yaml \
    --profile ptf up -d job-runner-api

docker compose \
    --env-file .env \
    -f docker/prod/docker-compose.yaml \
    exec job-runner-api \
    python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:10080/health').read().decode())"
```
