# Local Compose runtimes

This document is the Phase 7 implementation note for issue #57. It introduces
and explains the symmetric local Docker Compose runtime layout under
`docker/dev` and `docker/prod`.

The development runtime optimizes for debugging, host visibility, live-mounted
runtime assets, and direct inspection of local service UIs. The production-like
runtime optimizes for stricter local operation, reduced host exposure, functional
networks, non-root custom application images, and explicit temporary exceptions.

## When to use each runtime

| Runtime | Entry point | Primary use |
| ------- | ----------- | ----------- |
| Development | `docker/dev/docker-compose.yaml` | Debugging, local demos, broad host visibility, DockerOperator-based Airflow ML jobs. |
| Compatibility dev entrypoint | `docker-compose.yaml` | Existing habits and raw `docker compose` commands from the repository root. |
| Local production-like | `docker/prod/docker-compose.yaml` | Network and exposure validation, least-privilege rehearsal, monitoring smoke tests, future runner integration. |

Use `docker/dev` when iterating on DAGs, ML CLI containers, root-level DVC data,
logs, or model outputs. Use `docker/prod` when validating service boundaries,
reduced host exposure, non-root application containers, isolated runtime
workspaces, and the future runner migration path.

## Operational commands

The root Makefile now includes the dedicated runtime Makefiles:

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

## Runtime workspace strategy

The root `data`, `models`, and `logs` folders remain development, DVC, and
host-local workspaces. They are still used by the development runtime and by
local Python/DVC commands.

The local production-like runtime writes to ignored runtime workspaces under
`docker/prod/runtime`:

| Path | Purpose |
| ---- | ------- |
| `docker/prod/runtime/data` | Production-like generated data workspace. |
| `docker/prod/runtime/models` | Production-like generated model workspace. |
| `docker/prod/runtime/logs` | Production-like service and batch logs. |

Only the required business source CSV is mounted from the root development/DVC
workspace into the production-like ingestion service as read-only input:

```text
data/raw/comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv
```

This keeps DVC ownership local to the root workspace while allowing `docker/prod`
to run without writing into root `data`, `models`, or `logs`.

## Network topology

The `docker/prod` Compose file follows the target functional network design from
`docs/local-prod-network-topology.md`.

| Network | Main responsibility |
| ------- | ------------------- |
| `orchestration_net` | Airflow control plane, metadata DB, broker, and internal execution API. |
| `pipeline_runtime_net` | API refresh, batch job handoff, and Pushgateway writes. |
| `tracking_client_net` | MLflow client calls from model workloads. |
| `tracking_backend_net` | MLflow PostgreSQL and MinIO backend isolation. |
| `observability_net` | Prometheus scrapes, Grafana datasource access, and alert routing. |
| `dev_support_net` | Local SMTP capture for Airflow and Alertmanager. |

The broad development `mlops_net` remains available in `docker/dev` because the
current Airflow DockerOperator path depends on the development network model. It
is not used by `docker/prod`.

## Worker-pool status

The target worker-pool strategy from `docs/airflow-job-runner-strategy.md` is not
fully implemented in this story. This runtime deliberately avoids pretending that
the current DockerOperator path is production-like.

Current behavior in `docker/prod`:

- Airflow services do not mount `/var/run/docker.sock`.
- The Airflow worker does not run through the development root entrypoint.
- Production-like Airflow config points to `pipeline_runtime_net`, not
  `mlops_net`.
- Current DockerOperator-based ML DAG execution is a known temporary exception:
  it remains available in `docker/dev`, not in `docker/prod`.

Expected follow-up:

1. Add typed job submission schemas and a runner client under `src/`.
2. Add an internal `job-runner-api` and typed ML worker services.
3. Update the production-like DAG variant to submit runner jobs instead of
   creating containers.
4. Validate init and daily DAGs through the runner before removing the temporary
   exception from this document.

## Host exposure

The local production-like runtime publishes only operator-facing services needed
for local validation by default:

| Service | Reason |
| ------- | ------ |
| `api-dev` | Local prediction API, OpenAPI docs, and smoke tests. |
| `airflow-api-server` | Local DAG inspection and orchestration UI. |
| `mlflow-server` | Local tracking inspection while the model registry path matures. |
| `monitoring-grafana` | Local dashboard entrypoint. |

MinIO, Prometheus, Pushgateway, Alertmanager, MailHog, cAdvisor, Flower, Redis,
and PostgreSQL services stay internal in `docker/prod`.

## Mount strategy

| Mount | Runtime | Status | Reason |
| ----- | ------- | ------ | ------ |
| Root `data`, `models`, `logs` | Dev | Writable | Debug visibility, local DVC, and current DockerOperator jobs. |
| Root source CSV | Prod | Read-only | Required business input for ingestion. |
| `docker/prod/runtime/data` | Prod | Writable | Temporary production-like data workspace. |
| `docker/prod/runtime/models` | Prod | Writable | Temporary production-like model workspace. |
| `docker/prod/runtime/logs` | Prod | Writable | Temporary production-like log workspace. |
| Airflow DAG/config files | Prod | Read-only | DAG and config placement is explicit for this runtime. |
| Monitoring configs | Prod | Read-only | Prometheus, Alertmanager, and Grafana provisioning remain versioned assets. |

The expected next hardening step is an explicit artifact handoff contract using
object storage, release manifests, or a promotion service.

## Runtime identities and exceptions

Custom API and ML containers run as a non-root application user in
`docker/prod`. They also drop Linux capabilities and use `no-new-privileges`.
The prod Dockerfiles default this user to UID/GID `1000` to keep local bind-mount
writes compatible with common developer hosts.

Documented exceptions:

- Airflow uses the upstream Airflow image and its supported runtime user model.
- cAdvisor remains privileged because local container metrics require access to
  host and Docker runtime paths.
- Infrastructure images such as PostgreSQL, Redis, MinIO, Prometheus, Grafana,
  and Alertmanager keep their upstream image users unless a separate hardening
  story validates a safer override.

## Secrets and placeholders

`docker/dev` and `docker/prod` still read `.env` from the repository root. Do not
commit a populated `.env` file.

The production-like Airflow config uses placeholder credentials where values are
not safe to commit. Replace placeholders in a local branch or use a future secret
injection mechanism before running authenticated flows.

## Validation checklist

Minimum validation for this story:

```bash
make dev-compose-config
make prod-compose-config
```

Additional local smoke checks after startup:

```bash
make dev-start
make prod-start
make prod-ps

docker compose --env-file .env -f docker/prod/docker-compose.yaml \
    -p trafic-cycliste-prod exec monitoring-prometheus \
    wget -qO- http://api-dev:10000/metrics

docker compose --env-file .env -f docker/prod/docker-compose.yaml \
    -p trafic-cycliste-prod exec monitoring-grafana \
    wget -qO- http://monitoring-prometheus:9090/-/ready

docker compose --env-file .env -f docker/prod/docker-compose.yaml \
    -p trafic-cycliste-prod exec mlflow-server getent hosts mlflow-postgres

docker compose --env-file .env -f docker/prod/docker-compose.yaml \
    -p trafic-cycliste-prod exec airflow-worker getent hosts api-dev
```
