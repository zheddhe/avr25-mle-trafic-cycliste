# Local production-like Compose runtime

This document is the Phase 7 implementation note for issue #57. It introduces
and explains the separate local production-like Docker Compose entrypoint under
`docker/prod`.

The current root `docker-compose.yaml` and `docker/dev` runtime remain the local
development runtime. They optimize for debugging, host visibility, live-mounted
runtime assets, and direct inspection of local service UIs.

The `docker/prod` runtime optimizes for a different goal: it gives the project a
local target that is closer to a production operating model while still running
on a single developer host.

## When to use each runtime

| Runtime | Entry point | Primary use |
| ------- | ----------- | ----------- |
| Development | `docker-compose.yaml` | Debugging, local demos, broad host visibility, DockerOperator-based Airflow ML jobs. |
| Local production-like | `docker/prod/docker-compose.yaml` | Network and exposure validation, least-privilege rehearsal, monitoring smoke tests, future runner integration. |

Use `docker/dev` when iterating on DAGs, ML CLI containers, data files, logs, or
model outputs. Use `docker/prod` when validating service boundaries, reduced
host exposure, non-root application containers, and the future runner migration
path.

## Operational commands

The local production-like runtime has a dedicated Makefile so the root Makefile
and the current development workflow stay unchanged.

```bash
make -f docker/prod/Makefile prod-compose-config
make -f docker/prod/Makefile prod-ops
make -f docker/prod/Makefile prod-ps
make -f docker/prod/Makefile prod-logs SERVICE=api-dev
make -f docker/prod/Makefile prod-stop
```

The equivalent raw Compose validation command is:

```bash
docker compose \
    --env-file .env \
    -f docker/prod/docker-compose.yaml \
    --profile all \
    config
```

The existing validation command remains valid for the development runtime:

```bash
make compose-config
```

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

The broad development `mlops_net` is not used by `docker/prod`.

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

The production-like runtime reduces direct host mounts, but it does not remove
all artifact mounts in the first implementation.

| Mount | Status | Reason |
| ----- | ------ | ------ |
| `data/final` into API | Read-only | API serving needs promoted prediction artifacts. |
| `data` into ML one-off services | Writable temporary exception | The runner and artifact handoff contract are not implemented yet. |
| `models` into model service | Writable temporary exception | Model artifacts still use the local workspace. |
| `logs` into API, Airflow, and ML services | Writable temporary exception | Local log inspection remains the current evidence path. |
| Airflow DAG/config files | Read-only | DAG and config placement is explicit for this runtime. |
| Monitoring configs | Read-only | Prometheus, Alertmanager, and Grafana provisioning remain versioned assets. |

The expected next hardening step is an explicit artifact handoff contract using
object storage, release manifests, or a promotion service.

## Runtime identities and exceptions

Custom API and ML containers run as a non-root application user in
`docker/prod`. They also drop Linux capabilities and use `no-new-privileges`.

Documented exceptions:

- Airflow uses the upstream Airflow image and its supported runtime user model.
- cAdvisor remains privileged because local container metrics require access to
  host and Docker runtime paths.
- Infrastructure images such as PostgreSQL, Redis, MinIO, Prometheus, Grafana,
  and Alertmanager keep their upstream image users unless a separate hardening
  story validates a safer override.

## Secrets and placeholders

`docker/prod` still reads `.env` from the repository root. Do not commit a
populated `.env` file.

The production-like Airflow config uses placeholder credentials where values are
not safe to commit. Replace placeholders in a local branch or use a future secret
injection mechanism before running authenticated flows.

## Validation checklist

Minimum validation for this story:

```bash
make compose-config
make -f docker/prod/Makefile prod-compose-config
```

Additional local smoke checks after startup:

```bash
make -f docker/prod/Makefile prod-ops
make -f docker/prod/Makefile prod-ps

docker compose --env-file .env -f docker/prod/docker-compose.yaml \
    exec monitoring-prometheus wget -qO- http://api-dev:10000/metrics

docker compose --env-file .env -f docker/prod/docker-compose.yaml \
    exec monitoring-grafana wget -qO- http://monitoring-prometheus:9090/-/ready

docker compose --env-file .env -f docker/prod/docker-compose.yaml \
    exec mlflow-server getent hosts mlflow-postgres

docker compose --env-file .env -f docker/prod/docker-compose.yaml \
    exec airflow-worker getent hosts api-dev
```
