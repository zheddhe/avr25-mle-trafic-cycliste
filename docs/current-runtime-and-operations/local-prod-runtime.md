# Local Compose runtimes

This document describes the implemented local Docker Compose runtime model on
`main`. It is a current-state operational guide, not a Phase 8 backlog.

Planned runtime changes, remaining Phase 8 work, and validation debt are tracked
under [`../next-phase-design/`](../next-phase-design/).

The development runtime optimizes for debugging, host visibility, live-mounted
runtime assets, and direct inspection of local service UIs. The production-like
runtime optimizes for stricter local operation, reduced host exposure, functional
networks, non-root custom application images, and explicit temporary exceptions.

The Phase 7 implementation introduced the symmetric `docker/dev` and
`docker/prod` layout. The current Phase 8 implementation adds manifest and job
contract foundations, while the runner execution path remains planned work.

## When to use each runtime

| Runtime | Entry point | Primary use |
| ------- | ----------- | ----------- |
| Development | `docker/dev/docker-compose.yaml` | Debugging, local demos, broad host visibility, DockerOperator-based Airflow ML jobs. |
| Local production-like | `docker/prod/docker-compose.yaml` | Network and exposure validation, least-privilege rehearsal, isolated runtime workspaces, and monitoring smoke checks. |

Use `docker/dev` when iterating on DAGs, ML CLI containers, root-level DVC data,
logs, or model outputs.

Use `docker/prod` when validating implemented service boundaries, reduced host
exposure, non-root application containers, and isolated runtime workspaces.

Do not use a root-level Compose file. Runtime commands must go through the
runtime-specific Make targets or an explicit `docker compose -f` command.

## Documentation boundary

Use this document for current runtime operation:

- current commands;
- current workspaces;
- current networks;
- current mounts;
- current service exposure;
- current security exceptions.

Use [`../next-phase-design/artifact-handoff-strategy.md`](../next-phase-design/artifact-handoff-strategy.md)
and [`../next-phase-design/airflow-job-runner-strategy.md`](../next-phase-design/airflow-job-runner-strategy.md)
for target runner behavior, artifact-aware API serving, future smoke validation,
configuration hardening, and remaining Phase 8 tracking.

## Related documentation

Start from [`../README.md`](../README.md) for the full documentation map.

| Document | Role |
| -------- | ---- |
| [`repository-structure.md`](repository-structure.md) | Repository ownership, DVC boundaries, and runtime workspace ownership. |
| [`ports-and-services.md`](ports-and-services.md) | Implemented host exposure and internal-only services. |
| [`../architecture-references/runtime-communication-matrix.md`](../architecture-references/runtime-communication-matrix.md) | Implemented service traffic, networks, and mounts. |
| [`../architecture-references/runtime-security-boundaries.md`](../architecture-references/runtime-security-boundaries.md) | Runtime identities, Docker socket risk, and security boundaries. |
| [`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md) | Implemented production-like network topology. |
| [`../next-phase-design/artifact-handoff-strategy.md`](../next-phase-design/artifact-handoff-strategy.md) | Phase 8 target artifact handoff and remaining plan. |
| [`../next-phase-design/airflow-job-runner-strategy.md`](../next-phase-design/airflow-job-runner-strategy.md) | Phase 8 target runner-based Airflow execution. |

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
| `docker/prod/runtime/models` | Production-like generated model workspace. |
| `docker/prod/runtime/logs` | Production-like service and batch logs. |
| `docker/prod/runtime/artifacts` | Manifest-first artifact handoff root. |

Only the required business source CSV is mounted from the root development/DVC
workspace into the production-like ingestion service as read-only input:

```text
data/raw/comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv
```

This keeps DVC ownership local to the root workspace while allowing `docker/prod`
to run without writing into root `data`, `models`, or `logs`.

Current Phase 8 foundations available to runtime services:

- reusable artifact manifest models and store helpers;
- ML manifest emission wrappers;
- typed job request and status contracts.

## Network topology

The `docker/prod` Compose file implements the functional network design from
[`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md).

| Network | Main responsibility |
| ------- | ------------------- |
| `orchestration_net` | Airflow control plane, metadata DB, and broker. |
| `pipeline_runtime_net` | API refresh, batch job handoff, and Pushgateway writes. |
| `tracking_client_net` | MLflow client calls from model workloads. |
| `tracking_backend_net` | MLflow PostgreSQL and MinIO backend isolation. |
| `observability_net` | Prometheus scrapes, Grafana datasource access, and alert routing. |
| `dev_support_net` | Local SMTP capture for Airflow and Alertmanager. |

The broad development `mlops_net` remains available in `docker/dev` because the
current Airflow DockerOperator path depends on the development network model. It
is not used by `docker/prod`.

## Current worker-pool status

`docker/prod` already removes the main production-like anti-pattern from Airflow:
Airflow services do not mount `/var/run/docker.sock`.

Current behavior:

- Airflow services do not mount `/var/run/docker.sock` in `docker/prod`.
- The Airflow worker does not run through the development Docker socket entrypoint.
- Production-like Airflow config points to `pipeline_runtime_net`, not `mlops_net`.
- DockerOperator-based ML DAG execution remains available in `docker/dev` only.
- ML services can emit local promoted manifests when `ARTIFACT_MANIFEST_ROOT` is
  configured.
- Typed job contracts exist for the planned runner path.

A `job-runner-api` service, runner execution path, and production-like Airflow
DAG that calls the runner are not part of the current runtime yet. They are
tracked in [`../next-phase-design/airflow-job-runner-strategy.md`](../next-phase-design/airflow-job-runner-strategy.md).

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
| `docker/prod/runtime/data` | Prod | Writable | Production-like generated data workspace. |
| `docker/prod/runtime/models` | Prod | Writable | Production-like model workspace. |
| `docker/prod/runtime/logs` | Prod | Writable | Production-like service and batch log workspace. |
| `docker/prod/runtime/artifacts` | Prod | Writable by ML jobs | Manifest-first handoff root. |
| Airflow DAG/config files | Prod | Read-only | DAG and config placement is explicit for this runtime. |
| Monitoring configs | Prod | Read-only | Prometheus, Alertmanager, and Grafana provisioning remain versioned assets. |

Phase 8 hardens this model through explicit artifact manifests and promotion
rules, not by making the root development workspace production-like.

## Runtime identities and exceptions

Custom API and ML containers run as a non-root application user in `docker/prod`.
They also drop Linux capabilities and use `no-new-privileges`. The prod
Dockerfiles default this user to UID/GID `1000` to keep local bind-mount writes
compatible with common developer hosts.

Documented exceptions:

- Airflow uses the upstream Airflow image and its supported runtime user model.
- cAdvisor remains privileged because local container metrics require access to
  host and Docker runtime paths.
- Infrastructure images such as PostgreSQL, Redis, MinIO, Prometheus, Grafana,
  and Alertmanager keep their upstream image users unless a separate hardening
  story validates a safer override.

## Secrets and placeholders

`docker/dev` and `docker/prod` read `.env` from the repository root. Do not
commit a populated `.env` file.

The production-like Airflow config uses placeholder credentials where values are
not safe to commit. Replace placeholders locally or use a future secret injection
mechanism before running authenticated flows.

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
