# Airflow job runner remaining gaps

This document coordinates the remaining production-like Airflow and runner gaps
after the typed ML step services and production-like DAG chain were implemented.
Current runtime and architecture references describe the stable implemented
state.

## Scope and inputs

| Source | Use in this design |
| ------ | ------------------ |
| [`../README.md`](../README.md) | Documentation hierarchy and level rules. |
| [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md) | Current dev/prod runtime split, production-like DAGs, and runner API operation. |
| [`../architecture-references/runtime-communication-matrix.md`](../architecture-references/runtime-communication-matrix.md) | Current communication paths, runner boundary, and network traffic. |
| [`../architecture-references/runtime-security-boundaries.md`](../architecture-references/runtime-security-boundaries.md) | Runtime identities, Docker socket risk, and implemented service boundaries. |
| [`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md) | Implemented functional networks and service placement. |
| [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) | Manifest-first artifact handoff contract and open artifact gaps. |

## Implemented state now owned by current docs

The implemented production-like path is documented in the runtime and architecture
references. Stable wording should stay there, not in this design note:

- `job-runner-api` exposes typed job submission and status endpoints;
- `ml-ingest-prod`, `ml-features-prod`, and `ml-models-prod` execute concrete ML
  steps through internal FastAPI services;
- production-like Airflow uses `bike_traffic_orchestrator`, `bike_traffic_init`,
  and `bike_traffic_daily` without Docker socket access;
- Airflow chains ingest, features, models, manifest reads, and authenticated API
  refresh through the `api_prod` connection;
- `job-runner-api` and the ML step services keep execution serialized for the
  current single-service local runtime.

## Remaining coordination topics

| Topic | Status | Source of truth |
| ----- | ------ | --------------- |
| API serving from promoted manifests | Open | [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) |
| Runner job status durability | Open | Future runner storage design |
| Runner metrics and freshness observability | Open | Future observability design |
| Runtime configuration and secret validation | Open | Runtime hardening work |
| Full production-like smoke validation coverage | Open | CI/runtime validation work |

## Open design gaps

- The API still needs to serve predictions through promoted manifests instead of
  relying only on runtime final-data paths.
- Runner job status is in memory and is not durable across process restarts.
- Runner metrics and Prometheus/Grafana freshness views still need an explicit
  contract.
- Configuration validation still needs to reject unsafe placeholder values for
  custom services.
- Production-like smoke validation should cover the complete Airflow, runner, ML
  step service, artifact, and API refresh chain.

## Validation target

A complete validation should prove that:

- `docker/prod` Airflow has no Docker socket mount;
- Airflow can submit typed step jobs to `job-runner-api`;
- the runner can delegate each ML step without exposing Docker runtime control to
  Airflow;
- each ML step emits coherent artifact manifests;
- Airflow chains ingest, features, and models visibly;
- the API refresh runs only after successful model jobs;
- Prometheus/Grafana can observe job status and artifact freshness once the
  observability contract is implemented.
