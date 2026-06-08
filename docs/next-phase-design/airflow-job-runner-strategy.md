# Airflow job runner strategy

This document coordinates the remaining Phase 8 work for production-like Airflow
orchestration after typed ML step execution was added to `job-runner-api`.

Current runtime and architecture references describe what is already implemented.
This document keeps only the open validation and follow-up coordination targets.

## Scope and inputs

| Source | Use in this design |
| ------ | ------------------ |
| [`../README.md`](../README.md) | Documentation hierarchy and level rules. |
| [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md) | Current dev/prod runtime split and runner API operation. |
| [`../architecture-references/runtime-communication-matrix.md`](../architecture-references/runtime-communication-matrix.md) | Current communication paths, runner boundary, and network traffic. |
| [`../architecture-references/runtime-security-boundaries.md`](../architecture-references/runtime-security-boundaries.md) | Runtime identities, Docker socket risk, and implemented service boundaries. |
| [`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md) | Implemented functional networks and service placement. |
| [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) | Manifest-first artifact handoff contract and open artifact gaps. |
| [`../current-runtime-and-operations/repository-structure.md`](../current-runtime-and-operations/repository-structure.md) | DAG placement rules and the `docker/dev` versus `docker/prod` split. |

## Implemented runner boundary

The production-like runtime includes an internal `job-runner-api` FastAPI service.
It exposes:

- `GET /health` for service health;
- `POST /jobs` for one typed ML step request;
- `GET /jobs/{job_id}` for current job status.

The active request and status contracts live under `src/ml/jobs`. Accepted job
types are `ingest`, `features`, and `models`. The runner does not expose a
pipeline-wide runtime job.

Submitted jobs are stored in memory and delegated synchronously to internal ML
step services. Successful jobs return `JobResult` evidence with output paths,
optional metrics evidence, and optional artifact manifest references. Controlled
step failures are mapped to structured `JobError` payloads.

The service remains intentionally narrow. It is not a durable queue, worker pool,
distributed execution platform, Docker SDK wrapper, Kubernetes controller, or
full-pipeline scheduler.

## Implemented production-like orchestration path

Airflow owns pipeline orchestration.

The runner owns execution control for one allow-listed typed ML step at a time.
The production-like execution path uses visible Airflow steps:

1. submit and observe an ingest job;
2. submit and observe a feature engineering job;
3. submit and observe a model training and prediction job;
4. refresh the API after successful model jobs.

This keeps the ML chain observable and retryable from Airflow without giving
Airflow Docker runtime control.

## Development execution model

The current local development model still uses Airflow as both orchestrator and
container launcher. This model is practical for local development because it
reuses existing Docker images and keeps artifacts visible on the host. It is not
the production-like job boundary and should remain dev-only.

## Implementation progress

| Capability | Status | Source of truth |
| ---------- | ------ | --------------- |
| Artifact manifest models and store | Implemented | [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) |
| Local ML manifest emission | Implemented | [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) |
| Typed step job requests and statuses | Implemented | `src/ml/jobs/` |
| Internal runner API boundary | Implemented | `src/job_runner/` and runtime docs |
| Internal ML step services | Implemented | `src/ml/services/` and runtime docs |
| Production-like Airflow DAG using step runner jobs | Implemented | `docker/prod/airflow/dags/` and runtime docs |
| API serving from promoted manifests | Open | [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) |
| Production-like smoke validation | Open | Active validation work |
| Runtime configuration and secret validation | Open | Active runtime hardening work |

## Open design gaps

- The API still needs to serve predictions through promoted manifests.
- Runner metrics and Prometheus scrape integration are not implemented.
- Job status is in memory and is not durable across process restarts.
- Configuration validation still needs to reject unsafe placeholder values for
  custom services.
- Production-like smoke validation still needs to cover the complete Airflow,
  runner, ML step service, artifact, and API refresh chain.

## Validation target

A complete validation should prove that:

- `docker/prod` Airflow has no Docker socket mount;
- Airflow can submit typed step jobs to `job-runner-api`;
- the runner can delegate each ML step without exposing Docker runtime control to
  Airflow;
- each ML step emits coherent artifact manifests;
- Airflow chains ingest, features, and models visibly;
- the API refresh runs only after successful model jobs;
- Prometheus/Grafana can observe job status and artifact freshness.
