# Phase 9 bounded scale-out execution contract

This document defines the Phase 9 scale-out contract. It records the rules that
must remain true while bounded local parallelism is implemented and validated.

## Scope

Phase 9 targets bounded parallel execution for the local `docker/prod` runtime,
while keeping `docker/dev` structurally comparable for debugging and demos.

The intended outcome is controlled multi-counter execution with explicit
concurrency, retry, resource-safety, traceability, and manifest-promotion rules.

## Out of scope

The following topics remain outside Phase 9 unless a later story changes scope:

- Kubernetes or another remote workload runtime;
- durable distributed queues or distributed worker pools;
- remote deployment and production operations;
- object-storage-first artifact serving;
- full ETL source-chain redesign;
- production security hardening beyond preserving existing runtime boundaries.

## Current safe baseline

The current local runtime keeps the execution boundary narrow:

- Airflow owns DAG ordering and counter fan-out decisions;
- Airflow submits typed step jobs to the internal `job-runner-api`;
- the runner dispatches allow-listed jobs through `ml-gateway`;
- ML step services execute one typed job request at a time per service process;
- API refresh happens after successful model promotion;
- serving reads promoted prediction manifests through `current.json`.

This baseline is the rollback point for Phase 9 work. Increasing parallelism must
not bypass typed contracts, gateway routing, manifest promotion safety, or the
current Docker socket boundary.

## Target bounded concurrency model

Bounded parallelism can be enabled when the following behavior is documented and
testable:

- orchestrator fan-out across counters is bounded by configuration;
- child DAG concurrency is bounded by configuration;
- per-counter step order remains deterministic;
- runner in-flight service dispatch is bounded;
- ML service replica counts are explicit and conservative by default;
- API refresh happens only after the relevant model promotion succeeds;
- runtime logs and metrics expose enough evidence to diagnose slow or failed jobs.

The target is local bounded parallelism, not unbounded throughput.

## Per-counter ordering rules

For one counter and one logical run, the task order remains:

```text
init gate, when applicable
  -> ingest
  -> features
  -> models
  -> API refresh
```

Parallel execution must not reorder steps within a single counter run.

A daily run must not bypass the init requirement for the same counter. If init
state is missing or invalid, daily execution must fail safely instead of creating
partial artifacts.

## Serialization rules

The following operations must remain serialized:

- promotion of the same artifact type for the same counter;
- updates to the same promoted `current.json` path;
- init marker updates for the same counter;
- API refresh for the same counter when it depends on a newly promoted model.

The following operations may become parallel when resource limits allow it:

- execution for different counters;
- writes to independent counter-scoped artifact paths;
- API refreshes for different counters when API state remains consistent;
- ML step service calls that do not share unsafe mutable state.

## Manifest promotion invariants

Manifest promotion must preserve these invariants:

- there is one promoted `current.json` per artifact type and counter;
- readers must not observe partial `current.json` content;
- a failed promotion leaves the previous valid manifest intact when one exists;
- retries are deterministic for the same run inputs;
- cross-counter overwrites are impossible by construction;
- promoted manifest paths remain counter-scoped;
- payload checksum evidence remains valid after promotion.

The local filesystem manifest store implements these invariants with atomic
write-and-replace behavior and a scope-local promotion lock. The lock scope is
`<artifact_type>/<counter_id>`, which serializes same-counter promotion without
blocking independent counters unnecessarily.

## Failure and retry contract

Runner, ML service, Airflow, and API refresh failures must remain explicit and
traceable.

A failed job should provide enough evidence to identify:

- `job_type`;
- `counter_id`;
- `run_id` or trace id;
- `job_id`;
- service instance log file or `metrics_reference`;
- whether retry is safe;
- manifest or payload path affected, when applicable.

Retries must not corrupt promoted artifacts. Retrying a completed run should be
safe when inputs and paths are identical, or explicitly rejected with a clear
error.

API refresh failures must not invalidate the last successfully loaded artifact
state.

## Resource-safety contract

Parallelism must be bounded by explicit configuration. Defaults must remain safe
for a local development laptop.

Implementation stories should define or preserve:

- maximum active parent DAG runs;
- maximum active child DAG runs;
- maximum active tasks per child DAG;
- maximum runner in-flight jobs;
- ML step service timeout behavior;
- backpressure behavior when capacity is exhausted;
- monitoring signals for queued, running, succeeded, and failed jobs.

No story should introduce unbounded mapped task expansion or unbounded runner
submissions.

## Implemented prerequisites

The following prerequisites are implemented in the local runtime baseline:

- manifest promotion is atomic from the API reader perspective;
- run-scoped manifests and promoted `current.json` files remain counter-scoped;
- same artifact type and counter promotions are serialized with a scope-local
  lock;
- repeated promotion of identical manifest content is idempotent;
- failed promoted-manifest writes preserve the previous valid `current.json`;
- manifest-backed API loading keeps serving the latest promoted artifact;
- runner and ML service logs expose `run_id`, `job_id`, `job_type`, `counter_id`,
  and `metrics_reference`.

## Remaining implementation stories

Remaining implementation work includes:

- bounded multi-counter parallel execution;
- scale-out acceptance validation and operational observability;
- stronger API refresh consistency if refreshes become per-counter parallel;
- explicit resource limits and backpressure behavior for local scale-out.

## Expected file and test impact

Remaining implementation stories may update:

- `docker/prod/airflow/dags/`;
- `docker/prod/airflow/config/bike_dag_config.json`;
- `src/job_runner/`;
- `src/ml/` manifest emission callers when concurrency semantics change;
- `src/api/` refresh behavior when needed;
- `tests/unit/` for contract-level concurrency and idempotency checks;
- `tests/integration/` for runner and manifest behavior;
- `tests/acceptance/` for production-like scale-out validation.

Current runtime and architecture documentation should only be updated after the
related behavior is implemented and validated.
