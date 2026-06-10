# Phase 9 bounded scale-out execution contract

This document defines the Phase 9 scale-out contract. It mixes current-state
notes for implemented Phase 9 slices with future-state rules that must remain in
force before bounded runtime parallelism is enabled.

## Scope

Phase 9 targets bounded parallel execution for the local production-like
`docker/prod` runtime.

The intended outcome is controlled multi-counter execution with explicit
concurrency, retry, resource-safety, and manifest-promotion rules.

The current production-like orchestration baseline remains sequential until the
implementation stories change it. Manifest promotion concurrency-safety is now an
implemented prerequisite for later bounded parallelism.

## Out of scope

The following topics remain outside Phase 9 unless a later story explicitly
changes the scope:

- Kubernetes or another remote workload runtime;
- durable distributed queues or distributed worker pools;
- remote deployment and production operations;
- object-storage-first artifact serving;
- full ETL source-chain redesign;
- production security hardening beyond preserving existing runtime boundaries.

## Current sequential baseline

The current local production-like runtime intentionally keeps orchestration and
runner execution narrow:

- the parent orchestrator DAG runs one active run at a time;
- init and daily child DAGs run one active run and one active task at a time;
- Airflow submits typed step jobs to the internal `job-runner-api`;
- the runner delegates synchronously to allow-listed ML step services;
- the runner serializes ML step service calls inside the local process;
- API refresh happens after a successful model step;
- serving reads promoted prediction manifests through `current.json`.

This baseline is the safe rollback point for Phase 9 work.

## Target bounded concurrency model

Later Phase 9 implementation stories may enable bounded parallelism when the
following model is documented and testable:

- orchestrator fan-out across counters is bounded by configuration;
- child init and daily DAG concurrency is bounded by configuration;
- per-counter execution order remains deterministic;
- ML step service calls remain typed, allow-listed, and observable;
- API refresh happens only after the relevant counter model promotion succeeds;
- default limits remain conservative for local developer machines.

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
state is missing or invalid, daily execution must fail safely instead of
creating partial artifacts.

## Serialization rules

The following operations must remain serialized:

- promotion of the same artifact type for the same counter;
- updates to the same promoted `current.json` path;
- init marker updates for the same counter;
- API refresh for the same counter when it depends on a newly promoted model.

The following operations may become parallel when resource limits allow it:

- execution for different counters;
- writes to independent counter-scoped artifact paths;
- API refreshes for different counters when the API state remains consistent;
- ML step service calls that do not share unsafe mutable state.

## Manifest promotion invariants

Manifest promotion must preserve the following invariants:

- there is one promoted `current.json` per artifact type and counter;
- readers must not observe partial `current.json` content;
- a failed promotion leaves the previous valid manifest intact when one exists;
- retries are deterministic for the same run inputs;
- cross-counter overwrites are impossible by construction;
- promoted manifest paths remain counter-scoped;
- payload checksum evidence remains valid after promotion.

The local filesystem manifest store now implements these invariants with atomic
write-and-replace behavior and a scope-local promotion lock. The lock scope is
`<artifact_type>/<counter_id>`, which serializes same-counter promotion without
blocking independent counters unnecessarily.

## Failure and retry contract

Runner, ML service, Airflow, and API refresh failures must remain explicit and
traceable.

A failed job should provide enough evidence to identify:

- the job type;
- the counter id;
- the logical run directory or run id;
- the failed step;
- whether retry is safe;
- the manifest or payload path affected, when applicable.

Retries must not corrupt promoted artifacts. Retrying a completed run should be
safe when inputs and paths are identical, or explicitly rejected with a clear
error.

API refresh failures must not invalidate the last successfully loaded artifact
state.

## Resource-safety contract

Parallelism must be bounded by explicit configuration. Defaults must remain safe
for a local development laptop.

Phase 9 implementation stories should define:

- maximum active parent DAG runs;
- maximum active child DAG runs;
- maximum active tasks per child DAG;
- maximum runner in-flight jobs;
- ML step service timeout behavior;
- backpressure behavior when capacity is exhausted;
- monitoring signals for queued, running, succeeded, and failed jobs.

No story should introduce unbounded mapped task expansion or unbounded runner
submissions.

## Implemented Phase 9 slices

The following Phase 9 prerequisite is implemented in the local manifest store:

- manifest promotion is atomic from the API reader perspective;
- run-scoped manifests and promoted `current.json` files remain counter-scoped;
- same artifact type and counter promotions are serialized with a scope-local
  lock;
- repeated promotion of identical manifest content is idempotent;
- failed promoted-manifest writes preserve the previous valid `current.json`;
- manifest-backed API loading keeps serving the latest promoted artifact.

## Remaining implementation stories

This contract does not enable bounded parallel execution by itself. Remaining
implementation work includes:

- bounded multi-counter parallel execution;
- scale-out acceptance validation and operational observability;
- stronger API refresh consistency if refreshes become per-counter parallel;
- explicit resource limits and backpressure behavior for local scale-out.

## Expected file and test impact

The remaining implementation stories may update:

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
