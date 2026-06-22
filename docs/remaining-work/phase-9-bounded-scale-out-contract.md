# Phase 9 — Bounded scale-out execution

> **Status**: Future-state execution target. The current execution and
> artifact-promotion contract is documented in
> [`../architecture-references/execution-and-artifact-promotion-contract.md`](../architecture-references/execution-and-artifact-promotion-contract.md).

## Purpose

Phase 9 defines the additional behaviour that must be implemented and validated
before the local runtime can claim safe bounded multi-counter parallelism.

It does not redefine the current runner, gateway, manifest-promotion, or API
serving guarantees. Those current-state rules are architecture references and
remain mandatory compatibility constraints for Phase 9 work.

## Scope

Phase 9 targets bounded parallel execution for the local `docker/prod` runtime,
while keeping `docker/dev` structurally comparable for debugging and demos.

The intended outcome is controlled multi-counter execution with explicit
concurrency, retry, resource-safety, traceability, and operational evidence.

## Out of scope

The following topics remain outside Phase 9 unless a later story changes scope:

- Kubernetes or another remote workload runtime;
- durable distributed queues or distributed worker pools;
- remote deployment and production operations;
- object-storage-first artifact serving;
- full ETL source-chain redesign;
- production security hardening beyond preserving existing runtime boundaries.

## Required compatibility with the current architecture

Phase 9 must preserve the current execution and artifact-promotion contract:

- Airflow continues to own DAG ordering and fan-out decisions;
- typed jobs continue to cross `job-runner-api` and `ml-gateway`;
- promoted artifacts remain manifest-first and counter-scoped;
- same-scope `current.json` promotion remains atomic and serialized;
- an API refresh failure does not invalidate the last successfully loaded
  artifact state;
- runner and ML-service trace identifiers remain available for diagnosis.

## Target bounded concurrency model

Bounded parallelism can be enabled only when the following behaviour is
implemented, configured, and tested:

- orchestrator fan-out across counters is bounded by configuration;
- child DAG concurrency is bounded by configuration;
- per-counter step order remains deterministic;
- runner in-flight service dispatch is bounded;
- ML service replica counts are explicit and conservative by default;
- API refresh happens only after the relevant model promotion succeeds;
- logs and metrics provide evidence for queued, running, succeeded, retried, and
  failed jobs.

The target is local bounded parallelism, not unbounded throughput.

## Per-counter ordering and readiness

For one counter and one logical run, parallel execution must not reorder the
pipeline steps required by the active dataset and model workflow.

Until Phase 10 changes the source chain, a daily run must still fail safely when
the required initialization state is missing or invalid.

After Phase 10 is implemented, this becomes a dataset-readiness gate: a daily
run must require a valid canonical dataset state and must not publish partial
artifacts when bootstrap, source validation, or current-window reconstruction
has failed.

## Serialization and isolation objectives

The following must remain serialized or equivalently protected by a documented
concurrency control mechanism:

- promotion of the same artifact type for the same counter;
- updates to the same promoted `current.json` path;
- readiness-state updates for the same counter;
- API refresh for the same counter when it depends on a newly promoted model.

The following may become parallel only after capacity and isolation are proven:

- execution for different counters;
- writes to independent counter-scoped artifact paths;
- API refreshes for different counters when API state remains consistent;
- ML step service calls that do not share unsafe mutable state.

## Resource-safety and backpressure objectives

Parallelism must be bounded by explicit configuration. Defaults must remain safe
for a local development laptop.

Implementation stories must define and validate:

- maximum active parent DAG runs;
- maximum active child DAG runs;
- maximum active tasks per child DAG;
- maximum runner in-flight jobs;
- ML step service timeout and retry behaviour;
- backpressure behaviour when capacity is exhausted;
- resource limits per service family;
- monitoring signals for queueing, running, success, retry, failure, and slow
  jobs.

No story may introduce unbounded mapped-task expansion or unbounded runner
submissions.

## Acceptance evidence

The bounded scale-out claim requires production-like acceptance evidence showing
at least:

- concurrent execution for multiple counters within configured bounds;
- deterministic ordering for each counter;
- preservation of counter-scoped manifest promotion;
- safe retry behaviour without promoted-artifact corruption;
- clear capacity-exhaustion and timeout behaviour;
- observable job correlation across Airflow, runner, ML services, and API
  refresh.

## Expected file and test impact

Remaining implementation stories may update:

- `docker/prod/airflow/dags/`;
- `docker/prod/airflow/config/bike_dag_config.json`;
- `src/job_runner/`;
- `src/ml/` manifest emission callers when concurrency semantics change;
- `src/api/` refresh behaviour when needed;
- `tests/unit/` for concurrency, idempotency, and readiness contracts;
- `tests/integration/` for runner, gateway, and manifest behaviour;
- `tests/acceptance/` for production-like scale-out validation.

Stable behaviour becomes current documentation only after it is implemented and
validated.
