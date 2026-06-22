# Execution and artifact promotion contract

This document describes the implemented execution and artifact-promotion
behaviour shared by the local `docker/dev` and `docker/prod` runtimes.

It is a current-state architecture reference. Planned bounded scale-out work is
tracked separately under `docs/remaining-work/`.

## Scope

The contract covers the stable execution boundary, manifest-backed handoff,
promotion safety, traceability, and the limits of the validated local baseline.

It does not claim that multi-counter parallel execution, distributed workers, or
remote production operations are implemented.

## Implemented execution path

```text
Airflow DAG task
  -> job-runner-api
  -> ml-gateway
  -> ML step service
  -> run-scoped artifact manifest
  -> promoted current.json
  -> authenticated API refresh
  -> FastAPI serving from the promoted payload
```

Airflow owns DAG ordering and scheduling. `job-runner-api` accepts allow-listed,
typed job submissions and dispatches them through `ml-gateway`. The runner does
not execute arbitrary shell commands, orchestrate Docker, or discover the latest
output by scanning runtime directories.

`ml-gateway` provides stable typed routes to ingestion, feature, and model
services. This keeps runner configuration independent from service replica DNS.

## Artifact handoff and promotion

ML step services write payloads and run-scoped manifests under the artifact
repository root. Prediction serving resolves the promoted manifest referenced by
`predictions/<counter_id>/current.json`; it does not infer the current payload by
scanning `data/final` or another runtime folder.

The local manifest store guarantees the following for a single artifact type and
counter scope:

- `current.json` is updated through atomic write-and-replace behaviour;
- readers do not observe partial promoted-manifest content;
- promotion paths are counter-scoped;
- a failed promotion preserves the previous valid `current.json` when one exists;
- promotion of identical manifest content is idempotent;
- payload checksum evidence remains associated with the promoted manifest;
- same-scope promotion is serialized by the lock scope
  `<artifact_type>/<counter_id>`.

These guarantees protect the current local manifest-first serving path. They do
not by themselves prove safe concurrent execution for all counters or all API
refreshes.

## Ordering and failure boundaries

For the current single logical counter flow, model promotion precedes the API
refresh that loads the promoted prediction artifact. A refresh failure must not
invalidate the last successfully loaded API artifact state.

Runner and ML-service evidence includes the identifiers needed to trace a job
through the local runtime:

- `job_type`;
- `counter_id`;
- `run_id` or trace identifier;
- `job_id`;
- `metrics_reference`;
- service-instance log location when available.

## Relationship to future work

The current runtime has the execution and artifact-safety primitives required to
start bounded scale-out work. It has not yet validated the following as a
production-like contract:

- bounded multi-counter fan-out;
- explicit Airflow, runner, and service backpressure limits;
- resource behaviour under concurrent workloads;
- per-counter parallel API refresh consistency;
- scale-out acceptance evidence and operational dashboards.

Those gaps are owned by
[`../remaining-work/phase-9-bounded-scale-out-contract.md`](../remaining-work/phase-9-bounded-scale-out-contract.md).

## Related documentation

- [`runtime-communication-matrix.md`](runtime-communication-matrix.md)
- [`../current-runtime-and-operations/runtime-logging.md`](../current-runtime-and-operations/runtime-logging.md)
- [`../remaining-work/global-remaining-work.md`](../remaining-work/global-remaining-work.md)
