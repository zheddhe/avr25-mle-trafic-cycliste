# Execution and artifact promotion contract

This document describes the implemented execution and artifact-promotion
behaviour shared by the local `docker/dev` and `docker/prod` runtimes.

It is a current-state architecture reference. Future work beyond the validated
local baseline is tracked in `docs/remaining-work/global-remaining-work.md`.

## Scope

The contract covers the stable execution boundary, bounded local multi-counter
execution, manifest-backed handoff, promotion safety, traceability, and the
limits of the validated local baseline.

It does not claim distributed workers, a remote workload runtime, or remote
production operations are implemented.

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

## Local bounded scale-out

The local runtime supports bounded multi-counter execution. Airflow reads explicit
orchestrator and child-DAG concurrency limits from
`docker/*/airflow/config/bike_dag_config.json`; these limits bound active DAG
runs and tasks while preserving each counter's pipeline order.

The runner also bounds service dispatch through `JOB_RUNNER_MAX_IN_FLIGHT_JOBS`.
Its default is two in-flight jobs, enforced by a bounded semaphore. ML service
replica counts remain explicit runtime configuration, not an implicit scaling
behaviour.

For one counter and one logical run, the active workflow preserves its required
step order. A daily run fails safely when its required initialization state is
missing or invalid. Phase 10 will evolve that precondition into a dataset
readiness gate without weakening the failure-safety requirement.

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

These guarantees protect the local manifest-first serving path and allow
independent counter scopes to progress without cross-counter overwrite.

## Ordering and failure boundaries

Model promotion precedes the API refresh that loads the promoted prediction
artifact. A refresh failure must not invalidate the last successfully loaded API
artifact state.

Runner and ML-service evidence includes the identifiers needed to trace a job
through the local runtime:

- `job_type`;
- `counter_id`;
- `run_id` or trace identifier;
- `job_id`;
- `metrics_reference`;
- service-instance log location when available.

## Beyond the local baseline

The following are not part of the validated local contract and belong to global
remaining work:

- distributed orchestration, remote workers, or durable distributed queues;
- resource and capacity planning beyond the local Compose runtime;
- remote deployment, production ingress, and production operational SLOs;
- object-storage-first artifact serving.

## Related documentation

- [`runtime-communication-matrix.md`](runtime-communication-matrix.md)
- [`../current-runtime-and-operations/runtime-logging.md`](../current-runtime-and-operations/runtime-logging.md)
- [`../remaining-work/global-remaining-work.md`](../remaining-work/global-remaining-work.md)
