# Phase 8 production minimal target

This document coordinates the end-of-phase target for Phase 8. It is a
`next-phase-design/` document: it describes the remaining implementation target,
not the state already implemented on `main`.

## Objective

Make `docker/prod` minimally operational without an Airflow Docker socket, while
progressively removing implicit local path discovery through a versioned local
manifest that can reference either the production-like filesystem or, later,
MinIO-compatible object storage.

The end of Phase 8 should prove this minimal production-like path:

```text
Airflow DAG tasks
  -> job-runner-api
  -> ml-ingest-prod / ml-features-prod / ml-models-prod FastAPI services
  -> promoted artifact manifests under docker/prod/runtime/artifacts
  -> api-dev refresh and prediction serving from promoted manifests
```

## Target execution boundary

Airflow remains the pipeline orchestrator. It chooses counters, dates, retries,
and task dependencies.

`job-runner-api` remains the narrow execution control boundary. It accepts one
allow-listed typed step job at a time, records status, and maps failures to a
structured `JobStatus`. It must not become a Docker SDK wrapper, a shell command
runner, a durable queue, a worker pool, or a full-pipeline scheduler.

The ML services own the concrete step execution:

| Service | Target role | Required network scope |
| ------- | ----------- | ---------------------- |
| `ml-ingest-prod` | Execute one ingest request through the existing ingest CLI adapter and emit an interim dataset manifest. | `pipeline_runtime_net` |
| `ml-features-prod` | Execute one feature request through the existing feature CLI adapter and emit a feature dataset manifest. | `pipeline_runtime_net` |
| `ml-models-prod` | Execute one model request through the existing model CLI adapter, log MLflow evidence, and emit a prediction manifest. | `pipeline_runtime_net`, `tracking_client_net` |

The runner calls these services through Compose DNS on `pipeline_runtime_net`.
Airflow only calls the runner API and never asks Docker to start a container.

## Manifest-first handoff

The promoted manifest remains the release contract for served artifacts.
Consumers must not rediscover the current prediction by scanning local folders or
using implicit `latest` conventions.

For the end of Phase 8:

- `primary_backend="local"` remains the first functional backend;
- local payloads stay under `docker/prod/runtime`;
- `object_uri` can record optional `s3://` metadata;
- MinIO upload, download, object checksum verification, and object-storage-first
  API serving remain out of scope.

## Story mapping

| Issue | Scope |
| ----- | ----- |
| #70 | Add the ML step FastAPI services, switch the runner to HTTP service calls, and add the production-like Airflow DAG path. |
| #71 | Make the prediction API manifest-aware and harden the runtime configuration needed by manifest serving. |
| #72 | Add deterministic production-like smoke validation, realistic fixtures, negative-path checks, and final documentation cleanup. |

Closed or removed Phase 8 configuration and test-debt stories are intentionally
absorbed into those three implementation stories to keep the phase focused on one
minimal operational production path.

## Documentation movement rule

While this target is not fully implemented, keep it in `next-phase-design/`.

When each slice becomes stable:

1. move implemented operation details to
   `docs/current-runtime-and-operations/local-prod-runtime.md`;
2. move implemented communication, network, and security boundaries to
   `docs/architecture-references/`;
3. keep only still-open gaps in `next-phase-design/`.

## Out of scope for Phase 8 closure

The Phase 8 closure target does not include:

- object-storage-first serving;
- production secret management;
- distributed or durable runner queues;
- Kubernetes or remote deployment validation;
- full performance, load, or regression testing;
- replacing the development DockerOperator path.
