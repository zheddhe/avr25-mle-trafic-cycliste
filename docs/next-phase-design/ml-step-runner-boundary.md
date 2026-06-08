# ML step runner boundary

This note records the Phase 8 architecture correction before the runner execution
work continues.

## Decision

Airflow owns pipeline orchestration.

The internal runner owns execution of one typed ML step at a time. It should not
own the whole business pipeline as a single opaque runtime job.

The project should stop expanding the `PipelineJobRequest` concept. The active
production-like path should be based on explicit step-level jobs:

- ingest job;
- feature engineering job;
- model training and prediction job.

Airflow can then see, retry, and report each part of the ML chain without using
DockerOperator or mounting the Docker socket.

## Why this boundary matters

The first runner skeleton introduced a valid internal API boundary, but the next
execution design must avoid two failure modes:

1. making the runner a broad ML service that embeds ingestion, feature, and model
   dependencies in one container;
2. building a local scheduler that behaves like a lightweight Kubernetes clone
   before the project needs queues, worker pools, or dynamic pods.

The production-like goal is narrower: remove Docker socket access from Airflow
while preserving Airflow as the orchestrator.

## Responsibilities

| Component | Responsibility |
| --------- | -------------- |
| Airflow prod DAG | Select counters, derive ranges, build typed step requests, order ingest, features, and model tasks, apply retries, and refresh the API only after successful model output promotion. |
| `job-runner-api` | Accept one allow-listed typed ML step request, expose status, map failures to structured job errors, and return step-level result evidence. |
| ML step runtime | Execute the business code for exactly one step with its own image, dependencies, mounts, metrics, and manifest emission behavior. |
| Artifact manifests | Carry the handoff evidence between steps and toward the serving API. |
| API | Serve promoted prediction artifacts from manifests, without knowing runner or training internals. |

## Contract direction

The current contracts under `src/pipeline/contracts` are ML-specific. A future
refactor should move or alias them under an ML-specific namespace such as
`src/ml/contracts` or `src/ml/jobs`, after the documentation and open stories are
aligned.

`PipelineJobRequest` should not be the center of the runtime contract. The next
implementation should either remove it from the active path or keep it only as a
compatibility/deprecation object until callers are migrated to step-level jobs.

## Runner execution direction

The runner should support step-level execution first:

```text
Airflow prod DAG
  -> submit ingest job
  -> wait for terminal status
  -> submit features job
  -> wait for terminal status
  -> submit models job
  -> wait for terminal status
  -> refresh artifact-aware API
```

This keeps orchestration visible in Airflow and keeps the runner small enough to
replace later with Kubernetes Jobs, a queue-backed worker pool, or another
execution backend.

## Story impact

The remaining Phase 8 work should be interpreted as follows:

| Story | Direction after this decision |
| ----- | ----------------------------- |
| #69 | Implement runner-controlled execution for one typed ML step at a time. Do not implement a pipeline-wide runtime job as the main path. |
| #70 | Implement the production-like Airflow DAG as the component that chains ingest, features, and model jobs. |
| #71 | Keep artifact-aware API serving independent from runner internals. |
| #72 | Validate the visible Airflow step chain, runner step execution, manifests, API refresh, and no Docker socket access. |
| #73 | Validate configuration close to each service after runner/API boundaries are stable. |
| #81 | Keep realistic fixtures focused on step-level runner execution and Airflow orchestration visibility. |

## Out of scope

This decision does not implement the refactor from `src/pipeline/contracts` to an
ML-specific namespace. It also does not introduce queues, worker pools, dynamic
container creation, Docker SDK usage, or Kubernetes.
