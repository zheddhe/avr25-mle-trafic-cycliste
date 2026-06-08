# ML step runner boundary

This note records the Phase 8 architecture correction and the implemented
runner execution boundary for typed ML steps.

## Decision

Airflow owns pipeline orchestration.

The internal runner owns execution of one typed ML step at a time. It does not
own the whole business pipeline as a single opaque runtime job.

The active production-like path is based on explicit step-level jobs:

- ingest job;
- feature engineering job;
- model training and prediction job.

Airflow can then see, retry, and report each part of the ML chain without using
DockerOperator or mounting the Docker socket.

## Why this boundary matters

The first runner skeleton introduced a valid internal API boundary, but the
execution design avoids two failure modes:

1. making the runner a broad ML service that embeds ingestion, feature, and model
   dependencies in one container;
2. building a local scheduler that behaves like a lightweight Kubernetes clone
   before the project needs queues, worker pools, or dynamic pods.

The production-like goal is narrower: remove Docker socket access from Airflow
while preserving Airflow as the orchestrator.

## Responsibilities

Airflow prod DAG:

- select counters and derive ranges;
- build typed step requests;
- order ingest, features, and model tasks;
- apply retries;
- refresh the API only after successful model output promotion.

`job-runner-api`:

- accept one allow-listed typed ML step request;
- expose status;
- map failures to structured job errors;
- execute the step through a runner adapter;
- return step-level result evidence.

ML step runtime:

- execute the business code for exactly one step;
- keep step-specific image, dependency, mount, metric, and manifest behavior.

Artifact manifests carry the handoff evidence between steps and toward the
serving API. The API serves promoted prediction artifacts from manifests without
knowing runner or training internals.

## Contract direction

The active ML job contracts live under:

```text
src/ml/jobs/
├── __init__.py
├── contracts.py
└── status.py
```

The active runner contract no longer exposes a full-pipeline request. It accepts
only the `ingest`, `features`, and `models` job types. The previous
`src/pipeline/contracts` namespace is removed from the active path instead of
being kept as a compatibility layer.

## Runner execution direction

The runner supports step-level execution:

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

The current service executes the submitted step synchronously in the internal
runner process and records the `queued`, `running`, `succeeded`, and `failed`
state transitions in its in-memory state store. This keeps the runner small
enough to replace later with Kubernetes Jobs, a queue-backed worker pool, or
another execution backend.

Successful jobs return output path evidence, optional metrics evidence, and
manifest references derived from the typed job request and configured manifest
root. Controlled ML step failures are mapped to structured `JobError` payloads.

## Story impact

The remaining Phase 8 work should be interpreted as follows:

- #69 implements runner-controlled execution for one typed ML step at a time.
  The active contract does not expose a pipeline-wide runtime job.
- #70 implements the production-like Airflow DAG that chains ingest, features,
  and model jobs.
- #71 keeps artifact-aware API serving independent from runner internals.
- #72 validates the visible Airflow step chain, runner step execution,
  manifests, API refresh, and no Docker socket access.
- #73 validates configuration close to each service after runner/API boundaries
  are stable.
- #81 keeps realistic fixtures focused on step-level runner execution and
  Airflow orchestration visibility.

## Out of scope

This decision does not introduce queues, worker pools, dynamic container
creation, Docker SDK usage, or Kubernetes. It also does not implement the Airflow
DAG chain, which belongs to story #70.
