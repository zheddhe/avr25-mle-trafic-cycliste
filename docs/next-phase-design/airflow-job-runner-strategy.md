# Airflow job runner strategy

This document coordinates the active design for replacing DockerOperator-based ML
execution in the local production-like runtime.

The current runtime and architecture references describe what is implemented and
how it is wired. This document keeps the runner execution design, the open gaps,
and the coordination rules for the work that is still in progress.

## Scope and inputs

| Source | Use in this design |
| ------ | ------------------ |
| [`../README.md`](../README.md) | Documentation hierarchy and level rules. |
| [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md) | Current dev/prod runtime split and runner API operation. |
| [`../architecture-references/runtime-communication-matrix.md`](../architecture-references/runtime-communication-matrix.md) | Current communication paths and network traffic. |
| [`../architecture-references/runtime-security-boundaries.md`](../architecture-references/runtime-security-boundaries.md) | Runtime identities, Docker socket risk, and implemented service boundaries. |
| [`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md) | Implemented functional networks and service placement. |
| [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) | Manifest-first artifact handoff contract and open artifact gaps. |
| [`artifact-manifest-store.md`](artifact-manifest-store.md) | Implemented promotion helpers that runner jobs can call from execution code. |
| [`../current-runtime-and-operations/repository-structure.md`](../current-runtime-and-operations/repository-structure.md) | DAG placement rules and the `docker/dev` versus `docker/prod` split. |

## Current runner boundary

The production-like runtime now includes an internal `job-runner-api` FastAPI
service. It provides the API boundary needed before real execution is added.

Implemented behavior:

- `GET /health` returns a simple service health response;
- `POST /jobs` validates typed job requests from `src/pipeline/contracts/jobs.py`;
- `GET /jobs/{job_id}` returns the current `JobStatus` from
  `src/pipeline/contracts/statuses.py`;
- accepted jobs are stored in memory;
- accepted jobs stay in `queued` state;
- caller-provided `job_id` values are reused when present;
- runner-generated IDs are explicit and deterministic for the in-memory process;
- not-found errors use an explicit structured response;
- the API imports no Airflow, Docker SDK, or concrete container runtime code.

The service is an internal local production-like bridge. It is not a durable
queue, worker pool, or distributed execution platform.

## Decision summary

The preferred local production-like target is a controlled internal job runner
composed of:

1. a small internal job API;
2. a simple job state model;
3. typed ML worker execution paths;
4. an allow-list of supported job types and arguments;
5. explicit artifact handoff and observability contracts.

Airflow should submit and observe jobs. It should not create containers through
the host Docker socket in the production-like runtime.

This target keeps the architecture translatable to Kubernetes Jobs or CronJobs
later without forcing Kubernetes into the local Compose runtime.

Ingestion, feature engineering, and model training must remain separate business
microservices. They may share a common runner protocol, common schemas, and a
common base image pattern, but they should not be collapsed into a single generic
ML service.

## Current Airflow-triggered execution model in development

The current local development model uses Airflow as both orchestrator and
container launcher:

1. Airflow imports variables and connections during `airflow-init`.
2. DAG tasks read Docker image names, network names, MLflow endpoints, MinIO
   credentials, Pushgateway address, and UID/GID values from Airflow variables.
3. The development Airflow worker mounts `/var/run/docker.sock`.
4. The development Airflow worker starts ML containers for ingestion, feature
   engineering, training, and prediction.
5. ML containers read and write shared `data`, `logs`, and `models` paths.
6. Model jobs log run evidence to MLflow and push batch metrics to Pushgateway.
7. Airflow calls the FastAPI admin refresh endpoint after successful runs.

This model is practical for local development because it reuses existing Docker
images and keeps artifacts visible on the host. It is not a production-like job
boundary and should remain dev-only.

## Why broad container-runtime access is not production-like

A container with write access to the Docker socket can ask the host Docker daemon
to create containers, mount host paths, join networks, and access data or secrets
available to the daemon.

The development `airflow-worker` therefore mixes two responsibilities:

- orchestration: schedule tasks, track dependencies, expose retries, and keep DAG state;
- execution control: create runtime containers with host-level Docker privileges.

| Concern | Development behavior | Production-like target behavior |
| ------- | -------------------- | ------------------------------- |
| Privilege boundary | Airflow worker can control the host container runtime. | Airflow can call only a narrow job submission interface. |
| Runtime user | Worker uses a root entrypoint to align Docker socket access. | Airflow and ML workers run without Docker socket access. |
| Network scope | Jobs inherit networks selected by Airflow variables. | Jobs run on predefined functional networks. |
| Command scope | DAG code can construct container commands. | Runner accepts only allow-listed job types and arguments. |
| Artifact scope | Jobs write broad host-mounted folders. | Jobs publish through manifest-first artifact handoff. |
| Observability | DockerOperator status is mixed with container logs. | Runner exposes job state, retries, logs, and metrics. |

## Workload model

The runner must cover the existing ML pipeline shape:

| Job type | Current dev image or action | Main outputs | External dependencies |
| -------- | --------------------------- | ------------ | --------------------- |
| `ingest` | `ml-ingest-dev` | interim data, manifest, and ingest metrics | raw data, runtime data workspace, logs, Pushgateway. |
| `features` | `ml-features-dev` | processed features, manifest, and feature metrics | interim data, runtime data workspace, logs, Pushgateway. |
| `models` | `ml-models-dev` | forecasts, model artifacts, MLflow runs, prediction manifest | processed data, runtime data/model workspace, logs, MLflow, Pushgateway. |
| `pipeline` | coordinated ingest, features, and model steps | coherent multi-step job status and artifact handoff | all stage-specific dependencies. |

The init and daily DAGs should keep their business responsibility: choose
counters, derive ranges or dates, trigger jobs in the right order, and decide
whether downstream refresh is allowed. They should stop owning container runtime
details.

## Target architecture

```mermaid
flowchart LR
    airflow[Airflow DAG tasks] --> job_api[job-runner-api]
    job_api --> state[(job state)]
    job_api --> worker_ingest[ingest execution]
    job_api --> worker_features[features execution]
    job_api --> worker_models[models execution]

    worker_ingest --> artifacts[(artifact manifest store)]
    worker_features --> artifacts
    worker_models --> artifacts
    worker_models --> mlflow[mlflow-server]
    worker_ingest --> pushgateway[monitoring-pushgateway]
    worker_features --> pushgateway
    worker_models --> pushgateway

    airflow --> api[api-dev refresh]
    api --> artifacts
    prometheus[monitoring-prometheus] --> job_api
    prometheus --> pushgateway
```

The diagram shows the runner execution target. The current runtime topology is
kept in [`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md).

## Typed job contracts

The framework-neutral contracts are implemented under:

```text
src/pipeline/contracts/
├── __init__.py
├── jobs.py
└── statuses.py
```

They are Pydantic models used to describe the payloads that Airflow, the runner
API, and typed ML workers exchange. They deliberately do not import Airflow,
Docker SDK, FastAPI application instances, or runner implementation code.

Implemented request contracts include:

- `IngestJobRequest`;
- `FeatureJobRequest`;
- `ModelJobRequest`;
- `PipelineJobRequest`.

Implemented status and result contracts include:

- `JobStatus`;
- `JobResult`;
- `JobError`;
- `MetricsEvidence`.

`PipelineJobRequest` validates that related ingest, features, and model steps
share `run_id`, `counter_id`, and `manifest_root`, and that their handoff paths
match in order: ingest interim output, features input/output, and model input.

## Runner API contract

Airflow submits a typed job request to `job-runner-api`. The request includes:

- `dag_id`;
- `task_id`;
- `run_id`;
- `try_number`;
- `counter_id`;
- `job_type`;
- validated business parameters;
- expected input and output artifact references.

The API validates the request, assigns or reuses a `job_id`, records job state,
and returns a typed `JobStatus`.

The current API keeps state in memory. The execution implementation must keep the
same external contract while adding controlled dispatch and state transitions.

## Worker execution design

The execution path should use typed workers or typed execution adapters:

| Worker or execution path | Accepted jobs | Service-specific constraints |
| ------------------------ | ------------- | ---------------------------- |
| Ingestion execution | `ingest` only | Raw and interim data access, ingest configuration, Pushgateway metrics. |
| Feature execution | `features` only | Interim and processed data access, feature configuration, Pushgateway metrics. |
| Model execution | `models` only | Processed/final data, model artifacts, MLflow, optional MinIO/S3 credentials, Pushgateway metrics. |
| Pipeline execution | `pipeline` only | Coordinated stage execution using the same per-stage constraints. |

Execution paths may share a common job protocol:

- validate the job payload with a typed schema;
- resolve a safe command from an allow-list;
- execute the business entrypoint;
- publish status, logs, metrics, and artifact references.

They should not share one broad runtime configuration. Service-specific
dependencies, environment variables, mounted paths, network attachments,
healthchecks, and resource limits should remain explicit per worker or per job
type.

## Job state model

The shared `JobState` enum currently supports:

| State | Meaning | Airflow behavior |
| ----- | ------- | ---------------- |
| `pending` | Request was created but not yet queued. | Continue polling. |
| `queued` | Job is accepted and waiting for execution. | Continue polling or sensor deferral. |
| `running` | Execution started. | Continue polling and link logs. |
| `succeeded` | Command exited successfully and outputs were published. | Mark task successful. |
| `failed` | Command failed with a controlled error. | Fail the Airflow attempt. |
| `canceled` | Job was canceled by operator or cleanup policy. | Fail or skip according to DAG policy. |
| `expired` | Job exceeded retention or timeout. | Fail with an explicit timeout reason. |

Airflow retries should create distinct external attempts by including
`try_number` in the idempotency key. Re-submitting the same key should return the
existing `job_id` instead of duplicating work.

## Impact on Airflow DAGs

Development DAG responsibility:

- build container commands;
- select Docker images and networks;
- pass MLflow, MinIO, Pushgateway, UID, and GID variables;
- wait for the container exit code;
- call API refresh after successful jobs.

Production-like DAG responsibility:

- build typed business job specs;
- submit jobs to `job-runner-api`;
- wait for runner terminal state;
- map runner failure to Airflow failure;
- call API refresh through the existing HTTP connection;
- preserve DAG-level retry, schedule, and dependency semantics.

DAG code should stay near Airflow runtime assets under `docker/dev` or
`docker/prod` unless the project later decides to package DAGs as importable
application modules. Reusable client or schema logic may live under `src/` with
tests, but deployment-specific DAG wiring should remain close to the Airflow
runtime that consumes it.

## Init and daily DAG trigger flow

### Initial load DAG

1. Read `bike_dag_config.json`.
2. For each configured counter, submit `ingest`.
3. Submit `features` only after the matching ingest job succeeds.
4. Submit `models` only after the matching feature job succeeds.
5. Promote or refresh final prediction data only after required model outputs are promoted.
6. Fail the Airflow run if any required runner job fails.

### Daily DAG

1. Compute the daily range or business window.
2. Submit `ingest` with the daily range and counter configuration.
3. Submit `features` and `models` with the same idempotency context.
4. Promote or refresh final prediction data only after successful model jobs.
5. Keep Airflow retries at the orchestration layer while the runner records every
   external job attempt.

In both flows, Airflow never asks Docker to start a container. It only asks the
runner API to handle typed business jobs.

## Implementation progress

| Capability | Status | Source of truth |
| ---------- | ------ | --------------- |
| Artifact manifest models | Implemented | [`artifact-manifest-models.md`](artifact-manifest-models.md) |
| Manifest write, read, and promotion helpers | Implemented | [`artifact-manifest-store.md`](artifact-manifest-store.md) |
| Local ML manifest emission | Implemented | [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) |
| Typed job requests and statuses | Implemented | `src/pipeline/contracts/` |
| Internal runner API boundary | Implemented skeleton | This document and `src/job_runner/` |
| Typed ML execution through the runner | Open | This document |
| Production-like Airflow DAG using the runner | Open | This document |
| API serving from promoted manifests | Open | [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) |
| Production-like smoke validation | Open | Active validation work |
| Runtime configuration and secret validation | Open | Active runtime hardening work |

## Open design gaps

- The runner does not yet start typed ML workers or service-specific execution
  adapters.
- Job status is in memory and is not durable across process restarts.
- Airflow still needs a production-like DAG path that submits typed runner jobs.
- The API still needs to serve predictions through promoted manifests.
- Runner metrics and Prometheus scrape integration are not implemented.
- Configuration validation still needs to reject unsafe placeholder values for
  custom services.

## Validation target

A complete validation should prove that:

- `docker/prod` Airflow has no Docker socket mount;
- Airflow can submit a typed job to `job-runner-api`;
- the runner can execute the pipeline or a single step without exposing the
  Docker socket to Airflow;
- each ML step emits coherent artifact manifests;
- the API serves predictions by reading the promoted manifest;
- Prometheus/Grafana can observe job status and artifact freshness.

The current runner API boundary is validated with focused API tests covering
health, valid submission, invalid payloads, status retrieval, duplicate job IDs,
and not-found behavior.
