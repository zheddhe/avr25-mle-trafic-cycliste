# Hybrid artifact handoff strategy

This document coordinates the active artifact handoff design for generated ML
artifacts. It defines how artifacts are described, promoted, and consumed through
explicit manifests instead of implicit latest-file discovery.

Current runtime and architecture placement are documented in
`current-runtime-and-operations/` and `architecture-references/`. This document
keeps the consolidated handoff contract, current implemented coverage, and open
artifact-specific gaps.

## Decision summary

The project uses a hybrid manifest-first strategy:

1. The promotion manifest is the authority for served artifacts.
2. `docker/prod/runtime` remains the first local production-like backend.
3. Optional MinIO object URIs can be recorded when an artifact is also available
   through S3-compatible object storage.
4. The API, Airflow, and runner path must consume manifests instead of
   discovering files through implicit `latest` conventions.
5. Development and DVC workspaces stay separate from production-like generated
   runtime artifacts.

This hybrid decision keeps the active phase incremental. The project can validate
the production-like runtime with local bind-mounted artifacts first, while keeping
a stable contract that can later point to MinIO without changing every consumer.

## Current coverage summary

Implemented:

- strict artifact manifest models under `src/artifacts`;
- local checksum, manifest write, read, and promotion helpers;
- local manifest emission wrappers for ingest, features, and model jobs;
- typed ML step request and status contracts under `src/ml/jobs`;
- internal runner API boundary and synchronous typed step execution under
  `src/job_runner`;
- internal ML step services under `src/ml/services`;
- production-like runtime mounts for generated data and artifact manifests;
- production-like Airflow services without `/var/run/docker.sock`;
- production-like Airflow DAGs chaining runner-backed `ingest`, `features`, and
  `models` steps before authenticated API refresh.

Open artifact handoff gaps:

- API serving from promoted prediction manifests;
- production-like smoke validation using realistic test fixtures;
- configuration and placeholder hardening;
- optional MinIO upload, download, and checksum verification helpers.

The MinIO part is contract-compatible today through optional `s3://` object URIs
and through MLflow artifact storage. It is not yet the primary serving backend for
API prediction payloads.

## Source documents

| Source | Use in this strategy |
| ------ | -------------------- |
| [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md) | Runtime usage, runner API behavior, production-like DAGs, and `docker/prod/runtime` ownership. |
| [`../current-runtime-and-operations/repository-structure.md`](../current-runtime-and-operations/repository-structure.md) | Repository path ownership and DVC boundary. |
| [`../current-runtime-and-operations/ports-and-services.md`](../current-runtime-and-operations/ports-and-services.md) | Local host exposure and internal-only MinIO policy. |
| [`../current-runtime-and-operations/dependency-strategy.md`](../current-runtime-and-operations/dependency-strategy.md) | MLflow, MinIO, and runtime dependency compatibility. |
| [`../architecture-references/runtime-communication-matrix.md`](../architecture-references/runtime-communication-matrix.md) | Airflow, runner, and ML step service responsibility boundaries. |
| [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) | Remaining runner, observability, and validation gaps. |

## Why manifest-first

The current development flow can tolerate visible folders and manual inspection.
The production-like runtime needs a stricter handoff because implicit filesystem
conventions create MLOps risks.

| Risk | Manifest-first mitigation |
| ---- | ------------------------- |
| Consumers read different files as `latest`. | Consumers read the same promoted manifest. |
| Artifact origin is unclear. | Manifest records producer, run ID, dataset version, and model version. |
| Local and object storage paths drift. | Manifest stores local and optional object references together. |
| API serving is coupled to folder layout. | API serving resolves an explicit promoted artifact reference. |
| Airflow cannot audit what was served. | Airflow records or links the promoted manifest. |

The manifest is not a metadata side file. It is the release contract for one
promoted artifact.

## Implemented artifact package

The reusable artifact package lives under `src/artifacts/` and is
framework-neutral. It does not import Airflow, FastAPI, Docker SDK, MLflow, or
concrete ML pipeline modules.

`src/artifacts/schemas.py` exposes the manifest model and constrained enums for
artifact type, status, storage backend, producer, source, and storage metadata.
`src/artifacts/manifest_store.py` exposes helpers to write, promote, read, and
verify local manifests. `src/artifacts/checksums.py` exposes checksum helpers.

The manifest model validates required fields, local-only manifests with
repository-relative local paths, hybrid manifests with optional `s3://` object
URIs, constrained enum values, checksums when present, timezone-aware timestamps,
and undeclared fields through strict Pydantic configuration.

The first runtime implementation still uses `primary_backend="local"`.
`object_uri` remains optional and only records the S3-compatible reference when
the artifact is also available through MinIO or another object store.

## Workspace boundary

Root `data`, `models`, and `logs` remain development and DVC workspaces. They are
valid for local experiments, DVC reproduction, notebook exploration, and the
`docker/dev` runtime.

`docker/prod/runtime` is the local production-like runtime workspace. It is
ignored by Git, not DVC-managed, and owns generated artifacts produced while
validating `docker/prod`.

| Workspace | Owner | Main use | Promotion role |
| --------- | ----- | -------- | -------------- |
| `data/` | Development and DVC | Reproducible local data stages. | Source or comparison only. |
| `models/` | Development and DVC | Reproducible local model outputs. | Source or comparison only. |
| `logs/` | Development | Local service and batch logs. | No served artifact role. |
| `docker/prod/runtime/data` | Local production-like runtime | Generated runtime data and predictions. | Local promoted artifacts. |
| `docker/prod/runtime/models` | Local production-like runtime | Generated runtime model files. | Local promoted model references. |
| `docker/prod/runtime/logs` | Local production-like runtime | Runtime logs. | Audit evidence only. |
| `docker/prod/runtime/artifacts` | Local production-like runtime | Manifest-first handoff root. | Promoted manifest and payload root. |
| MinIO object storage | MLflow or artifact backend | Optional S3-compatible storage. | Optional object reference. |

## Artifact lifecycle

Artifacts move through five logical states.

| State | Meaning | Typical owner |
| ----- | ------- | ------------- |
| `produced` | A job wrote an output candidate and has enough metadata to describe it. | ML step job. |
| `validated` | Required checks passed, including schema, checksum, and business validation. | ML step job or validation code. |
| `promoted` | A stable manifest pointer now marks the artifact as current. | Promotion helper. |
| `served` | The API has loaded or refreshed from the promoted manifest. | FastAPI service. |
| `archived` | The artifact is retained for audit or rollback but is no longer current. | Artifact store or retention job. |

Only `promoted` artifacts are eligible for default API serving. `served` is an
operational observation, not a replacement for `promoted`.

## Manifest ownership

A manifest is created by the component that has the best evidence about the
artifact:

- ML jobs know the output path, counter, dataset, model, metrics, and checksum
  evidence.
- The artifact store validates manifests and updates the stable promoted pointer.
- A step runner returns manifest references and validates expected outputs
  without fabricating metadata that belongs to ML code.
- Airflow orchestrates job order and records manifest references, but should not
  fabricate artifact metadata that belongs to ML code.
- The API reads the promoted manifest and must not guess file paths when the
  manifest is available.

The first implementation writes a real `current.json` file instead of a symlink.
This is more portable across Windows, WSL, Docker bind mounts, and CI runners.

## Local backend conventions

Local production-like artifacts should stay under `docker/prod/runtime`.

The implemented manifest store writes counter-scoped manifests as:

```text
<manifest_root>/<artifact_type>/<counter_id>/<run_id>/manifest.json
<manifest_root>/<artifact_type>/<counter_id>/current.json
```

With the production-like Compose configuration, `manifest_root` is
`/app/artifacts/manifests` inside ML containers and maps to
`docker/prod/runtime/artifacts/manifests` on the host.

Local paths stored in manifests should be repository-relative when possible.

## Optional MinIO object URI conventions

MinIO is optional in the handoff contract. A manifest can be local-only or hybrid.
When an artifact is also uploaded to object storage, the manifest should include
an S3-compatible URI.

Recommended URI shape:

```text
s3://<bucket>/artifacts/<artifact_type>/<counter_id>/<run_id>/<file_name>
```

Credentials, endpoint URLs, and access keys must remain runtime configuration and
must not be embedded in manifests.

Current limitation: object URIs are metadata only for the artifact handoff path.
The active phase does not yet implement artifact download, checksum verification
against object storage, or object-storage-first API serving.

## Component interactions

| Component | Responsibility in the contract |
| --------- | ------------------------------ |
| ML jobs | Produce artifacts, compute checksums, emit candidate manifests. |
| Runner | Execute one typed step job at a time, keep job status, and return manifest references. |
| Airflow | Submit step jobs, wait for terminal status, chain the ML workflow, and record manifest references. |
| MLflow | Track runs, metrics, parameters, model metadata, and model artifact payloads. |
| MinIO | Store MLflow artifact payloads and optional future object-backed handoff payloads. |
| API | Read `current.json`, validate metadata, serve the referenced artifact. |
| Prometheus/Grafana | Observe freshness, job status, and current artifact metadata. |

The contract deliberately keeps Airflow out of low-level filesystem discovery,
keeps the runner out of full-pipeline orchestration, and keeps the API out of
training job internals.

## Implementation progress

| Capability | Status |
| ---------- | ------ |
| Define artifact handoff vocabulary. | Done |
| Add strict Pydantic manifest models. | Done |
| Add checksum, write, read, and promotion helpers. | Done |
| Make ML jobs emit local manifests. | Done |
| Add typed step job contracts. | Done |
| Add the internal runner API boundary. | Done |
| Execute typed ML step jobs through the runner. | Done |
| Add production-like Airflow DAG chaining runner steps. | Done |
| Make the API load the promoted local manifest. | Open |
| Add production-like smoke validation. | Open |
| Harden runtime configuration and secrets validation. | Open |
| Strengthen realistic phase test coverage. | Open |
| Add optional object-storage handoff helpers. | Open |
| Switch `primary_backend` to object storage. | Deferred |

## Known technical and functional debt

The remaining work should leave explicit debt where the active phase deliberately
stays incremental:

- realistic integration fixtures derived from the real raw-data schema are still
  needed;
- unit coverage should be expanded for invalid manifests, runner state errors,
  API manifest loading failures, and runtime configuration failures;
- `current.json` discovery should stay explicit and counter-scoped so the API
  does not reintroduce implicit latest-file scanning;
- runner execution should stay typed and allow-listed, not a generic shell
  command runner or full-pipeline scheduler;
- object storage is not yet a primary artifact handoff or serving backend despite
  the optional `object_uri` contract.

## Out of scope for the current implementation wave

The current implementation wave does not implement:

- object-storage-first serving;
- production secret management;
- distributed runner queues;
- remote deployment validation;
- full performance or load testing.
