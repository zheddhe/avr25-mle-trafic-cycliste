# Hybrid artifact handoff strategy

This document is the canonical Phase 8 artifact handoff reference. It defines
how generated artifacts are described, promoted, and consumed through explicit
manifests instead of implicit latest-file discovery.

Implementation details are intentionally split into smaller documents:

| Document | Responsibility |
| -------- | -------------- |
| [`artifact-manifest-models.md`](artifact-manifest-models.md) | Implemented Pydantic manifest contract from issue #64. |
| [`artifact-manifest-store.md`](artifact-manifest-store.md) | Implemented checksum, manifest store, and `current.json` promotion helpers from issue #65. |
| [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) | Runner and Airflow execution target for the remaining Phase 8 stories. |

## Decision summary

The project uses a hybrid manifest-first strategy:

1. The promotion manifest is the authority for served artifacts.
2. `docker/prod/runtime` remains the first local production-like backend.
3. Optional MinIO object URIs can be recorded when an artifact is also available
   through S3-compatible object storage.
4. The API, Airflow, and the future runner must consume manifests instead of
   discovering files through implicit `latest` conventions.
5. Development and DVC workspaces stay separate from production-like generated
   runtime artifacts.

This hybrid decision keeps Phase 8 incremental. The project can validate the
production-like runtime with local bind-mounted artifacts first, while keeping a
stable contract that can later point to MinIO without changing every consumer.

## Current coverage summary

The Phase 8 foundation is now implemented, but the production-like execution path
is not complete yet.

Implemented:

- strict artifact manifest models under `src/artifacts`;
- local manifest store helpers with checksum verification and atomic
  `current.json` replacement;
- local manifest emission wrappers for ingest, features, and model jobs;
- typed pipeline request and status contracts under `src/pipeline/contracts`;
- production-like runtime mounts for generated data and artifact manifests;
- production-like Airflow services without `/var/run/docker.sock`.

Remaining before the Phase 8 goal is operational end-to-end:

- internal `job-runner-api` skeleton;
- runner execution of typed ML jobs;
- production-like Airflow DAG path calling the runner;
- API serving from promoted prediction manifests;
- production-like smoke validation using realistic test fixtures;
- configuration and placeholder hardening.

The MinIO part is contract-compatible today through optional `s3://` object URIs.
It is not yet a functional upload, download, verification, or primary serving
backend.

## Source documents

| Source | Use in this strategy |
| ------ | -------------------- |
| [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md) | Runtime usage and `docker/prod/runtime` ownership. |
| [`../current-runtime-and-operations/repository-structure.md`](../current-runtime-and-operations/repository-structure.md) | Repository path ownership and DVC boundary. |
| [`../current-runtime-and-operations/ports-and-services.md`](../current-runtime-and-operations/ports-and-services.md) | Local host exposure and internal-only MinIO policy. |
| [`../current-runtime-and-operations/dependency-strategy.md`](../current-runtime-and-operations/dependency-strategy.md) | MLflow, MinIO, and runtime dependency compatibility. |
| [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) | Target runner and worker-pool execution model. |

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
| `produced` | A job wrote an output candidate and has enough metadata to describe it. | ML job or worker. |
| `validated` | Required checks passed, including schema, checksum, and business validation. | ML job, worker, or validation step. |
| `promoted` | A stable manifest pointer now marks the artifact as current. | Promotion helper or runner. |
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
- A runner can validate job outputs and call promotion helpers.
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

Current limitation: object URIs are metadata only. Phase 8 does not yet implement
artifact upload, download, checksum verification against object storage, or
object-storage-first API serving.

## Identifier conventions

| Field | Convention | Example |
| ----- | ---------- | ------- |
| `run_id` | UTC timestamp plus a stable slug. | `2026-06-06T140000Z-sebastopol-ns` |
| `counter_id` | Existing project counter identifier. | `Sebastopol_N-S_airflow` |
| `dataset_version` | DVC revision, source snapshot label, or runtime input version. | `local-dev-dvc` |
| `model_version` | MLflow run ID, registry version, or local model release label. | `mlflow-run-<short_sha>` |

`run_id` values should be unique per external attempt. Airflow retries should
include enough context to avoid overwriting previous attempts unless a deliberate
idempotency rule returns the same job.

## Manifest shape

The canonical JSON shape is implemented by the Pydantic models documented in
[`artifact-manifest-models.md`](artifact-manifest-models.md).

```json
{
  "schema_version": "1.0",
  "artifact_type": "predictions",
  "status": "promoted",
  "run_id": "2026-06-06T140000Z-sebastopol-ns",
  "counter_id": "Sebastopol_N-S_airflow",
  "created_at": "2026-06-06T14:00:00Z",
  "producer": {
    "service": "ml-models-prod",
    "image": "ml-models:prod"
  },
  "source": {
    "raw_file_name": "initial_with_feats.csv",
    "dataset_version": "local-dev-dvc",
    "model_version": "mlflow-run-20260606"
  },
  "storage": {
    "primary_backend": "local",
    "local_path": "docker/prod/runtime/data/final/Sebastopol_N-S_airflow/y_full.csv",
    "object_uri": "s3://mlflow/artifacts/predictions/Sebastopol_N-S_airflow/2026-06-06T140000Z-sebastopol-ns/y_full.csv",
    "checksum_sha256": "..."
  }
}
```

Expected storage rules:

- `primary_backend` is `local` for the first implementation.
- `local_path` is required while `docker/prod/runtime` is the serving backend.
- `object_uri` is optional and can be omitted for local-only artifacts.
- `checksum_sha256` should describe the referenced local artifact file, or the
  canonical object payload once object storage becomes the primary backend.

## Promotion rules

A candidate artifact becomes current only when promotion updates the stable
manifest pointer.

Implemented promotion sequence:

```text
validate manifest -> verify local payload -> write run manifest -> replace current.json
```

Promotion uses a temporary file plus atomic replacement so consumers do not read
partially written `current.json` files.

Promotion must not happen when:

- the manifest is invalid;
- the referenced local artifact is missing;
- the checksum does not match;
- the artifact status is not eligible for serving;
- required producer, source, or storage metadata is missing.

The implemented details and explicit exceptions are documented in
[`artifact-manifest-store.md`](artifact-manifest-store.md).

## Component interactions

| Component | Responsibility in the contract |
| --------- | ------------------------------ |
| ML jobs | Produce artifacts, compute checksums, emit candidate manifests. |
| Runner | Execute typed jobs, persist job status, validate outputs, call promotion helpers. |
| Airflow | Submit jobs, wait for terminal status, record manifest references. |
| MLflow | Track runs, metrics, parameters, and model evidence. |
| MinIO | Optionally store object-backed artifact payloads. |
| API | Read `current.json`, validate metadata, serve the referenced artifact. |
| Prometheus/Grafana | Observe freshness, job status, and current artifact metadata later. |

The contract deliberately keeps Airflow out of low-level filesystem discovery and
keeps the API out of training job internals.

## Phase 8 implementation status and remaining plan

| Step | Story | Status | Owner document |
| ---- | ----- | ------ | -------------- |
| Define artifact handoff vocabulary. | #63 | Done | This document. |
| Add strict Pydantic manifest models. | #64 | Done | [`artifact-manifest-models.md`](artifact-manifest-models.md) |
| Add checksum, write, read, and promotion helpers. | #65 | Done | [`artifact-manifest-store.md`](artifact-manifest-store.md) |
| Make ML jobs emit local manifests. | #66 | Done | [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) |
| Add typed job contracts. | #67 | Done | [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) |
| Add the internal `job-runner-api` skeleton. | #68 | Remaining | [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) |
| Execute typed ML jobs through the runner. | #69 | Remaining | [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) |
| Add production-like Airflow DAG using the runner. | #70 | Remaining | [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) |
| Make the API load the promoted local manifest. | #71 | Remaining | Future API implementation note. |
| Add production-like smoke validation. | #72 | Remaining | Future smoke validation note. |
| Harden runtime configuration and secrets validation. | #73 | Remaining | Future runtime hardening note. |
| Strengthen realistic Phase 8 test coverage. | Follow-up technical debt | Remaining | Future test-fixture issue. |
| Add optional MinIO upload/download helpers. | Later | Remaining | Future object-storage implementation note. |
| Switch `primary_backend` to object storage. | Later | Deferred | Requires API and runner support first. |

## Known technical and functional debt

The remaining stories should not aim for a perfect production platform. They
should leave explicit debt where Phase 8 deliberately stays incremental:

- realistic integration fixtures derived from the real raw-data schema are still
  needed;
- unit coverage should be expanded for invalid manifests, runner state errors,
  API manifest loading failures, and runtime configuration failures;
- `current.json` discovery should stay explicit and counter-scoped so the API
  does not reintroduce implicit latest-file scanning;
- runner execution should stay typed and allow-listed, not a generic shell
  command runner;
- MinIO is not yet a functional artifact backend despite the optional
  `object_uri` contract.

## Out of scope for the current implementation wave

The current Phase 8 implementation wave does not implement:

- object-storage-first serving;
- production secret management;
- distributed runner queues;
- remote deployment validation;
- full performance or load testing.
