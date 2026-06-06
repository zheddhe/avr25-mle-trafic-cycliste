# Hybrid artifact handoff strategy

This document is the Phase 8 design artifact for issue #63. It defines the
manifest-first contract used to hand off generated artifacts between ML jobs,
Airflow, the future job runner, MLflow, optional MinIO object storage, and the
FastAPI prediction service.

The goal is design only. This document does not implement Python manifest models,
manifest writers, MinIO upload helpers, API serving changes, or ML job changes.
Those changes are intentionally split into later Phase 8 stories.

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

## Scope and inputs

| Source | Use in this design |
| ------ | ------------------ |
| [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md) | Dev/prod runtime split and `docker/prod/runtime` ownership. |
| [`../current-runtime-and-operations/repository-structure.md`](../current-runtime-and-operations/repository-structure.md) | Repository path ownership and DVC boundary. |
| [`../current-runtime-and-operations/ports-and-services.md`](../current-runtime-and-operations/ports-and-services.md) | Local host exposure and internal-only MinIO policy. |
| [`../current-runtime-and-operations/dependency-strategy.md`](../current-runtime-and-operations/dependency-strategy.md) | MLflow, MinIO, and runtime dependency compatibility. |
| [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) | Target runner and worker-pool execution model. |
| Phase 7 issue #57 | Production-like Compose runtime foundation. |

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

- ML prediction jobs know the output path, counter, dataset, model, metrics, and checksum evidence.
- A runner or promotion helper can validate the manifest and update the stable promoted pointer.
- Airflow orchestrates job order and records the manifest reference, but should not fabricate artifact metadata that belongs to ML code.
- The API reads the promoted manifest and must not guess file paths when the manifest is available.

The first implementation should write a real `current.json` file instead of a
symlink. This is more portable across Windows, WSL, Docker bind mounts, and CI
runners.

## Local backend conventions

Local production-like artifacts should stay under `docker/prod/runtime`.

Recommended paths:

```text
docker/prod/runtime/artifacts/
├── manifests/
│   ├── current.json
│   └── runs/
│       └── <run_id>/
│           └── <artifact_type>-manifest.json
├── data/
│   └── final/
│       └── <counter_id>/
│           └── <run_id>/
│               └── predictions.parquet
└── models/
    └── <counter_id>/
        └── <model_version>/
            └── model.joblib
```

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

The initial manifest shape should remain JSON-compatible and map cleanly to
future Pydantic models.

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
    "raw_file_name": "comptage-velo-donnees-compteurs-2024-2025_Enriched_ML-ready_data.csv",
    "dataset_version": "local-dev-dvc",
    "model_version": "mlflow-run-20260606"
  },
  "storage": {
    "primary_backend": "local",
    "local_path": "docker/prod/runtime/artifacts/data/final/Sebastopol_N-S_airflow/2026-06-06T140000Z-sebastopol-ns/predictions.parquet",
    "object_uri": "s3://mlflow/artifacts/predictions/Sebastopol_N-S_airflow/2026-06-06T140000Z-sebastopol-ns/predictions.parquet",
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

Initial promotion rule:

```text
write candidate manifest -> validate -> write or replace current.json
```

Promotion must be atomic enough that consumers never observe a partially written
`current.json`. The implementation story can choose a portable write strategy,
for example writing to a temporary file and replacing the target file.

Promotion must not happen when:

- the manifest is invalid;
- the referenced local artifact is missing;
- the checksum does not match;
- the artifact status is not eligible for serving;
- required producer, source, or storage metadata is missing.

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

## Migration path

Phase 8 should move in small validated steps:

1. Document the contract and vocabulary in this file.
2. Add strict Pydantic manifest models.
3. Add manifest write, checksum, and promotion helpers.
4. Make ML prediction jobs emit local-only manifests.
5. Add typed job contracts and runner execution around manifest references.
6. Make the API load the promoted local manifest instead of scanning folders.
7. Add smoke validation proving runner, Airflow, API, and monitoring integration.
8. Add optional MinIO upload/download helpers when the local contract is stable.
9. Switch `primary_backend` from `local` to object storage only after API and
   runner consumers support it.

## Later stories consuming this contract

Known Phase 8 consumers are:

- #64: implement artifact manifest models and validation;
- #65: implement artifact writer and promotion helpers;
- #66: adapt the ML pipeline to emit artifact manifests;
- #67: introduce typed pipeline job contracts;
- #68: implement the internal job-runner API skeleton;
- #69: execute typed ML jobs through the runner;
- #70: add the production-like Airflow DAG using the runner API;
- #71: make the API artifact-aware;
- #72: add production-like smoke validation;
- #73: harden runtime configuration and secrets validation.

## Out of scope for this design

- Python Pydantic schemas.
- Manifest store implementation.
- ML job manifest emission.
- API serving changes.
- MinIO upload/download helpers.
- Production secret management.
