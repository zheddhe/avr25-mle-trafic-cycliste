# ML pipeline artifact manifest emission

This document records the implementation outcome for Phase 8 issue #66. The
hybrid artifact handoff design remains documented in
[`artifact-handoff-strategy.md`](artifact-handoff-strategy.md).

## Implemented package

Prediction manifest emission is implemented through a small framework-neutral
helper:

```text
src/ml/models/artifact_manifest_emission.py
```

The helper builds a validated `ArtifactManifest` for the prediction output
written by `src/ml/models/train_and_predict.py`. It stays independent from
Airflow, FastAPI, Docker SDK, and MLflow internals.

## Runtime behavior

`train_and_predict` keeps the existing development and DVC behavior by default:

- prediction CSV files are still written under `data/final/<sub_dir>`;
- model files are still written under `models/<sub_dir>`;
- no manifest is emitted unless an artifact manifest root is configured.

When `--artifact-manifest-root` or `ARTIFACT_MANIFEST_ROOT` is provided, the
model step emits a local prediction manifest after successful artifact
generation.

The manifest includes:

- `run_id` from runtime context or a UTC fallback;
- `counter_id` from runtime context or `sub_dir`;
- producer metadata for the model job;
- source metadata for the processed dataset and model version;
- local storage metadata and a SHA-256 checksum for `y_full.csv`.

## Manifest store layout

The manifest store writes one run-scoped file and one stable pointer per
`artifact_type` and `counter_id`:

```text
<manifest_root>/<artifact_type>/<counter_id>/<run_id>/manifest.json
<manifest_root>/<artifact_type>/<counter_id>/current.json
```

For local validation, use a repository-relative manifest root:

```bash
ARTIFACT_MANIFEST_ROOT=logs/artifacts/manifests make local-models
```

This produces a layout similar to:

```text
logs/artifacts/manifests/
└── predictions/
    └── Sebastopol_N-S_local/
        ├── <run_id>/
        │   └── manifest.json
        └── current.json
```

Promotion validates the local payload checksum before replacing the scoped
`current.json`. This keeps one current prediction manifest per counter and avoids
cross-counter overwrites in Airflow multi-counter runs.

## Shared technical metrics helpers

The duplicated technical metrics helpers previously present in ingest, features,
and models utilities are now centralized in:

```text
src/metrics/pipeline_metrics.py
```

The shared module owns:

- `canonical_site`;
- `push_step_metrics`;
- `track_pipeline_step`;
- the slug helper used to derive stable fallback labels.

The model-specific `push_business_metrics` function remains in
`src/ml/models/models_utils.py` because it owns model KPI names and semantics.

## Docker runtime impact

The ML Docker images copy the shared metrics package. The model images also copy
`src/artifacts` and `src/ml/models/artifact_manifest_emission.py`.

All internal imports use the repository-root namespace, for example
`src.ml.models.train_and_predict`. Docker images therefore keep `/app` as the
runtime import root and run entrypoints with `python -m src...`.

`docker/prod/docker-compose.yaml` directly mounts
`docker/prod/runtime/artifacts` into the model container as `/app/artifacts` and
configures:

```text
ARTIFACT_MANIFEST_ROOT=/app/artifacts/manifests
ARTIFACT_REPOSITORY_ROOT=/app
ARTIFACT_PRODUCER_SERVICE=ml-models-prod
ARTIFACT_PRODUCER_IMAGE=ml-models:prod
```

The payload file remains written through the existing runtime data mount. Inside
the container, the manifest `local_path` is relative to `/app`, for example:

```text
data/final/Sebastopol_N-S/y_full.csv
```

On the host, this corresponds to:

```text
docker/prod/runtime/data/final/Sebastopol_N-S/y_full.csv
```

This keeps checksum verification local to the executing workload while preserving
the production-like runtime workspace boundary.

## Out of scope

This implementation does not add runner integration, API serving from manifests,
or mandatory MinIO upload. `object_uri` remains optional and is populated only
when an `s3://` URI is provided.
