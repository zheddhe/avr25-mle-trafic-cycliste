# ML pipeline artifact manifest emission

This document records the implementation intent for Phase 8 issue #66. The
hybrid artifact handoff design remains documented in
[`artifact-handoff-strategy.md`](artifact-handoff-strategy.md).

## Implemented package

Prediction manifest emission is implemented through a small framework-neutral
helper:

```text
src/ml/models/artifact_manifest_emission.py
```

The helper builds a validated `ArtifactManifest` for the prediction output written
by `src/ml/models/train_and_predict.py`. It stays independent from Airflow,
FastAPI, Docker SDK, and MLflow internals.

## Runtime behavior

`train_and_predict` keeps the existing development and DVC behavior by default:

- prediction CSV files are still written under `data/final/<sub_dir>`;
- model files are still written under `models/<sub_dir>`;
- no manifest is emitted unless an artifact manifest root is configured.

When `--artifact-manifest-root` or `ARTIFACT_MANIFEST_ROOT` is provided, the model
step emits a local-only prediction manifest after successful artifact generation.
The manifest includes:

- `run_id` from CLI/runtime context or a UTC fallback;
- `counter_id` from runtime context or `sub_dir`;
- producer metadata for the model job;
- source metadata for the processed dataset and model version;
- local storage metadata and a SHA-256 checksum for `y_full.csv`.

The first implementation promotes the manifest through the existing store helper.
That writes both the run-scoped manifest and `current.json` under the configured
manifest root.

## Production-like Compose wiring

`docker/prod` mounts `docker/prod/runtime/artifacts` into the model container as
`/app/artifacts` and configures:

```text
ARTIFACT_MANIFEST_ROOT=/app/artifacts/manifests
ARTIFACT_REPOSITORY_ROOT=/app
ARTIFACT_PRODUCER_SERVICE=ml-models-prod
ARTIFACT_PRODUCER_IMAGE=ml-models:prod
```

The payload file remains written through the existing runtime data mount. Inside
the container the manifest `local_path` is relative to `/app`, for example:

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

The duplicate technical metrics helpers noted in the issue comments remain a
separate follow-up to avoid expanding the acceptance scope of this story.
