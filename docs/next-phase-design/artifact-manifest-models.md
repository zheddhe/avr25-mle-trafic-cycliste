# Artifact manifest models

This document records the implementation outcome for Phase 8 issue #64. The
hybrid artifact handoff design remains documented in
[`artifact-handoff-strategy.md`](artifact-handoff-strategy.md).

## Implemented package

The reusable Python contract lives under:

```text
src/artifacts/
├── __init__.py
├── exceptions.py
└── schemas.py
```

The package is intentionally independent from Airflow and FastAPI internals so
ML jobs, the future runner, Airflow integration code, and API serving code can
share the same validated manifest contract.

## Models and enums

`src/artifacts/schemas.py` exposes:

- `ArtifactManifest`;
- `ArtifactProducer`;
- `ArtifactSource`;
- `ArtifactStorage`;
- `ArtifactType`;
- `ArtifactStatus`;
- `StorageBackend`;
- `validate_artifact_manifest`.

The initial schema version is `1.0`. The implementation uses Pydantic field
descriptions and docstrings so the contract remains readable for developers and
can be reused by generated documentation later.

## Validation rules

The manifest model validates:

- required fields: schema version, artifact type, status, run id, counter id,
  creation timestamp, producer, source, and storage;
- local-only manifests with repository-relative `local_path` values;
- hybrid manifests with local paths plus optional `s3://` object URIs;
- storage backend values through constrained enums;
- URI-like local paths, absolute paths, and parent traversal;
- SHA-256 checksums when present;
- timezone-aware `created_at` timestamps;
- undeclared fields through strict Pydantic configuration.

The first runtime implementation still uses `primary_backend="local"`.
`object_uri` remains optional and only records the S3-compatible reference when
the artifact is also available through MinIO or another object store.

## Test coverage

Unit tests live in:

```text
tests/artifacts/test_artifact_manifest_schemas.py
```

They cover the acceptance cases from issue #64:

- valid local manifest;
- valid hybrid manifest;
- missing required field;
- invalid backend;
- invalid URI cases.

## Remaining Phase 8 work

This story only implements the manifest contract. The following topics remain
for later Phase 8 stories:

- writing manifests to disk;
- checksum verification against real payloads;
- promotion helpers and `current.json` replacement;
- ML job emission of manifests;
- API serving from promoted manifests;
- MinIO upload/download helpers.
