# Artifact manifest store

This document records the implementation outcome for Phase 8 issue #65. It
extends the manifest model work from issue #64 with local filesystem helpers for
writing, validating, reading, and promoting artifact manifests.

The strategy and remaining Phase 8 plan are kept in
[`artifact-handoff-strategy.md`](artifact-handoff-strategy.md). This document is
limited to the implemented Python store layer.

## Implemented package

The reusable artifact package now includes:

```text
src/artifacts/
├── __init__.py
├── checksums.py
├── exceptions.py
├── manifest_store.py
└── schemas.py
```

The implementation remains framework-neutral. It does not import Airflow,
FastAPI, Docker SDK, MLflow, or concrete ML pipeline modules.

## Manifest store helpers

`src/artifacts/manifest_store.py` exposes:

- `write_manifest`;
- `promote_manifest`;
- `read_manifest`;
- `read_current_manifest`;
- `verify_local_payload`.

`write_manifest` validates a raw payload or `ArtifactManifest` instance and
writes a run-scoped manifest below:

```text
<manifest_root>/runs/<run_id>/<artifact_type>-manifest.json
```

`promote_manifest` validates the manifest, verifies the local payload when a
`local_path` is present, writes the run-scoped copy, and atomically replaces:

```text
<manifest_root>/current.json
```

The stable promoted pointer is a real JSON file, not a symlink. This keeps the
first implementation portable across Windows, WSL, Docker bind mounts, and CI
runners.

## Checksum helper

`src/artifacts/checksums.py` exposes `compute_sha256`. It reads local files in
chunks and returns a lowercase SHA-256 hex digest.

Promotion verifies `storage.checksum_sha256` when it is present. A missing
checksum does not block promotion yet, because schema issue #64 made the checksum
optional. ML manifest emission in issue #66 should provide checksums for
prediction artifacts.

## Explicit errors

The artifact package now exposes explicit exceptions for common failure modes:

- `ArtifactManifestValidationError`;
- `ArtifactManifestNotFoundError`;
- `ArtifactPayloadNotFoundError`;
- `ArtifactChecksumMismatchError`.

Consumers can use these exceptions to distinguish invalid metadata, missing
manifests, missing local payloads, and checksum mismatches.

## Test coverage

Unit tests live in:

```text
tests/artifacts/test_artifact_manifest_store.py
```

They cover:

- SHA-256 checksum calculation;
- missing local payload handling;
- run-scoped manifest writing;
- invalid manifest rejection;
- `current.json` promotion;
- current manifest reading;
- missing and invalid manifest errors;
- checksum mismatch handling;
- rejection of non-promotable statuses.

## Current integration status

The helpers are available for later Phase 8 consumers, but no ML job, runner,
Airflow DAG, or API endpoint calls them yet. Integration is tracked centrally in
[`artifact-handoff-strategy.md`](artifact-handoff-strategy.md).
