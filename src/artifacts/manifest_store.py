"""Filesystem-backed artifact manifest store helpers.

The helpers write validated counter-scoped manifests and promote stable
``current.json`` files without introducing Airflow, FastAPI, or Docker coupling.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import ValidationError

from src.artifacts.checksums import compute_sha256
from src.artifacts.exceptions import (
    ArtifactChecksumMismatchError,
    ArtifactManifestNotFoundError,
    ArtifactManifestValidationError,
)
from src.artifacts.schemas import ArtifactManifest, ArtifactStatus

CURRENT_MANIFEST_NAME = "current.json"
RUN_MANIFEST_NAME = "manifest.json"
PROMOTION_LOCK_NAME = ".promotion.lock"
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
LOCK_POLL_INTERVAL_SECONDS = 0.05

_PROMOTABLE_STATUSES = {
    ArtifactStatus.PRODUCED,
    ArtifactStatus.VALIDATED,
    ArtifactStatus.PROMOTED,
}


def write_manifest(
    manifest: ArtifactManifest | dict[str, Any],
    manifest_root: str | Path,
) -> Path:
    """Validate and write a counter-scoped run manifest.

    The path is::

        <manifest_root>/<artifact_type>/<counter_id>/<run_id>/manifest.json

    Args:
        manifest: Validated manifest model or raw manifest payload.
        manifest_root: Directory that owns counter-scoped manifests.

    Returns:
        Path to the written run-scoped manifest.

    Raises:
        ArtifactManifestValidationError: if validation fails.
    """

    validated_manifest = _validate_manifest(manifest)
    manifest_path = _build_run_manifest_path(
        manifest_root=manifest_root,
        manifest=validated_manifest,
    )
    _write_json_atomic(manifest_path, _dump_manifest(validated_manifest))

    return manifest_path


def promote_manifest(
    manifest: ArtifactManifest | dict[str, Any],
    manifest_root: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    """Validate payload evidence and replace the counter current manifest.

    Promotion writes the run-scoped manifest first, then atomically replaces::

        <manifest_root>/<artifact_type>/<counter_id>/current.json

    Promotions are serialized per artifact type and counter id. Different
    counter scopes use different lock files and can progress independently.

    Args:
        manifest: Validated manifest model or raw manifest payload.
        manifest_root: Directory that owns counter-scoped manifests.
        repository_root: Base directory used to resolve manifest local paths.

    Returns:
        Path to the promoted ``current.json`` manifest.

    Raises:
        ArtifactManifestValidationError: if the manifest is invalid, not
            eligible for promotion, or its scope lock cannot be acquired.
        ArtifactPayloadNotFoundError: propagated from the checksum helper when
            the referenced local payload is missing.
        ArtifactChecksumMismatchError: if checksum metadata does not match.
    """

    validated_manifest = _validate_manifest(manifest)
    _ensure_promotable_status(validated_manifest)
    verify_local_payload(validated_manifest, repository_root=repository_root)

    scope_path = _build_artifact_scope_path(
        manifest_root=manifest_root,
        manifest=validated_manifest,
    )
    with _manifest_scope_lock(scope_path):
        write_manifest(validated_manifest, manifest_root=manifest_root)
        current_path = scope_path / CURRENT_MANIFEST_NAME
        _write_json_atomic(current_path, _dump_manifest(validated_manifest))

    return current_path


def read_manifest(path: str | Path) -> ArtifactManifest:
    """Read one manifest JSON file and return a validated model."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ArtifactManifestNotFoundError(
            f"Artifact manifest does not exist: {manifest_path}"
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactManifestValidationError(
            f"Artifact manifest is not valid JSON: {manifest_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ArtifactManifestValidationError(
            f"Artifact manifest must contain a JSON object: {manifest_path}"
        )

    return _validate_manifest(payload)


def read_current_manifest(
    manifest_root: str | Path,
    artifact_type: str,
    counter_id: str,
) -> ArtifactManifest:
    """Read the stable current manifest for one artifact type and counter."""

    current_path = (
        Path(manifest_root)
        / artifact_type
        / counter_id
        / CURRENT_MANIFEST_NAME
    )
    return read_manifest(current_path)


def verify_local_payload(
    manifest: ArtifactManifest,
    repository_root: str | Path = ".",
) -> str | None:
    """Verify the local payload referenced by a manifest when present.

    Args:
        manifest: Validated artifact manifest.
        repository_root: Base directory used to resolve repository-relative
            paths.

    Returns:
        The computed checksum when a local payload exists, otherwise ``None``.

    Raises:
        ArtifactPayloadNotFoundError: propagated from the checksum helper when
            the referenced local payload is missing.
        ArtifactChecksumMismatchError: if checksum metadata does not match.
    """

    local_path = manifest.storage.local_path
    if local_path is None:
        return None

    payload_path = Path(repository_root) / local_path
    checksum = compute_sha256(payload_path)
    expected_checksum = manifest.storage.checksum_sha256
    if expected_checksum and checksum != expected_checksum:
        raise ArtifactChecksumMismatchError(
            "Artifact checksum mismatch for "
            f"{local_path}: expected {expected_checksum}, got {checksum}"
        )

    return checksum


def _validate_manifest(
    manifest: ArtifactManifest | dict[str, Any],
) -> ArtifactManifest:
    if isinstance(manifest, ArtifactManifest):
        return manifest

    try:
        return ArtifactManifest.model_validate(manifest)
    except ValidationError as error:
        raise ArtifactManifestValidationError(str(error)) from error


def _ensure_promotable_status(manifest: ArtifactManifest) -> None:
    if manifest.status not in _PROMOTABLE_STATUSES:
        raise ArtifactManifestValidationError(
            "artifact status is not eligible for promotion: "
            f"{manifest.status.value}"
        )


def _build_run_manifest_path(
    manifest_root: str | Path,
    manifest: ArtifactManifest,
) -> Path:
    return (
        _build_artifact_scope_path(manifest_root=manifest_root, manifest=manifest)
        / manifest.run_id
        / RUN_MANIFEST_NAME
    )


def _build_current_manifest_path(
    manifest_root: str | Path,
    manifest: ArtifactManifest,
) -> Path:
    return (
        _build_artifact_scope_path(manifest_root=manifest_root, manifest=manifest)
        / CURRENT_MANIFEST_NAME
    )


def _build_artifact_scope_path(
    manifest_root: str | Path,
    manifest: ArtifactManifest,
) -> Path:
    return Path(manifest_root) / manifest.artifact_type.value / manifest.counter_id


def _dump_manifest(manifest: ArtifactManifest) -> str:
    return manifest.model_dump_json(indent=2) + "\n"


@contextmanager
def _manifest_scope_lock(scope_path: Path) -> Iterator[None]:
    """Acquire a process-independent lock for one artifact/counter scope."""

    scope_path.mkdir(parents=True, exist_ok=True)
    lock_path = scope_path / PROMOTION_LOCK_NAME
    deadline = time.monotonic() + DEFAULT_LOCK_TIMEOUT_SECONDS
    file_descriptor: int | None = None

    while file_descriptor is None:
        try:
            file_descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as error:
            if time.monotonic() >= deadline:
                raise ArtifactManifestValidationError(
                    "Timed out waiting for artifact manifest promotion lock: "
                    f"{lock_path}"
                ) from error
            time.sleep(LOCK_POLL_INTERVAL_SECONDS)

    try:
        _write_lock_owner(file_descriptor)
        yield
    finally:
        os.close(file_descriptor)
        _remove_lock_file(lock_path)


def _write_lock_owner(file_descriptor: int) -> None:
    lock_payload = f"pid={os.getpid()}\n".encode("utf-8")
    os.write(file_descriptor, lock_payload)
    os.fsync(file_descriptor)


def _remove_lock_file(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _write_json_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = Path(tmp_file.name)

        os.replace(tmp_path, path)
        tmp_path = None
        _fsync_directory(path.parent)
    finally:
        if tmp_path is not None:
            _remove_temporary_file(tmp_path)


def _remove_temporary_file(tmp_path: Path) -> None:
    try:
        tmp_path.unlink()
    except FileNotFoundError:
        return


def _fsync_directory(path: Path) -> None:
    try:
        file_descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
