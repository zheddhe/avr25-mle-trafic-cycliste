"""Filesystem-backed artifact manifest store helpers.

The helpers write validated run-scoped manifests and promote a stable
``current.json`` file without introducing Airflow, FastAPI, or Docker coupling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import ValidationError

from artifacts.checksums import compute_sha256
from artifacts.exceptions import (
    ArtifactChecksumMismatchError,
    ArtifactManifestNotFoundError,
    ArtifactManifestValidationError,
    ArtifactPayloadNotFoundError,
)
from artifacts.schemas import ArtifactManifest, ArtifactStatus

CURRENT_MANIFEST_NAME = "current.json"
RUNS_DIR_NAME = "runs"

_PROMOTABLE_STATUSES = {
    ArtifactStatus.PRODUCED,
    ArtifactStatus.VALIDATED,
    ArtifactStatus.PROMOTED,
}


def write_manifest(
    manifest: ArtifactManifest | dict[str, Any],
    manifest_root: str | Path,
) -> Path:
    """Validate and write a run-scoped manifest below the manifest root.

    The path is ``<manifest_root>/runs/<run_id>/<artifact_type>-manifest.json``.

    Args:
        manifest: Validated manifest model or raw manifest payload.
        manifest_root: Directory that owns current and run-scoped manifests.

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
    """Validate payload evidence and replace the stable current manifest.

    Promotion writes the run-scoped manifest first, then atomically replaces
    ``<manifest_root>/current.json`` with the same validated payload.

    Args:
        manifest: Validated manifest model or raw manifest payload.
        manifest_root: Directory that owns current and run-scoped manifests.
        repository_root: Base directory used to resolve manifest local paths.

    Returns:
        Path to the promoted ``current.json`` manifest.

    Raises:
        ArtifactManifestValidationError: if the manifest is invalid or not
            eligible for promotion.
        ArtifactPayloadNotFoundError: if the referenced local payload is missing.
        ArtifactChecksumMismatchError: if checksum metadata does not match.
    """

    validated_manifest = _validate_manifest(manifest)
    _ensure_promotable_status(validated_manifest)
    verify_local_payload(validated_manifest, repository_root=repository_root)

    write_manifest(validated_manifest, manifest_root=manifest_root)
    current_path = Path(manifest_root) / CURRENT_MANIFEST_NAME
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


def read_current_manifest(manifest_root: str | Path) -> ArtifactManifest:
    """Read the stable current manifest from a manifest root."""

    return read_manifest(Path(manifest_root) / CURRENT_MANIFEST_NAME)


def verify_local_payload(
    manifest: ArtifactManifest,
    repository_root: str | Path = ".",
) -> str | None:
    """Verify the local payload referenced by a manifest when present.

    Args:
        manifest: Validated artifact manifest.
        repository_root: Base directory used to resolve repository-relative paths.

    Returns:
        The computed checksum when a local payload exists, otherwise ``None``.

    Raises:
        ArtifactPayloadNotFoundError: if the local artifact is missing.
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


def _validate_manifest(manifest: ArtifactManifest | dict[str, Any]) -> ArtifactManifest:
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
    file_name = f"{manifest.artifact_type.value}-manifest.json"
    return Path(manifest_root) / RUNS_DIR_NAME / manifest.run_id / file_name


def _dump_manifest(manifest: ArtifactManifest) -> str:
    return manifest.model_dump_json(indent=2) + "\n"


def _write_json_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
