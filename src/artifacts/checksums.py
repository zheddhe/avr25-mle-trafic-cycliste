"""Checksum helpers for local artifact payload validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.artifacts.exceptions import ArtifactPayloadNotFoundError

_CHUNK_SIZE_BYTES = 1024 * 1024


def compute_sha256(path: str | Path) -> str:
    """Compute the SHA-256 checksum of a local artifact file.

    Args:
        path: Local file path to hash.

    Returns:
        Lowercase SHA-256 hexadecimal digest.

    Raises:
        ArtifactPayloadNotFoundError: if the path does not point to a file.
    """

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ArtifactPayloadNotFoundError(
            f"Artifact payload does not exist: {artifact_path}"
        )

    digest = hashlib.sha256()
    with artifact_path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)

    return digest.hexdigest()
