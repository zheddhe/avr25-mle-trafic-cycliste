"""Custom exceptions for artifact manifest handling."""

from __future__ import annotations


class ArtifactManifestError(ValueError):
    """Base exception raised by artifact manifest helpers."""


class ArtifactManifestValidationError(ArtifactManifestError):
    """Raised when a raw artifact manifest payload fails validation."""


class ArtifactManifestNotFoundError(ArtifactManifestError):
    """Raised when an expected artifact manifest file is missing."""


class ArtifactPayloadNotFoundError(ArtifactManifestError):
    """Raised when a local artifact payload referenced by a manifest is missing."""


class ArtifactChecksumMismatchError(ArtifactManifestError):
    """Raised when a local artifact payload checksum does not match metadata."""
