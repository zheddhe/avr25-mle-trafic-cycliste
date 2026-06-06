"""Custom exceptions for artifact manifest handling."""

from __future__ import annotations


class ArtifactManifestError(ValueError):
    """Base exception raised by artifact manifest helpers."""


class ArtifactManifestValidationError(ArtifactManifestError):
    """Raised when a raw artifact manifest payload fails validation."""
