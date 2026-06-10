"""Unit tests for artifact manifest exceptions."""

from __future__ import annotations

from src.artifacts import exceptions


class TestArtifactManifestExceptions:
    """Unit tests for the artifact exception hierarchy."""

    def test_domain_exceptions_share_manifest_error_base(self) -> None:
        exception_classes = (
            exceptions.ArtifactManifestValidationError,
            exceptions.ArtifactManifestNotFoundError,
            exceptions.ArtifactPayloadNotFoundError,
            exceptions.ArtifactChecksumMismatchError,
        )

        for exception_class in exception_classes:
            assert issubclass(exception_class, exceptions.ArtifactManifestError)
            assert issubclass(exception_class, ValueError)
