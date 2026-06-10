"""Unit tests for artifact checksum helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.artifacts.checksums import compute_sha256
from src.artifacts.exceptions import ArtifactPayloadNotFoundError


class TestComputeSha256:
    """Unit tests for compute_sha256."""

    def test_returns_expected_digest(self, tmp_path: Path) -> None:
        payload_path = tmp_path / "payload.txt"
        payload_path.write_text("bike traffic\n", encoding="utf-8")

        checksum = compute_sha256(payload_path)

        assert checksum == (
            "1540cc2351515a9d24ab9e4c8cac0fae"
            "99a461dcfe8f9b4758ae3f39db30eed2"
        )

    def test_missing_payload_raises_explicit_error(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "missing.csv"

        with pytest.raises(ArtifactPayloadNotFoundError, match="does not exist"):
            compute_sha256(missing_path)
