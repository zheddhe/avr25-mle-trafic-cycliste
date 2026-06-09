"""Runtime configuration for the prediction serving API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ApiConfigurationError(RuntimeError):
    """Raised when a mandatory API runtime setting is missing."""


@dataclass(frozen=True)
class ApiSettings:
    """Manifest-first API runtime settings."""

    manifest_root: Path
    repository_root: Path
    counter_ids: tuple[str, ...]


def load_settings() -> ApiSettings:
    """Load explicit API serving settings from environment variables."""

    return ApiSettings(
        manifest_root=Path(_required_env("ARTIFACT_MANIFEST_ROOT")),
        repository_root=Path(_required_env("ARTIFACT_REPOSITORY_ROOT")),
        counter_ids=_parse_csv_env("API_COUNTER_IDS"),
    )


def _required_env(name: str) -> str:
    """Return a mandatory environment variable or raise an explicit error."""

    value = os.getenv(name)
    if value is None or not value.strip():
        raise ApiConfigurationError(
            f"Missing required environment variable: {name}"
        )
    return value.strip()


def _parse_csv_env(name: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, "")
    return tuple(
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    )
