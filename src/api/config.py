"""Runtime configuration for the prediction serving API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiSettings:
    """Manifest-first API runtime settings."""

    manifest_root: Path
    repository_root: Path
    counter_ids: tuple[str, ...]


def load_settings() -> ApiSettings:
    """Load explicit API serving settings from environment variables."""

    return ApiSettings(
        manifest_root=Path(
            os.getenv("ARTIFACT_MANIFEST_ROOT", "/app/artifacts/manifests")
        ),
        repository_root=Path(os.getenv("ARTIFACT_REPOSITORY_ROOT", "/app")),
        counter_ids=_parse_csv_env("API_COUNTER_IDS"),
    )


def _parse_csv_env(name: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, "")
    return tuple(
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    )
