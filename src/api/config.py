"""Runtime configuration for the prediction serving API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.env import ConfigurationError, get_env, get_required_env


class ApiConfigurationError(RuntimeError):
    """Raised when a mandatory API runtime setting is missing."""


@dataclass(frozen=True)
class ApiSettings:
    """Manifest-first API runtime settings."""

    manifest_root: Path
    repository_root: Path
    counter_ids: tuple[str, ...]


def load_settings() -> ApiSettings:
    """Load validated API serving settings from process environment."""

    return ApiSettings(
        manifest_root=Path(_required_env("ARTIFACT_MANIFEST_ROOT")),
        repository_root=Path(_required_env("ARTIFACT_REPOSITORY_ROOT")),
        counter_ids=_parse_csv_env("API_COUNTER_IDS"),
    )


def _required_env(name: str) -> str:
    """Return a mandatory API environment variable."""

    try:
        return get_required_env(name)
    except ConfigurationError as exc:
        raise ApiConfigurationError(str(exc)) from exc


def _parse_csv_env(name: str) -> tuple[str, ...]:
    """Parse a comma-separated optional environment variable."""

    raw_value = get_env(name, default="") or ""
    return tuple(
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    )
