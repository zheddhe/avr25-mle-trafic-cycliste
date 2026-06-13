"""Process environment access for runtime services."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from src.common.logger import get_logger

LOGGER = get_logger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when a mandatory runtime configuration value is missing."""


def get_env(
    name: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    """Return a stripped process environment value or an optional default."""

    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()

    if required:
        LOGGER.error("Missing required environment variable: %s", name)
        raise ConfigurationError(f"Missing required environment variable: {name}")

    if default is not None:
        return default
    else:
        return ""


def get_required_env(name: str) -> str:
    """Return a mandatory process environment value."""

    value = get_env(name, required=True)
    if value is None:
        LOGGER.error("Missing required environment variable: %s", name)
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


def get_optional_env(name: str, *, default: str | None = None) -> str | None:
    """Return an optional process environment value."""

    return get_env(name, default=default)


@contextmanager
def patched_env(updates: Mapping[str, str]) -> Iterator[None]:
    """Temporarily expose a typed job contract as environment variables."""

    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
