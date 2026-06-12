"""Logging helpers for runtime services and CLI entrypoints."""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_MANAGED_HANDLER_FLAG = "_bike_traffic_managed_handler"


def configure_logging(level: int | str | None = "INFO") -> None:
    """Configure root logging once for container stdout collection."""

    log_level = _normalize_log_level(level)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    handler = _find_managed_handler(root_logger)
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        setattr(handler, _MANAGED_HANDLER_FLAG, True)
        root_logger.addHandler(handler)

    handler.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger without configuring handlers."""

    return logging.getLogger(name)


def _find_managed_handler(logger: logging.Logger) -> logging.Handler | None:
    """Return the stream handler managed by this module, if present."""

    for handler in logger.handlers:
        if getattr(handler, _MANAGED_HANDLER_FLAG, False):
            return handler
    return None


def _normalize_log_level(level: int | str | None) -> int:
    """Return a numeric logging level, defaulting invalid values to INFO."""

    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = logging.getLevelName(level.strip().upper())
        if isinstance(normalized, int):
            return normalized
    return logging.INFO
