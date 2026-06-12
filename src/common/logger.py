"""Logging helpers for runtime services and CLI entrypoints."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_PROJECT_LOGGER_NAME = "src"
_MANAGED_HANDLER_FLAG = "_bike_traffic_managed_handler"
_MANAGED_STREAM_FLAG = "_bike_traffic_stream_handler"
_MANAGED_FILE_PATH = "_bike_traffic_log_file_path"
_CONTEXT_FILE_PATH: ContextVar[str | None] = ContextVar(
    "bike_traffic_log_file_path",
    default=None,
)
_SAFE_LOG_PART_PATTERN = re.compile(r"[^A-Za-z0-9_.=-]+")


class _ContextLogFileFilter(logging.Filter):
    """Keep temporary job file handlers scoped to one execution context."""

    def __init__(self, log_file_path: Path) -> None:
        super().__init__()
        self.log_file_path = str(log_file_path)

    def filter(self, record: logging.LogRecord) -> bool:
        return _CONTEXT_FILE_PATH.get() == self.log_file_path


def configure_logging(
    level: int | str | None = "INFO",
    *,
    log_file_path: str | Path | None = None,
    logger_name: str = _PROJECT_LOGGER_NAME,
) -> None:
    """Configure project logging once for stream and optional file output."""

    log_level = _normalize_log_level(level)
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    stream_handler = _find_managed_stream_handler(logger)
    if stream_handler is None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        setattr(stream_handler, _MANAGED_HANDLER_FLAG, True)
        setattr(stream_handler, _MANAGED_STREAM_FLAG, True)
        logger.addHandler(stream_handler)
    stream_handler.setLevel(log_level)

    if log_file_path is not None:
        file_handler = _find_managed_file_handler(logger, Path(log_file_path))
        if file_handler is None:
            file_handler = _build_file_handler(Path(log_file_path), log_level)
            logger.addHandler(file_handler)
        file_handler.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger without configuring handlers."""

    return logging.getLogger(name)


@contextmanager
def job_logging_context(
    log_file_path: str | Path,
    *,
    level: int | str | None = "INFO",
    logger_name: str = _PROJECT_LOGGER_NAME,
) -> Iterator[Path]:
    """Write project logs to a job-scoped file for one execution context."""

    log_level = _normalize_log_level(level)
    logger = logging.getLogger(logger_name)
    configure_logging(level=log_level, logger_name=logger_name)

    resolved_path = _resolve_log_file_path(log_file_path)
    file_handler = _build_file_handler(resolved_path, log_level)
    file_handler.addFilter(_ContextLogFileFilter(resolved_path))

    logger.addHandler(file_handler)
    token = _CONTEXT_FILE_PATH.set(str(resolved_path))
    try:
        yield resolved_path
    finally:
        _CONTEXT_FILE_PATH.reset(token)
        logger.removeHandler(file_handler)
        file_handler.close()


def build_log_file_path(*parts: str) -> Path:
    """Build a safe path below the repository-local logs directory."""

    safe_parts = [safe_log_path_part(part) for part in parts]
    return Path("logs", *safe_parts)


def safe_log_path_part(value: str | None, *, fallback: str = "unknown") -> str:
    """Return a filesystem-safe log path component."""

    if value is None:
        return fallback
    stripped = value.strip()
    if not stripped:
        return fallback
    return _SAFE_LOG_PART_PATTERN.sub("_", stripped)


def _find_managed_stream_handler(
    logger: logging.Logger,
) -> logging.Handler | None:
    for handler in logger.handlers:
        if getattr(handler, _MANAGED_STREAM_FLAG, False):
            return handler
    return None


def _find_managed_file_handler(
    logger: logging.Logger,
    log_file_path: Path,
) -> logging.Handler | None:
    resolved_path = _resolve_log_file_path(log_file_path)
    for handler in logger.handlers:
        handler_path = getattr(handler, _MANAGED_FILE_PATH, None)
        if handler_path == str(resolved_path):
            return handler
    return None


def _build_file_handler(
    log_file_path: Path,
    log_level: int,
) -> logging.FileHandler:
    resolved_path = _resolve_log_file_path(log_file_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(
        resolved_path,
        mode="a",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(log_level)
    setattr(handler, _MANAGED_HANDLER_FLAG, True)
    setattr(handler, _MANAGED_FILE_PATH, str(resolved_path))
    return handler


def _resolve_log_file_path(log_file_path: str | Path) -> Path:
    return Path(log_file_path).expanduser().resolve()


def _normalize_log_level(level: int | str | None) -> int:
    """Return a numeric logging level, defaulting invalid values to INFO."""

    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = logging.getLevelName(level.strip().upper())
        if isinstance(normalized, int):
            return normalized
    return logging.INFO
