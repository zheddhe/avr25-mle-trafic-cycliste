"""Unit tests for logging helpers."""

from __future__ import annotations

import logging

from src.common import logger as logger_module
from src.common.logger import configure_logging, get_logger


class TestCommonLogger:
    """Unit tests for centralized logging configuration."""

    def test_configure_logging_is_idempotent(self) -> None:
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        original_level = root_logger.level
        root_logger.handlers.clear()
        try:
            configure_logging(level="INFO")
            first_handlers = list(root_logger.handlers)
            configure_logging(level="DEBUG")
            second_handlers = list(root_logger.handlers)

            assert len(first_handlers) == 1
            assert second_handlers == first_handlers
            assert root_logger.level == logging.DEBUG
        finally:
            root_logger.handlers.clear()
            root_logger.handlers.extend(original_handlers)
            root_logger.setLevel(original_level)

    def test_get_logger_returns_named_logger(self) -> None:
        runtime_logger = get_logger("src.example")

        assert runtime_logger.name == "src.example"

    def test_normalize_invalid_level_defaults_to_info(self) -> None:
        level = logger_module._normalize_log_level("invalid")

        assert level == logging.INFO
