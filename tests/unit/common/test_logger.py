"""Unit tests for logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from src.common import logger as logger_module
from src.common.logger import (
    configure_logging,
    get_logger,
    job_logging_context,
    safe_log_path_part,
)


class TestCommonLogger:
    """Unit tests for centralized logging configuration."""

    def test_configure_logging_is_idempotent(self) -> None:
        project_logger = logging.getLogger("src")
        original_handlers = list(project_logger.handlers)
        original_level = project_logger.level
        original_propagate = project_logger.propagate
        project_logger.handlers.clear()
        try:
            configure_logging(level="INFO")
            first_handlers = list(project_logger.handlers)
            configure_logging(level="DEBUG")
            second_handlers = list(project_logger.handlers)

            assert len(first_handlers) == 1
            assert second_handlers == first_handlers
            assert project_logger.level == logging.DEBUG
            assert project_logger.propagate is False
        finally:
            project_logger.handlers.clear()
            project_logger.handlers.extend(original_handlers)
            project_logger.setLevel(original_level)
            project_logger.propagate = original_propagate

    def test_configure_logging_writes_to_service_file(
        self,
        tmp_path: Path,
    ) -> None:
        project_logger = logging.getLogger("src")
        original_handlers = list(project_logger.handlers)
        original_level = project_logger.level
        original_propagate = project_logger.propagate
        project_logger.handlers.clear()
        log_path = tmp_path / "logs" / "api" / "main.log"
        try:
            configure_logging(level="INFO", log_file_path=log_path)
            get_logger("src.api.example").info("Loaded %s rows", 3)
            for handler in project_logger.handlers:
                handler.flush()

            assert "Loaded 3 rows" in log_path.read_text(encoding="utf-8")
        finally:
            for handler in list(project_logger.handlers):
                project_logger.removeHandler(handler)
                handler.close()
            project_logger.handlers.extend(original_handlers)
            project_logger.setLevel(original_level)
            project_logger.propagate = original_propagate

    def test_job_logging_context_scopes_file_output(
        self,
        tmp_path: Path,
    ) -> None:
        project_logger = logging.getLogger("src")
        original_handlers = list(project_logger.handlers)
        original_level = project_logger.level
        original_propagate = project_logger.propagate
        project_logger.handlers.clear()
        log_path = tmp_path / "logs" / "ml" / "ingest" / "run_job.log"
        try:
            runtime_logger = get_logger("src.ml.ingest.example")
            configure_logging(level="INFO")
            runtime_logger.info("outside before")
            with job_logging_context(log_path, level="INFO"):
                runtime_logger.info("inside job_id=%s", "job-1")
            runtime_logger.info("outside after")

            content = log_path.read_text(encoding="utf-8")
            assert "inside job_id=job-1" in content
            assert "outside before" not in content
            assert "outside after" not in content
        finally:
            for handler in list(project_logger.handlers):
                project_logger.removeHandler(handler)
                handler.close()
            project_logger.handlers.extend(original_handlers)
            project_logger.setLevel(original_level)
            project_logger.propagate = original_propagate

    def test_get_logger_returns_named_logger(self) -> None:
        runtime_logger = get_logger("src.example")

        assert runtime_logger.name == "src.example"

    def test_safe_log_path_part_replaces_unsafe_characters(self) -> None:
        value = safe_log_path_part("run/2026:06")

        assert value == "run_2026_06"

    def test_normalize_invalid_level_defaults_to_info(self) -> None:
        level = logger_module._normalize_log_level("invalid")

        assert level == logging.INFO
