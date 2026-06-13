"""Unit tests for process environment helpers."""

from __future__ import annotations

import logging
import os

import pytest
from src.common import env as env_module
from src.common.env import (
    ConfigurationError,
    get_env,
    get_optional_env,
    get_required_env,
    patched_env,
)


class TestCommonEnv:
    """Unit tests for centralized runtime environment access."""

    def test_get_required_env_returns_stripped_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APP_VALUE", " configured ")

        value = get_required_env("APP_VALUE")

        assert value == "configured"

    def test_get_env_prefers_environment_over_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APP_VALUE", "from_env")

        value = get_env("APP_VALUE", default="from_default")

        assert value == "from_env"

    def test_get_optional_env_returns_default_when_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("APP_VALUE", raising=False)

        value = get_optional_env("APP_VALUE", default="fallback")

        assert value == "fallback"

    def test_get_required_env_raises_and_logs_when_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("APP_VALUE", raising=False)
        runtime_logger = logging.getLogger(env_module.LOGGER.name)
        runtime_logger.addHandler(caplog.handler)
        try:
            with pytest.raises(ConfigurationError, match="APP_VALUE"):
                get_required_env("APP_VALUE")
        finally:
            runtime_logger.removeHandler(caplog.handler)

        assert "Missing required environment variable: APP_VALUE" in caplog.text

    def test_blank_required_env_is_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APP_VALUE", "  ")

        with pytest.raises(ConfigurationError, match="APP_VALUE"):
            get_required_env("APP_VALUE")

    def test_patched_env_restores_previous_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EXISTING_KEY", "before")

        with patched_env({"EXISTING_KEY": "after", "NEW_KEY": "value"}):
            assert os.environ["EXISTING_KEY"] == "after"
            assert os.environ["NEW_KEY"] == "value"

        assert os.environ["EXISTING_KEY"] == "before"
        assert "NEW_KEY" not in os.environ
