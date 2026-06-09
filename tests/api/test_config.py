# tests/api/test_config.py
from __future__ import annotations

from pathlib import Path

import pytest

from src.api.config import (
    ApiConfigurationError,
    _parse_csv_env,
    _required_env,
    load_settings,
)


class TestApiConfig:
    def test_required_env_returns_configured_value(self, monkeypatch) -> None:
        monkeypatch.setenv("ARTIFACT_MANIFEST_ROOT", " /app/artifacts/manifests ")

        value = _required_env("ARTIFACT_MANIFEST_ROOT")

        assert value == "/app/artifacts/manifests"

    def test_required_env_raises_when_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("ARTIFACT_MANIFEST_ROOT", raising=False)

        with pytest.raises(ApiConfigurationError) as exc_info:
            _required_env("ARTIFACT_MANIFEST_ROOT")

        assert "ARTIFACT_MANIFEST_ROOT" in str(exc_info.value)

    def test_required_env_raises_when_blank(self, monkeypatch) -> None:
        monkeypatch.setenv("ARTIFACT_REPOSITORY_ROOT", "  ")

        with pytest.raises(ApiConfigurationError) as exc_info:
            _required_env("ARTIFACT_REPOSITORY_ROOT")

        assert "ARTIFACT_REPOSITORY_ROOT" in str(exc_info.value)

    def test_parse_csv_env_ignores_empty_values(self, monkeypatch) -> None:
        monkeypatch.setenv("API_COUNTER_IDS", "alpha, beta,, gamma, ")

        counter_ids = _parse_csv_env("API_COUNTER_IDS")

        assert counter_ids == ("alpha", "beta", "gamma")

    def test_parse_csv_env_returns_empty_tuple_when_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("API_COUNTER_IDS", raising=False)

        counter_ids = _parse_csv_env("API_COUNTER_IDS")

        assert counter_ids == ()

    def test_load_settings_reads_manifest_runtime_environment(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        manifest_root = tmp_path / "artifacts" / "manifests"
        repository_root = tmp_path / "repository"
        monkeypatch.setenv("ARTIFACT_MANIFEST_ROOT", str(manifest_root))
        monkeypatch.setenv("ARTIFACT_REPOSITORY_ROOT", str(repository_root))
        monkeypatch.setenv("API_COUNTER_IDS", "counter-a,counter-b")

        settings = load_settings()

        assert settings.manifest_root == manifest_root
        assert settings.repository_root == repository_root
        assert settings.counter_ids == ("counter-a", "counter-b")

    def test_load_settings_raises_when_manifest_root_is_missing(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("ARTIFACT_MANIFEST_ROOT", raising=False)
        monkeypatch.setenv("ARTIFACT_REPOSITORY_ROOT", str(tmp_path))

        with pytest.raises(ApiConfigurationError) as exc_info:
            load_settings()

        assert "ARTIFACT_MANIFEST_ROOT" in str(exc_info.value)
