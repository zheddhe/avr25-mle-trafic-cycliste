"""Unit tests for the feature engineering CLI module."""

from __future__ import annotations

from click import Command

from src.ml.features.build_features import main


class TestBuildFeaturesCli:
    """Unit tests for the build_features Click entrypoint."""

    def test_main_is_click_command(self) -> None:
        assert isinstance(main, Command)
        assert main.name == "main"
