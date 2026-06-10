"""Unit tests for the train_and_predict CLI module."""

from __future__ import annotations

from click import Command

from src.ml.models.train_and_predict import _extract_site_orientation, main


class TestTrainAndPredictCli:
    """Unit tests for the train_and_predict Click entrypoint."""

    def test_main_is_click_command(self) -> None:
        assert isinstance(main, Command)
        assert main.name == "main"

    def test_extract_site_orientation_uses_sub_dir_parts(self) -> None:
        assert _extract_site_orientation("Sebastopol_N-S_airflow") == (
            "Sebastopol",
            "N-S",
        )

    def test_extract_site_orientation_falls_back_to_na(self) -> None:
        assert _extract_site_orientation("Sebastopol") == ("Sebastopol", "NA")
