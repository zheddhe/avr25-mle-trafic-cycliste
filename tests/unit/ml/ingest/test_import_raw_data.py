"""Unit tests for the raw ingestion CLI module."""

from __future__ import annotations

from click import Command
from src.ml.ingest.import_raw_data import main


class TestImportRawDataCli:
    """Unit tests for the import_raw_data Click entrypoint."""

    def test_main_is_click_command(self) -> None:
        assert isinstance(main, Command)
        assert main.name == "main"
