# tests/api/test_main.py
from __future__ import annotations

import sys
from importlib import import_module

from fastapi import FastAPI


class TestApiMain:
    def test_main_exposes_fastapi_app(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(
            "ARTIFACT_MANIFEST_ROOT",
            str(tmp_path / "artifacts" / "manifests"),
        )
        monkeypatch.setenv("ARTIFACT_REPOSITORY_ROOT", str(tmp_path))
        sys.modules.pop("src.api.main", None)

        main = import_module("src.api.main")

        assert isinstance(main.app, FastAPI)
        assert main.app.title == "API du trafic cycliste"
