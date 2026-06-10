"""Unit tests for shared pipeline metrics helpers."""

from __future__ import annotations

import pytest

from src.metrics import pipeline_metrics
from src.metrics.pipeline_metrics import (
    canonical_site,
    push_step_metrics,
    slug_label_value,
    track_pipeline_step,
)


class TestPipelineMetrics:
    """Unit tests for pipeline metrics normalization helpers."""

    def test_slug_label_value_normalizes_accents_and_separators(self) -> None:
        assert slug_label_value("Totem 73 boulevard de Sébastopol") == (
            "Totem_73_boulevard_de_Sebastopol"
        )

    def test_canonical_site_returns_raw_value_when_site_short_is_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SITE_SHORT", raising=False)

        assert canonical_site("Totem 73 boulevard de Sébastopol") == (
            "Totem 73 boulevard de Sébastopol"
        )

    def test_canonical_site_prefers_site_short_environment_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SITE_SHORT", "Sebastopol")

        assert canonical_site("Totem 73 boulevard de Sébastopol") == "Sebastopol"

    def test_canonical_site_falls_back_to_site_path_slug(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SITE_SHORT", raising=False)
        monkeypatch.delenv("SITE", raising=False)
        monkeypatch.setenv("SITE_PATH", "/data/raw/Totem Sébastopol.csv")

        assert canonical_site(None) == "data_raw_Totem_Sebastopol_csv"

    def test_push_step_metrics_is_disabled_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []
        monkeypatch.setenv("DISABLE_METRICS_PUSH", "1")
        monkeypatch.setattr(
            pipeline_metrics,
            "pushadd_to_gateway",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )

        push_step_metrics(
            step="models",
            duration_s=1.5,
            records=42,
            status="success",
            labels={"site": "Sebastopol", "orientation": "N-S"},
        )

        assert calls == []

    def test_push_step_metrics_pushes_normalized_labels(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []
        monkeypatch.setenv("DISABLE_METRICS_PUSH", "0")
        monkeypatch.setenv("PUSHGATEWAY_ADDR", "pushgateway:9091")
        monkeypatch.setattr(
            pipeline_metrics,
            "pushadd_to_gateway",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )

        push_step_metrics(
            step="models",
            duration_s=1.5,
            records=-3,
            status="error",
            labels={"site": "Sebastopol", "orientation": "N-S"},
        )

        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] == "pushgateway:9091"
        assert kwargs["job"] == "bike-traffic"
        assert kwargs["grouping_key"] == {
            "site": "Sebastopol",
            "orientation": "N-S",
        }

    def test_track_pipeline_step_pushes_success_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        def fake_push_step_metrics(**kwargs) -> None:
            calls.append(kwargs)

        monkeypatch.setattr(
            pipeline_metrics,
            "push_step_metrics",
            fake_push_step_metrics,
        )

        with track_pipeline_step("features", {"site": "Sebastopol"}) as payload:
            payload["records"] = 12

        assert calls[0]["step"] == "features"
        assert calls[0]["records"] == 12
        assert calls[0]["status"] == "success"

    def test_track_pipeline_step_pushes_error_status_before_reraising(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        def fake_push_step_metrics(**kwargs) -> None:
            calls.append(kwargs)

        monkeypatch.setattr(
            pipeline_metrics,
            "push_step_metrics",
            fake_push_step_metrics,
        )

        with pytest.raises(RuntimeError, match="boom"):
            with track_pipeline_step("ingest", {"site": "Sebastopol"}):
                raise RuntimeError("boom")

        assert calls[0]["step"] == "ingest"
        assert calls[0]["records"] == 0
        assert calls[0]["status"] == "error"
