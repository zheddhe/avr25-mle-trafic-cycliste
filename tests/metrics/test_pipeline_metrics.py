"""Unit tests for shared pipeline metrics helpers."""

from __future__ import annotations

from src.metrics.pipeline_metrics import canonical_site, slug_label_value


class TestPipelineMetrics:
    """Unit tests for pipeline metrics normalization helpers."""

    def test_slug_label_value_normalizes_accents_and_separators(self) -> None:
        assert slug_label_value("Totem 73 boulevard de Sébastopol") == (
            "Totem_73_boulevard_de_Sebastopol"
        )

    def test_canonical_site_returns_raw_value_when_site_short_is_missing(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.delenv("SITE_SHORT", raising=False)

        assert canonical_site("Totem 73 boulevard de Sébastopol") == (
            "Totem 73 boulevard de Sébastopol"
        )

    def test_canonical_site_prefers_site_short_environment_value(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("SITE_SHORT", "Sebastopol")

        assert canonical_site("Totem 73 boulevard de Sébastopol") == "Sebastopol"
