"""Unit tests for shared pipeline metrics helpers."""

from __future__ import annotations

from src.metrics.pipeline_metrics import canonical_site, slug_label_value


class TestPipelineMetrics:
    """Unit tests for pipeline metrics normalization helpers."""

    def test_slug_label_value_normalizes_accents_and_separators(self) -> None:
        assert slug_label_value("Totem 73 boulevard de Sébastopol") == (
            "totem-73-boulevard-de-sebastopol"
        )

    def test_canonical_site_returns_stable_counter_label(self) -> None:
        assert canonical_site("Totem 73 boulevard de Sébastopol", "N-S") == (
            "Sebastopol_N-S"
        )
