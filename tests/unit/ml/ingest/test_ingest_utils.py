"""Unit tests for ingestion utility functions."""

from __future__ import annotations

import pandas as pd

from src.ml.ingest.ingest_utils import apply_percent_range_selection


class TestApplyPercentRangeSelection:
    """Unit tests for apply_percent_range_selection."""

    def test_selects_requested_percent_range(self) -> None:
        dataframe = pd.DataFrame({"value": [0, 1, 2, 3, 4]})

        result = apply_percent_range_selection(dataframe, (20.0, 80.0))

        assert result["value"].tolist() == [1, 2, 3]

    def test_reversed_range_returns_empty_dataframe(self) -> None:
        dataframe = pd.DataFrame({"value": [0, 1, 2]})

        result = apply_percent_range_selection(dataframe, (80.0, 20.0))

        assert result.empty
