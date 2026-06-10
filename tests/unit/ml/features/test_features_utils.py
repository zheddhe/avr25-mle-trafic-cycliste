"""Unit tests for feature engineering utility functions."""

from __future__ import annotations

import pandas as pd

from src.ml.features.features_utils import (
    DatetimePeriodicsTransformer,
    extract_datetime_periodic_features,
)


class TestExtractDatetimePeriodicFeatures:
    """Unit tests for extract_datetime_periodic_features."""

    def test_adds_expected_datetime_columns(self) -> None:
        dataframe = pd.DataFrame(
            {"timestamp": ["2026-06-07 10:00:00+0000"]},
        )

        result = extract_datetime_periodic_features(dataframe, "timestamp")

        assert "timestamp_utc" in result.columns
        assert "timestamp_local" in result.columns
        assert "timestamp_hour" in result.columns
        assert "timestamp_sin_hour" in result.columns


class TestDatetimePeriodicsTransformer:
    """Unit tests for DatetimePeriodicsTransformer."""

    def test_transform_drops_original_timestamp_column(self) -> None:
        transformer = DatetimePeriodicsTransformer(timestamp_col="timestamp")
        dataframe = pd.DataFrame(
            {"timestamp": ["2026-06-07 10:00:00+0000"]},
        )

        result = transformer.transform(dataframe)

        assert "timestamp" not in result.columns
        assert "timestamp_utc" in result.columns
