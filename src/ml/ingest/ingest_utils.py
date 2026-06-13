"""Ingestion helpers for raw cyclist counter datasets."""

from __future__ import annotations

import pandas as pd

from src.common.logger import get_logger

LOGGER = get_logger(__name__)


def apply_percent_range_selection(
    df: pd.DataFrame,
    range_pct: tuple[float, float],
) -> pd.DataFrame:
    """Return a chronological slice selected by percentage bounds."""

    start_pct, end_pct = range_pct

    if df.empty or start_pct >= end_pct:
        LOGGER.warning("Invalid or empty range provided, returning empty DataFrame.")
        return df.iloc[0:0].copy()

    n_rows = len(df)
    start_idx = int(n_rows * (start_pct / 100))
    end_idx = int(n_rows * (end_pct / 100))

    return df.iloc[start_idx:end_idx].copy()
