# src/ml/ingest/ingest_utils.py
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def apply_percent_range_selection(
    df: pd.DataFrame,
    range_pct: tuple[float, float],
) -> pd.DataFrame:
    """
    Subset a DataFrame based on a percentage range.

    Args:
        df (pd.DataFrame): The input DataFrame, sorted chronologically.
        range_pct (tuple): Start and end percentage in (0.0 to 100.0).

    Returns:
        pd.DataFrame: A sliced copy of the DataFrame.
    """

    start_pct, end_pct = range_pct

    if df.empty or start_pct >= end_pct:
        logger.warning("Invalid or empty range provided — returning empty DataFrame.")
        return df.iloc[0:0].copy()

    n_rows = len(df)
    start_idx = int(n_rows * (start_pct / 100))
    end_idx = int(n_rows * (end_pct / 100))

    return df.iloc[start_idx:end_idx].copy()
