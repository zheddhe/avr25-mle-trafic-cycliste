import pandas as pd
import numpy as np
from typing import Tuple
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import pytz

logger = logging.getLogger(__name__)


def train_test_split_time_aware(
    df: pd.DataFrame,
    timestamp_cols: list,
    target_col: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series]:
    """
    Chronological train/test split, preserving datetime columns for visualization.

    Args:
        df: DataFrame containing all data.
        timestamp_cols: list of columns related to time (e.g., ['_utc', '_local']).
        target_col: Name of the target column.
        test_size: Fraction of data to use for testing.

    Returns:
        X_train, X_train_dates, X_test, X_test_dates, y_train, y_test
    """
    df = df.copy()

    # Extract specifically datetime cols for later usage in time series
    features_dates = df[timestamp_cols].copy()

    # Features/target split
    features = df.drop(columns=timestamp_cols + [target_col])
    target = df[target_col]

    # Chronological split
    n_test = int(len(df) * test_size)
    X_train = features[:-n_test]
    X_train_dates = features_dates[:-n_test]
    X_test = features[-n_test:]
    X_test_dates = features_dates[-n_test:]
    y_train = target[:-n_test]
    y_test = target[-n_test:]

    return X_train, X_train_dates, X_test, X_test_dates, y_train, y_test


def apply_percent_range_selection(df: pd.DataFrame,
                                  range_pct: tuple[float, float]) -> pd.DataFrame:
    """
    Subset a DataFrame based on a percentage range.

    Args:
        df (pd.DataFrame): The input DataFrame, sorted chronologically.
        range_pct (tuple): Start and end percentage in (0.0 to 100.0).

    Returns:
        pd.DataFrame: A sliced copy of the DataFrame.
    """
    start_pct, end_pct = range_pct

    # Sanity checks
    if df.empty or start_pct >= end_pct:
        logger.warning("Invalid or empty range provided — returning empty DataFrame.")
        return df.iloc[0:0].copy()

    n = len(df)
    start_idx = int(n * (start_pct / 100))
    end_idx = int(n * (end_pct / 100))

    return df.iloc[start_idx:end_idx].copy()


class ColumnFilterTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to select a predefined subset of columns from a DataFrame.

    Parameters:
        columns_to_keep (list of str): list of column names to retain.
    """

    def __init__(self, columns_to_keep: list[str]):
        self.columns_to_keep = columns_to_keep

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [col for col in self.columns_to_keep if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns in input: {missing_cols}")
        return X[self.columns_to_keep]


def extract_datetime_periodic_features(
    df: pd.DataFrame,
    timestamp_col: str,
    tz_local: str = "Europe/Paris"
) -> pd.DataFrame:
    """
    Parse ISO8601 timestamps in `timestamp_col`, convert to UTC then to local time,
    and extract calendar and periodic (sin/cos) components.

    Args:
        df: Input DataFrame.
        timestamp_col: Column with ISO8601 timestamp strings.
        tz_local: Timezone for conversion.

    Returns:
        pd.DataFrame: Enriched copy of df.
    """
    df = df.copy()
    try:
        df[f"{timestamp_col}_utc"] = pd.to_datetime(
            df[timestamp_col],
            format="%Y-%m-%dT%H:%M:%S%z",
            utc=True
        )
        df[f"{timestamp_col}_local"] = (
            df[f"{timestamp_col}_utc"]
            .dt.tz_convert(pytz.timezone(tz_local))
        )
        ts = df[f"{timestamp_col}_local"]
        df[f"{timestamp_col}_year"] = ts.dt.year
        df[f"{timestamp_col}_month"] = ts.dt.month
        df[f"{timestamp_col}_day"] = ts.dt.day
        df[f"{timestamp_col}_day_of_year"] = ts.dt.dayofyear
        df[f"{timestamp_col}_day_of_week"] = ts.dt.dayofweek
        df[f"{timestamp_col}_hour"] = ts.dt.hour
        df[f"{timestamp_col}_week"] = ts.dt.isocalendar().week
        df[f"{timestamp_col}_week_end"] = df[
            f"{timestamp_col}_day_of_week"
        ].apply(lambda x: 1 if x in [5, 6] else 0)
        df[f"{timestamp_col}_sin_hour"] = np.sin(
            2 * np.pi * df[f"{timestamp_col}_hour"] / 24
        )
        df[f"{timestamp_col}_cos_hour"] = np.cos(
            2 * np.pi * df[f"{timestamp_col}_hour"] / 24
        )
        df[f"{timestamp_col}_sin_day_of_week"] = np.sin(
            2 * np.pi * df[f"{timestamp_col}_day_of_week"] / 7
        )
        df[f"{timestamp_col}_cos_day_of_week"] = np.cos(
            2 * np.pi * df[f"{timestamp_col}_day_of_week"] / 7
        )
        df[f"{timestamp_col}_sin_month"] = np.sin(
            2 * np.pi * df[f"{timestamp_col}_month"] / 12
        )
        df[f"{timestamp_col}_cos_month"] = np.cos(
            2 * np.pi * df[f"{timestamp_col}_month"] / 12
        )
        df[f"{timestamp_col}_sin_week"] = np.sin(
            2 * np.pi * df[f"{timestamp_col}_week"] / 52
        )
        df[f"{timestamp_col}_cos_week"] = np.cos(
            2 * np.pi * df[f"{timestamp_col}_week"] / 52
        )
        df[f"{timestamp_col}_sin_day_of_year"] = np.sin(
            2 * np.pi * df[f"{timestamp_col}_day_of_year"] / 365
        )
        df[f"{timestamp_col}_cos_day_of_year"] = np.cos(
            2 * np.pi * df[f"{timestamp_col}_day_of_year"] / 365
        )
        return df

    except Exception as exc:
        logger.error(
            "Error in datetime feature extraction for '%s': %s",
            timestamp_col, exc
        )
        raise


class DatetimePeriodicsTransformer(BaseEstimator, TransformerMixin):
    """
    scikit-learn transformer that extracts datetime components and periodic features
    from a timestamp column, and drops the original timestamp col.

    Parameters:
        timestamp_col (str): name of the timestamp column in ISO8601 format.
    """

    def __init__(self, timestamp_col: str):
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_t = extract_datetime_periodic_features(X, timestamp_col=self.timestamp_col)
        cols_to_drop = [self.timestamp_col]
        return X_t.drop(columns=cols_to_drop, errors="ignore")
