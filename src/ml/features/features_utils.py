# src/ml/features/features_utils.py
from __future__ import annotations

import os
import time
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import pytz
from contextlib import contextmanager
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway

PUSHGATEWAY_ADDR = os.getenv("PUSHGATEWAY_ADDR", "pushgateway:9091")
DISABLE_METRICS_PUSH = os.getenv("DISABLE_METRICS_PUSH", "0")

logger = logging.getLogger(__name__)


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
            format="%Y-%m-%d %H:%M:%S%z",
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


def _push_metrics(step: str, duration_s: float, records: int, status: str, labels: dict):
    """
    Envoie:
      - pipeline_task_duration_seconds{step, status}
      - pipeline_new_records_total{step} (incrément)
    Les 'labels' (ex: dag, task, run_id, site) sont passés en grouping_key
    pour segmenter par run.
    """
    # environment variable check to allow metric push
    if DISABLE_METRICS_PUSH == "1":
        logger.info("Push metrics to gateway is disabled")
        return

    reg = CollectorRegistry()

    g_dur = Gauge(
        "pipeline_task_duration_seconds",
        "Durée d'une étape batch",
        ["step", "status"],
        registry=reg,
    )
    c_rec = Counter(
        "pipeline_new_records_total",
        "Nouveaux enregistrements traités",
        ["step"],
        registry=reg,
    )

    g_dur.labels(step=step, status=status).set(float(duration_s))
    c_rec.labels(step=step).inc(int(max(records, 0)))

    logger.info(f"Pusing metrics to [{PUSHGATEWAY_ADDR}]...")
    push_to_gateway(
        PUSHGATEWAY_ADDR,
        job="ml_pipeline",
        grouping_key=labels,
        registry=reg,
    )
    logger.info("Metrics pushed to gateway")


@contextmanager
def track_pipeline_step(step: str, labels: dict):
    """
    Context manager qui mesure la durée automatiquement et pousse à la fin.
    Utilisation:
        with track_pipeline_step("ingest", labels) as m:
            # ... traitement ...
            m["records"] = nb_lignes
    """
    start = time.time()
    payload = {"records": 0}
    status = "success"
    try:
        yield payload
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start
        _push_metrics(
            step=step, duration_s=duration, records=payload["records"],
            status=status, labels=labels
        )


def push_once(step: str, records: int, duration_s: float, status: str, labels: dict):
    """Alternative simple si vous ne voulez pas de context manager."""
    # environment variable check to allow metric push
    if DISABLE_METRICS_PUSH == "1":
        logger.info("Push metrics to gateway is disabled")
        return

    _push_metrics(
        step=step, duration_s=duration_s, records=records,
        status=status, labels=labels
    )
