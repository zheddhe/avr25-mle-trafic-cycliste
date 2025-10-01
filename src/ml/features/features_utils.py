# src/ml/features/features_utils.py
from __future__ import annotations

import os
import time
import pandas as pd
import numpy as np
import logging
import unicodedata
import re
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
import pytz
from contextlib import contextmanager
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway

PUSHGATEWAY_ADDR = os.getenv("PUSHGATEWAY_ADDR", "monitoring-pushgateway:9091")
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


def push_step_metrics(
    step: str,
    duration_s: float,
    records: int,
    status: str,
    labels: dict,
) -> None:
    """
    Push ETL metrics with unified labels.
    Labels on series: task, status, site, orientation.
    Grouping key: site, orientation (no task/dag/run_id).
    """
    if os.getenv("DISABLE_METRICS_PUSH") == "1":
        logger.info("Push metrics to gateway is disabled")
        return

    site = canonical_site(labels.get("site"))
    orientation = labels.get("orientation") or os.getenv("ORIENTATION", "NA")
    status = "success" if status == "success" else "failed"

    reg = CollectorRegistry()
    g_dur = Gauge(
        "bike_task_duration_seconds",
        "Batch step duration (seconds)",
        ["task", "status", "site", "orientation"],
        registry=reg,
    )
    c_rec = Counter(
        "bike_records",
        "Processed records",
        ["task", "site", "orientation"],
        registry=reg,
    )

    g_dur.labels(step, status, site, orientation).set(float(duration_s))
    c_rec.labels(step, site, orientation).inc(max(int(records), 0))

    logger.info(
        f"Pushing metrics to [{PUSHGATEWAY_ADDR}] "
        f"with grouping_key=[{site} {orientation}]"
    )
    push_to_gateway(
        PUSHGATEWAY_ADDR,
        job="bike-traffic",
        grouping_key={"site": site, "orientation": orientation},
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
        push_step_metrics(
            step=step, duration_s=duration, records=payload["records"],
            status=status, labels=labels
        )


def _slug(value: str) -> str:
    value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value


def canonical_site(raw: Optional[str]) -> str:
    """
    Return a canonical 'site' label, harmonized across all steps.
    Priority:
    1) explicit short name via SITE_SHORT (if provided)
    2) SITE (env) as-is
    3) best-effort slug from any raw value
    """
    site_short = os.getenv("SITE_SHORT")
    if site_short:
        return site_short

    if raw:
        return raw

    site = os.getenv("SITE")
    if site:
        return site

    # fallback: try to build something stable from a path-like
    site_path = os.getenv("SITE_PATH", "")
    return _slug(site_path) if site_path else "NA"
