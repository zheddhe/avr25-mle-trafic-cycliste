# src/ml/ingest/ingest_utils.py
from __future__ import annotations

import os
import time
import pandas as pd
import logging
import unicodedata
import re
from typing import Optional
from contextlib import contextmanager
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway

PUSHGATEWAY_ADDR = os.getenv("PUSHGATEWAY_ADDR", "monitoring-pushgateway:9091")
DISABLE_METRICS_PUSH = os.getenv("DISABLE_METRICS_PUSH", "0")

logger = logging.getLogger(__name__)


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
