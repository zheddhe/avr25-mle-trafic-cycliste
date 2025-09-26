# src/ml/ingest/ingest_utils.py
from __future__ import annotations

import os
import time
import pandas as pd
import logging
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
    _push_metrics(
        step=step, duration_s=duration_s, records=records,
        status=status, labels=labels
    )
