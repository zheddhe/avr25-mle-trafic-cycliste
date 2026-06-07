"""Shared technical metrics helpers for ML pipeline steps."""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    pushadd_to_gateway,
)

DEFAULT_PUSHGATEWAY_ADDR = "unknown_address:9091"
METRICS_JOB_NAME = "bike-traffic"

logger = logging.getLogger(__name__)


def slug_label_value(value: str) -> str:
    """Normalize a free-form value into a Prometheus-safe label fragment."""

    normalized = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")


def canonical_site(raw: str | None) -> str:
    """
    Return a stable site label shared by all ML pipeline steps.

    Priority:
    1. explicit short name through ``SITE_SHORT``;
    2. raw label passed by the caller;
    3. ``SITE`` environment value;
    4. best-effort slug from ``SITE_PATH``;
    5. ``NA`` fallback.
    """

    site_short = os.getenv("SITE_SHORT")
    if site_short:
        return site_short

    if raw:
        return raw

    site = os.getenv("SITE")
    if site:
        return site

    site_path = os.getenv("SITE_PATH", "")
    return slug_label_value(site_path) if site_path else "NA"


def push_step_metrics(
    step: str,
    duration_s: float,
    records: int,
    status: str,
    labels: dict[str, Any],
) -> None:
    """
    Push common technical batch metrics for one ML pipeline step.

    Metrics push stays disabled when ``DISABLE_METRICS_PUSH=1`` so unit tests
    and local commands can run without a Pushgateway.
    """

    if os.getenv("DISABLE_METRICS_PUSH", "1") == "1":
        logger.info("Push metrics to gateway is disabled")
        return

    site = canonical_site(labels.get("site"))
    orientation = labels.get("orientation") or os.getenv("ORIENTATION", "NA")
    normalized_status = "success" if status == "success" else "failed"
    pushgateway_addr = os.getenv("PUSHGATEWAY_ADDR", DEFAULT_PUSHGATEWAY_ADDR)

    registry = CollectorRegistry()
    duration_gauge = Gauge(
        "bike_task_duration_seconds",
        "Batch step duration (seconds)",
        ["task", "status", "site", "orientation"],
        registry=registry,
    )
    records_counter = Counter(
        "bike_records",
        "Processed records",
        ["task", "site", "orientation"],
        registry=registry,
    )

    duration_gauge.labels(
        step,
        normalized_status,
        site,
        orientation,
    ).set(float(duration_s))
    records_counter.labels(step, site, orientation).inc(max(int(records), 0))

    logger.info(
        "Pushing metrics to [%s] with grouping_key=[%s %s]",
        pushgateway_addr,
        site,
        orientation,
    )
    pushadd_to_gateway(
        pushgateway_addr,
        job=METRICS_JOB_NAME,
        grouping_key={"site": site, "orientation": orientation},
        registry=registry,
    )
    logger.info("Metrics pushed to gateway")


@contextmanager
def track_pipeline_step(step: str, labels: dict[str, Any]) -> Iterator[dict[str, int]]:
    """
    Measure a pipeline step duration and push common technical metrics.

    Usage:
        with track_pipeline_step("ingest", labels) as metrics:
            metrics["records"] = len(df)
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
            step=step,
            duration_s=duration,
            records=payload["records"],
            status=status,
            labels=labels,
        )
