# src/monitoring/metrics_push.py
from __future__ import annotations
import os, time, typing as t
from contextlib import contextmanager
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway

PUSHGATEWAY_ADDR = os.getenv("PUSHGATEWAY_ADDR", "pushgateway:9091")

def _push_metrics(step: str, duration_s: float, records: int, status: str, labels: dict):
    """
    Envoie:
      - pipeline_task_duration_seconds{step, status}
      - pipeline_new_records_total{step} (incrément)
    Les 'labels' (ex: dag, task, run_id, site) sont passés en grouping_key
    pour segmenter par run.
    """
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
    # Utiliser inc() pour rester monotone
    c_rec.labels(step=step).inc(int(max(records, 0)))

    push_to_gateway(
        PUSHGATEWAY_ADDR,
        job="ml_pipeline",
        grouping_key=labels,   # ex: {"dag":"init_load_and_deploy_docker","task":"etl.ingest","run_id":"..."}
        registry=reg,
    )

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
        _push_metrics(step=step, duration_s=duration, records=payload["records"], status=status, labels=labels)

def push_once(step: str, records: int, duration_s: float, status: str, labels: dict):
    """Alternative simple si vous ne voulez pas de context manager."""
    _push_metrics(step=step, duration_s=duration_s, records=records, status=status, labels=labels)
