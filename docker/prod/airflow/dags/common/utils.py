"""Shared utilities for production-like Airflow DAGs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    from airflow.sdk import Variable
except ImportError:  # pragma: no cover - Airflow 2 compatibility fallback.
    from airflow.models import Variable

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("airflow.task")

DEFAULT_CONFIG_PATH = "/opt/airflow/config/bike_dag_config.json"


class ModelingCfg(BaseModel):
    """Modeling hyperparameters for one counter."""

    ar: int = 7
    mm: int = 1
    roll: int = 24
    test_ratio: float = 0.25
    grid_iter: int = 0


class CounterCfg(BaseModel):
    """Configuration for one counter."""

    raw_file_name: str
    site: str
    orientation: str
    interim_name: str
    processed_name: str
    final_basename: str = "y_full.csv"
    modeling: ModelingCfg = Field(default_factory=ModelingCfg)


class SchedulingCfg(BaseModel):
    """Scheduling configuration shared by init and daily DAGs."""

    initial_anchor_date: str
    daily_increment_pct: float


class DagCfg(BaseModel):
    """Top-level DAG configuration."""

    counters: dict[str, CounterCfg]
    scheduling: SchedulingCfg


def _get_variable(name: str, default: str | None = None) -> str | None:
    try:
        return Variable.get(name)
    except Exception:
        return default


def _load_config() -> tuple[DagCfg, str]:
    """Load DAG configuration from Airflow Variable or mounted JSON file."""

    logger.info("[utils] Loading DAG config")
    cfg_ref = _get_variable("bike_dag_config", DEFAULT_CONFIG_PATH)
    if cfg_ref is None:
        raise ValueError("bike_dag_config is not defined")

    raw_config = _read_config_source(cfg_ref)
    try:
        cfg = DagCfg.model_validate_json(raw_config)
    except ValidationError as exc:
        raise ValueError(f"Invalid bike DAG config: {exc}") from exc

    default_counter_id = _get_variable("default_counter_id")
    if not default_counter_id:
        default_counter_id = next(iter(cfg.counters))

    logger.info(
        "[utils] Config loaded: %s counters, default=%s",
        len(cfg.counters),
        default_counter_id,
    )
    return cfg, default_counter_id


def _read_config_source(cfg_ref: str) -> str:
    if cfg_ref.strip().startswith("{"):
        logger.info("[utils] Config source: inline JSON")
        json.loads(cfg_ref)
        return cfg_ref

    path = Path(cfg_ref)
    logger.info("[utils] Config source: file %s", path)
    if not path.exists():
        raise ValueError(f"bike_dag_config invalid path: {cfg_ref}")

    return path.read_text(encoding="utf-8")


def _list_counters_payload() -> list[dict[str, str]]:
    """Build a list of payloads for TriggerDagRunOperator expansion."""

    logger.info("[utils] Listing counters")
    cfg, default_counter = _load_config()
    counters = list((cfg.counters or {}).keys())
    if not counters:
        logger.warning(
            "[utils] No counters found, falling back to default: %s",
            default_counter,
        )
        counters = [default_counter]

    logger.info("[utils] Counters: %s", counters)
    return [{"counter_id": counter_id} for counter_id in counters]
