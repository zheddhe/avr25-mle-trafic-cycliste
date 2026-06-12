"""Shared utilities for production-like Airflow DAGs."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


def get_airflow_task_logger() -> logging.Logger:
    """Return the Airflow task logger used by DAG utilities."""

    return logging.getLogger("airflow.task")


LOGGER = get_airflow_task_logger()

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


class ConcurrencyCfg(BaseModel):
    """Bounded local production-like Airflow concurrency limits."""

    orchestrator_max_active_runs: int = Field(default=1, ge=1)
    orchestrator_max_active_tasks: int = Field(default=2, ge=1)
    child_max_active_runs: int = Field(default=2, ge=1)
    child_max_active_tasks: int = Field(default=4, ge=1)


class DagCfg(BaseModel):
    """Top-level DAG configuration."""

    counters: dict[str, CounterCfg]
    scheduling: SchedulingCfg
    concurrency: ConcurrencyCfg = Field(default_factory=ConcurrencyCfg)


def _load_config() -> tuple[DagCfg, str]:
    """Load the mounted production-like DAG configuration."""

    LOGGER.debug("[utils] Loading DAG config")
    raw_config = _read_config_source(DEFAULT_CONFIG_PATH)
    try:
        cfg = DagCfg.model_validate_json(raw_config)
    except ValidationError as exc:
        raise ValueError(f"Invalid bike DAG config: {exc}") from exc

    default_counter_id = next(iter(cfg.counters))
    LOGGER.debug(
        "[utils] Config loaded: %s counters, default=%s",
        len(cfg.counters),
        default_counter_id,
    )
    return cfg, default_counter_id


def _load_concurrency_config() -> ConcurrencyCfg:
    """Load only the bounded Airflow concurrency configuration."""

    cfg, _ = _load_config()
    return cfg.concurrency


def _read_config_source(cfg_ref: str) -> str:
    path = Path(cfg_ref)
    LOGGER.debug("[utils] Config source: file %s", path)
    if not path.exists():
        LOGGER.error("[utils] DAG config file not found: %s", path)
        raise ValueError(f"bike_dag_config invalid path: {cfg_ref}")

    return path.read_text(encoding="utf-8")


def _list_counters_payload() -> list[dict[str, str]]:
    """Build a list of payloads for TriggerDagRunOperator expansion."""

    LOGGER.debug("[utils] Listing counters")
    cfg, default_counter = _load_config()
    counters = list((cfg.counters or {}).keys())
    if not counters:
        LOGGER.warning(
            "[utils] No counters found, falling back to default: %s",
            default_counter,
        )
        counters = [default_counter]

    LOGGER.debug("[utils] Counters: %s", counters)
    return [{"counter_id": counter_id} for counter_id in counters]
