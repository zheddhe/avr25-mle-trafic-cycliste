"""
src/airflow/dags_common/utils.py
-----------------

Shared utilities for Airflow DAGs:
- Pydantic models for DAG configuration
- Functions to load config and list counters
- Required-variable helpers that fail fast with clear logging

This file must not declare any DAG object.
"""

from __future__ import annotations

import logging
from pathlib import Path

from airflow.exceptions import AirflowNotFoundException
from airflow.sdk import Variable
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("airflow.task")


# --------------------------------------------------------------------------- #
# Pydantic Models
# --------------------------------------------------------------------------- #
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
    """Scheduling configuration for the DAG."""

    initial_anchor_date: str
    daily_increment_pct: float  # No default: must be explicit in bike_dag_config.json


class DagCfg(BaseModel):
    """Top-level DAG configuration."""

    counters: dict[str, CounterCfg]
    scheduling: SchedulingCfg


# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def _missing(msg: str) -> str:
    """
    Raise a clear error for a required Airflow Variable that is missing.

    Logs at ERROR level so the message appears in the Airflow scheduler/worker
    logs even when the DAG is parsed.

    Parameters
    ----------
    msg : str
        Human-readable explanation of what is missing and how to fix it.

    Returns
    -------
    str
        Never returns; always raises.
    """
    logger.error(f"REQUIRED VARIABLE MISSING: {msg}")
    raise ValueError(msg)


def _required_var(name: str) -> str:
    """
    Read an Airflow Variable that MUST be set.

    Raises ``ValueError`` with a clear message (and ERROR-level log) when the
    variable is absent or empty.  This is the preferred way to read variables
    that have no sensible local default.

    Parameters
    ----------
    name : str
        Name of the Airflow Variable.

    Returns
    -------
    str
        The variable value.

    Raises
    ------
    ValueError
        If the variable is not found or is empty.
    """
    try:
        value = Variable.get(name)
        if not value:
            _missing(f"Airflow Variable '{name}' is defined but empty. "
                     f"Run 'make ops' to import variables from variables.json.")
        logger.debug(f"[utils] Variable '{name}' resolved")
        return value
    except AirflowNotFoundException:
        _missing(f"Airflow Variable '{name}' is not defined. "
                 f"Run 'make ops' to import variables from variables.json.")
        raise AirflowNotFoundException(f"Airflow Variable '{name}' is not defined.")


def _load_config() -> tuple[DagCfg, str]:
    """
    Load DAG configuration from Airflow Variable ``bike_dag_config``.

    The variable may contain:
    - Inline JSON
    - A path to a JSON file

    Returns
    -------
    Tuple[DagCfg, str]
        Parsed configuration and the selected counter_id.

    Raises
    ------
    ValueError
        If the variable is missing, the file path does not exist, or the JSON is invalid.
    """
    logger.info("[utils] Loading DAG config")
    cfg_ref = Variable.get("bike_dag_config")
    default_counter_id = _required_var("default_counter_id")

    if not cfg_ref:
        logger.error("[utils] bike_dag_config is empty")
        raise ValueError("bike_dag_config not defined")

    # Case 1: inline JSON
    if cfg_ref.strip().startswith("{"):
        logger.info("[utils] Config source: inline JSON")
        try:
            return DagCfg.model_validate_json(cfg_ref), default_counter_id
        except ValidationError as exc:
            logger.error(f"[utils] Invalid inline JSON: {exc}")
            raise ValueError(f"Invalid inline JSON in bike_dag_config: {exc}") from exc

    # Case 2: path to JSON file
    path = Path(cfg_ref)
    logger.info(f"[utils] Config source: file {path}")
    if not path.exists():
        logger.error(f"[utils] Config file not found: {cfg_ref}")
        raise ValueError(f"bike_dag_config invalid path: {cfg_ref}")

    try:
        cfg = DagCfg.model_validate_json(path.read_text())
        logger.info(f"[utils] Config loaded: {len(cfg.counters)} counters, default={default_counter_id}")
        return cfg, default_counter_id
    except ValidationError as exc:
        logger.error(f"[utils] Invalid JSON in {cfg_ref}: {exc}")
        raise ValueError(f"Invalid JSON content in {cfg_ref}: {exc}") from exc


def _list_counters_payload() -> list[dict[str, str]]:
    """
    Build a list of payloads for TriggerDagRunOperator expansion.

    Returns
    -------
    List[Dict[str, str]]
        List of {"counter_id": <id>} dicts.
    """
    logger.info("[utils] Listing counters")
    cfg, default_counter = _load_config()
    counters = list((cfg.counters or {}).keys())
    if not counters:
        logger.warning(f"[utils] No counters found, falling back to default: {default_counter}")
        counters = [default_counter]
    logger.info(f"[utils] Counters: {counters}")
    return [{"counter_id": cid} for cid in counters]
