"""
src/airflow/dags_common/utils.py
-----------------

Shared utilities for Airflow DAGs:
- Pydantic models for DAG configuration
- Functions to load config and list counters

This file must not declare any DAG object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

from airflow.models import Variable
from pydantic import BaseModel, Field, ValidationError


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
    daily_increment_pct: float = 1.0


class DagCfg(BaseModel):
    """Top-level DAG configuration."""

    counters: Dict[str, CounterCfg]
    scheduling: SchedulingCfg


# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def _load_config() -> Tuple[DagCfg, str]:
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
    cfg_ref = Variable.get("bike_dag_config", default_var="")
    counter_id = Variable.get("bike_counter_id", default_var="Sebastopol_N-S_mlops")

    if not cfg_ref:
        raise ValueError("bike_dag_config not defined")

    # Case 1: inline JSON
    if cfg_ref.strip().startswith("{"):
        try:
            return DagCfg.model_validate_json(cfg_ref), counter_id
        except ValidationError as exc:
            raise ValueError(f"Invalid inline JSON in bike_dag_config: {exc}") from exc

    # Case 2: path to JSON file
    path = Path(cfg_ref)
    if not path.exists():
        raise ValueError(f"bike_dag_config invalid path: {cfg_ref}")

    try:
        return DagCfg.model_validate_json(path.read_text()), counter_id
    except ValidationError as exc:
        raise ValueError(f"Invalid JSON content in {cfg_ref}: {exc}") from exc


def _list_counters_payload() -> List[Dict[str, str]]:
    """
    Build a list of payloads for TriggerDagRunOperator expansion.

    Returns
    -------
    List[Dict[str, str]]
        List of {"counter_id": <id>} dicts.
    """
    cfg, default_counter = _load_config()
    counters = list((cfg.counters or {}).keys())
    if not counters:
        counters = [default_counter]
    return [{"counter_id": cid} for cid in counters]
