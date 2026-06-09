"""Deprecated simplified production-like ML DAG.

The production-like runtime now mirrors the development DAG structure with:

- ``bike_traffic_orchestrator_dag.py``;
- ``bike_traffic_pipeline_dag.py`` containing ``bike_traffic_init`` and
  ``bike_traffic_daily``.

This module intentionally declares no DAG and can be deleted once the branch is
rebased locally.
"""
