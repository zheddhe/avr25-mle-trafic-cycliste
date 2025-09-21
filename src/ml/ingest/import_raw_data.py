# src/ml/ingest/import_raw_data.py
from __future__ import annotations

import os
import json
import sys
from pathlib import Path
import re
import logging
import unicodedata
import click
import pandas as pd
from typing import Optional

from src.ml.ingest.data_utils import apply_percent_range_selection


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def write_manifest(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -------------------------------------------------------------------
# Logs Management
# -------------------------------------------------------------------
log_dir = os.path.join("logs", "ml")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "import_raw_data.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def slugify_ascii(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    text_ascii = norm.encode("ascii", "ignore").decode("ascii")
    text_ascii = re.sub(r"[^A-Za-z0-9\-_]+", "_", text_ascii)
    text_ascii = re.sub(r"_+", "_", text_ascii).strip("_")
    return (text_ascii[:64] or "counter")


# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------
@click.command()
@click.option(
    "--raw-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the raw CSV file.",
)
@click.option(
    "--site",
    type=str,
    required=True,
    help="Exact value of 'nom_du_site_de_comptage'.",
)
@click.option(
    "--orientation",
    type=str,
    required=True,
    help="Exact value of 'orientation_compteur' (e.g. 'N-S').",
)
@click.option(
    "--range-start",
    type=float,
    default=0.0,
    show_default=True,
    help="Start percent [0..100] for chronological slice.",
)
@click.option(
    "--range-end",
    type=float,
    default=100.0,
    show_default=True,
    help="End percent [0..100] for chronological slice.",
)
@click.option(
    "--timestamp-col",
    type=str,
    default="date_et_heure_de_comptage",
    show_default=True,
    help="ISO8601 timestamp column in raw CSV.",
)
@click.option(
    "--sub-dir",
    type=click.Path(file_okay=False),
    default=None,
    help=("ASCII Target <sub-dir> under data/interim. Default derived from site and "
          "orientation."),
)
@click.option(
    "--interim-name",
    type=str,
    default="initial.csv",
    show_default=True,
    help="interim CSV filename inside data/interim/<sub-dir>.",
)
def main(
    raw_path: str,
    site: str,
    orientation: str,
    range_start: float,
    range_end: float,
    timestamp_col: str,
    sub_dir: Optional[str],
    interim_name: str,
) -> None:
    """
    Extract a single counter from the raw dataset and save an interim slice.
    """
    if not (0.0 <= range_start <= 100.0 and 0.0 <= range_end <= 100.0):
        logger.error("range-start/range-end must be within [0, 100]. Got (%s, %s).",
                     range_start, range_end)
        sys.exit(2)
    if range_start > range_end:
        logger.error("range-start must be <= range-end. Got (%s, %s).",
                     range_start, range_end)
        sys.exit(2)

    try:
        logger.info("Loading raw CSV [%s] ...", raw_path)
        df = pd.read_csv(raw_path, index_col=0)
    except Exception as exc:
        logger.exception("Failed to load raw CSV: %s", exc)
        sys.exit(1)

    key_cols = ["nom_du_site_de_comptage", "orientation_compteur"]
    missing = [c for c in key_cols + [timestamp_col] if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)

    grp = df.groupby(key_cols)
    key = (site, orientation)
    if key not in grp.groups:
        logger.warning("Counter not found: %s", key)
        sys.exit(1)

    df_counter = grp.get_group(key).copy()
    logger.info(f"Counter [{site} | {orientation}] has {len(df_counter)} rows.")

    # Ensure chronological order based on the timestamp string format in raw
    try:
        df_counter[timestamp_col] = pd.to_datetime(
            df_counter[timestamp_col],
            format="%Y-%m-%dT%H:%M:%S%z",
            utc=True,
        )
    except Exception:
        # fallback: let pandas infer if exact format fails
        df_counter[timestamp_col] = pd.to_datetime(
            df_counter[timestamp_col], utc=True
        )

    df_counter = df_counter.sort_values(timestamp_col).reset_index(drop=True)

    # Percent slice
    df_counter = apply_percent_range_selection(
        df_counter, (range_start, range_end)
    )
    if df_counter.empty:
        logger.warning("Slice produced an empty DataFrame.")
        sys.exit(1)

    # Output path
    if sub_dir is None:
        sub_dir = slugify_ascii(f"{site}_{orientation}")
    out_dir = os.path.join("data", "interim", sub_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, interim_name)

    df_counter.to_csv(out_path, index=True)
    logger.info(f"Saved interim slice to [{out_path}] ({len(df_counter)} rows).")

    # write a manifest if required in environment variable
    man = os.getenv("MANIFEST_INGEST")
    if man:
        try:
            write_manifest(man, {
                "inputs": {
                    "raw_path": str(Path(raw_path).resolve()),
                    "site": site,
                    "orientation": orientation,
                    "range": [range_start, range_end],
                    "timestamp_col": timestamp_col,
                },
                "outputs": {
                    "interim_path": str(out_path)
                },
                "run": {
                    "run_id": os.getenv("RUN_ID"),
                    "sub_dir": sub_dir
                }
            })
            logger.info("Ingest manifest written to [%s].", man)
        except Exception as exc:
            logger.warning("Failed to write manifest [%s]: %s", man, exc)

    logger.info("Data ingestion ended successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
