from fastapi import FastAPI, Query
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Traffic Prediction API",
    description="Expose precomputed traffic forecasts from counters in Paris",
    version="1.0.0",
)

PREDICTIONS_FILE = Path("/models/predictions.parquet")


@app.get("/predictions", tags=["predictions"])
def get_all_predictions():
    """Return all available predictions"""
    if not PREDICTIONS_FILE.exists():
        return {"error": "Predictions file not found"}
    df = pd.read_parquet(PREDICTIONS_FILE)
    return df.to_dict(orient="records")


@app.get("/predictions/{counter_id}", tags=["predictions"])
def get_predictions_by_counter(counter_id: str):
    """Return predictions for a specific counter"""
    if not PREDICTIONS_FILE.exists():
        return {"error": "Predictions file not found"}
    df = pd.read_parquet(PREDICTIONS_FILE)
    df_filtered = df[df["counter_id"] == counter_id]
    return df_filtered.to_dict(orient="records")


@app.get("/predictions/by_date", tags=["predictions"])
def get_predictions_by_date(date: str = Query(..., example="2025-09-08")):
    """Return predictions for a specific date"""
    if not PREDICTIONS_FILE.exists():
        return {"error": "Predictions file not found"}
    df = pd.read_parquet(PREDICTIONS_FILE)
    df_filtered = df[df["date"] == date]
    return df_filtered.to_dict(orient="records")
