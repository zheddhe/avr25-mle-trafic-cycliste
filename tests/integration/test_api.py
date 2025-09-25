# tests/integration/test_api.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import pandas as pd
from src.api.main import app, df_predictions
import src.api.main as main

pytestmark = pytest.mark.integration

client = TestClient(app)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _auth_headers(username: str, password: str) -> dict[str, str]:
    import base64

    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# ---------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------
class TestApiIntegration:
    def test_verify_admin_access(self) -> None:
        resp = client.get("/verify", headers=_auth_headers("remy", "remy"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["message"] == "API is healthy."
        assert body["role"] == "admin"

    def test_verify_forbidden_for_user(self) -> None:
        resp = client.get("/verify", headers=_auth_headers("user1", "user1"))
        assert resp.status_code == 403

    def test_me_endpoint(self) -> None:
        resp = client.get("/me", headers=_auth_headers("user2", "user2"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "user2"
        assert body["role"] == "user"

    def test_counters_empty_predictions(self) -> None:
        # purge store
        df_predictions.clear()
        resp = client.get("/counters", headers=_auth_headers("user1", "user1"))
        assert resp.status_code == 418
        body = resp.json()
        assert body["type"] == "PredictionsNotLoaded"

    def test_refresh_admin(self) -> None:
        resp = client.post("/admin/refresh", headers=_auth_headers("remy", "remy"))
        assert resp.status_code == 200
        body = resp.json()
        assert "counters_before" in body
        assert "counters_after" in body
        root_path = Path(body["data_root"])
        assert list(root_path.parts[-2:]) == ["data", "final"]

    def test_get_predictions_with_data(self, monkeypatch) -> None:
        # mock of main df_predictions dataframe content
        data = {
            "date_et_heure_de_comptage_local": ["2025-09-23 08:00:00"],
            "date_et_heure_de_comptage_utc": ["2025-09-23 06:00:00"],
            "y_true": [123],
            "y_pred": [120.5],
            "forecast_mode": [False],
        }
        df = pd.DataFrame(data)
        main.df_predictions.clear()
        main.df_predictions["Sebastopol_N-S"] = df

        resp = client.get(
            "/predictions/Sebastopol_N-S",
            headers=_auth_headers("user1", "user1"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["item"]) == 1
        assert body["item"][0]["y_true"] == 123
        assert pytest.approx(body["item"][0]["y_pred"], rel=1e-6) == 120.5
