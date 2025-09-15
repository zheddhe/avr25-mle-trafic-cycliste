# tests/integration/test_pipeline.py
from __future__ import annotations

import json
# import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Any
import pandas as pd
import pytest
import inspect
from click.testing import CliRunner

MODULE_ALIASES = {
    "import_raw_data": [
        "src.ml.data.import_raw_data",
        "ml.data.import_raw_data",
        "import_raw_data",
    ],
    "build_features": [
        "src.ml.features.build_features",
        "ml.features.build_features",
        "build_features",
    ],
    "train_and_predict": [
        "src.ml.models.train_and_predict",
        "ml.models.train_and_predict",
        "train_and_predict",
    ],
}

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _raw_csv_path(raw_file_name: str) -> Path:
    return _project_root() / "data" / "raw" / raw_file_name


def _interim_path(sub_dir: str, name: str) -> Path:
    return _project_root() / "data" / "interim" / sub_dir / name


def _processed_path(sub_dir: str, name: str) -> Path:
    return _project_root() / "data" / "processed" / sub_dir / name


def _final_dir(sub_dir: str) -> Path:
    return _project_root() / "data" / "final" / sub_dir


def _models_dir(sub_dir: str) -> Path:
    return _project_root() / "models" / sub_dir


def _make_runner():
    kwargs = {}
    if "mix_stderr" in inspect.signature(CliRunner.__init__).parameters:
        kwargs["mix_stderr"] = False
    return CliRunner(**kwargs)


def _invoke_click(script_key: str, argv: list[str]) -> int:
    """
    1) python -m <module>
    2) exécution d'un .py si trouvé (fallback)
    """
    aliases = MODULE_ALIASES.get(script_key, [script_key])

    # 1) python -m <module>
    for mod_name in aliases:
        cp = subprocess.run(
            [sys.executable, "-m", mod_name, *argv],
            capture_output=True,
            text=True,
        )
        if cp.returncode == 0:
            return 0
        if "No module named" not in (cp.stderr or ""):
            sys.stderr.write((cp.stdout or "") + (cp.stderr or ""))
            return int(cp.returncode)

    # 2) fallback fichiers (si jamais)
    fname = aliases[0].split(".")[-1] + ".py"
    candidates = [
        _project_root() / fname,
        _project_root() / "src" / "ml" / "data" / "import_raw_data.py"
        if script_key == "import_raw_data" else None,
        _project_root() / "src" / "ml" / "features" / "build_features.py"
        if script_key == "build_features" else None,
        _project_root() / "src" / "ml" / "models" / "train_and_predict.py"
        if script_key == "train_and_predict" else None,
    ]
    for p in [c for c in candidates if c is not None]:
        if p.exists():
            cp = subprocess.run(
                [sys.executable, str(p), *argv],
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0:
                sys.stderr.write((cp.stdout or "") + (cp.stderr or ""))
            return int(cp.returncode)

    sys.stderr.write(f"Could not locate module/file for: {script_key}\n")
    return 2


def _read_csv_nrows(p: Path) -> int:
    df = pd.read_csv(p, nrows=10_000)  # guard memory
    return int(len(df))


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def site_test() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Import scenarios from test_config.SITE_TEST. We cap to 5 scenarios.
    """
    import tests.integration.test_scenario as cfg  # type: ignore

    site_test = dict(cfg.SITE_TEST)  # copy
    assert isinstance(site_test, dict), "SITE_TEST must be a dict."
    return dict(list(site_test.items())[:5])


@pytest.fixture(scope="session")
def raw_file_name() -> str:
    try:
        import tests.integration.test_scenario as cfg  # type: ignore
        return cfg.RAW_FILE_NAME
    except Exception:
        from tests.integration import test_scenario as cfg  # type: ignore
        return cfg.RAW_FILE_NAME


@pytest.fixture(scope="session")
def ensure_raw_exists(raw_file_name: str) -> Path:
    p = _raw_csv_path(raw_file_name)
    if not p.exists():
        pytest.skip(f"Raw CSV not found: {p}")
    return p


@pytest.fixture(autouse=True)
def isolate_mlflow(tmp_path, monkeypatch):
    mlruns = tmp_path / "mlruns"
    mlruns.mkdir(parents=True, exist_ok=True)
    uri = mlruns.resolve().as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    yield


# ---------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------
def _params_from_scenario(
    key: Tuple[str, str], cfg: Dict[str, Any], raw_path: Path
) -> Dict[str, Any]:
    site, orientation = key
    sub_dir = cfg["sub_dir"]
    return {
        "site": site,
        "orientation": orientation,
        "sub_dir": sub_dir,
        "raw_path": str(raw_path),
        "interim_name": cfg["interim_file_name"],
        "processed_name": cfg["processed_file_name"],
        "range_start": float(cfg["range"][0]),
        "range_end": float(cfg["range"][1]),
        "ar": int(cfg["temp_feats"][0]),
        "mm": int(cfg["temp_feats"][1]),
        "roll": int(cfg["temp_feats"][2]),
        "test_ratio": float(cfg["test_ratio"]),
        "grid_iter": int(cfg.get("iter_grid_search", 0)),
    }


# ---------------------------------------------------------------------
# The integration test: import -> features -> train/predict
# ---------------------------------------------------------------------
class TestPipelineE2E:
    @pytest.mark.parametrize("scenario_idx", range(5))
    def test_pipeline_scenarios(
        self,
        scenario_idx: int,
        site_test: Dict[Tuple[str, str], Dict[str, Any]],
        ensure_raw_exists: Path,
    ) -> None:
        if scenario_idx >= len(site_test):
            pytest.skip("Fewer than 5 scenarios in SITE_TEST.")

        key = list(site_test.keys())[scenario_idx]
        cfg = site_test[key]
        params = _params_from_scenario(key, cfg, ensure_raw_exists)

        # ---------------------------
        # 1) import_raw_data
        # ---------------------------
        sub_dir = params["sub_dir"]
        interim_path = _interim_path(params["sub_dir"], params["interim_name"])

        rc_ing = _invoke_click(
            "import_raw_data",
            [
                "--raw-path",
                str(params["raw_path"]),
                "--site",
                str(params["site"]),
                "--orientation",
                str(params["orientation"]),
                "--range-start",
                str(params["range_start"]),
                "--range-end",
                str(params["range_end"]),
                "--sub-dir",
                str(sub_dir),
                "--interim-name",
                str(interim_path.name),
            ],
        )
        assert rc_ing == 0, "import_raw_data failed."
        assert interim_path.exists(), "Interim CSV not created."
        assert _read_csv_nrows(interim_path) > 0, "Interim CSV is empty."

        # ---------------------------
        # 2) build_features
        # ---------------------------
        interim_path = _interim_path(params["sub_dir"], params["interim_name"])
        processed_name = params["processed_name"]
        processed_path = _processed_path(params["sub_dir"], params["processed_name"])

        rc_feat = _invoke_click(
            "build_features",
            [
                "--interim-path",
                str(interim_path),
                "--processed-name",
                str(processed_name),
                "--timestamp-col",
                "date_et_heure_de_comptage",
            ],
        )
        assert rc_feat == 0, "build_features failed."
        assert processed_path.exists(), "Processed CSV not created."
        nrows = _read_csv_nrows(processed_path)
        assert nrows > 0, "Processed CSV is empty."

        # Basic feature sanity: time columns present for training
        df_head = pd.read_csv(processed_path, nrows=5)
        for col in [
            "date_et_heure_de_comptage_utc",
            "date_et_heure_de_comptage_local",
        ]:
            assert col in df_head.columns, f"Missing column: {col}"

        # ---------------------------
        # 3) train_and_predict
        # ---------------------------
        rc_train = _invoke_click(
            "train_and_predict",
            [
                "--processed-path",
                str(processed_path),
                "--sub-dir",
                str(params["sub_dir"]),
                "--target-col",
                "comptage_horaire",
                "--ts-col-utc",
                "date_et_heure_de_comptage_utc",
                "--ts-col-local",
                "date_et_heure_de_comptage_local",
                "--ar",
                str(params["ar"]),
                "--mm",
                str(params["mm"]),
                "--roll",
                str(params["roll"]),
                "--test-ratio",
                str(params["test_ratio"]),
                "--grid-iter",
                str(params["grid_iter"]),
            ],
        )
        assert rc_train == 0, "train_and_predict failed."

        # ---------------------------
        # Artefacts existence checks
        # ---------------------------
        # Data final
        fdir = _final_dir(params["sub_dir"])
        for name in [
            "X_train.csv",
            "X_test.csv",
            "y_train.csv",
            "y_test.csv",
            "y_train_pred.csv",
            "y_test_pred.csv",
            "y_full.csv",
        ]:
            assert (fdir / name).exists(), f"Missing final file: {name}"

        # Models dir
        mdir = _models_dir(params["sub_dir"])
        for name in [
            "pipe_model.pkl",
            "ar_transformer.pkl",
            "hyperparams.json",
            "metrics.json",
        ]:
            assert (mdir / name).exists(), f"Missing model artifact: {name}"

        # Parse metrics.json for minimal structure
        with open(mdir / "metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        assert isinstance(metrics, dict), "metrics.json must be a dict"
        # Common metrics (keys may vary but usually exist)
        for k in ["R2", "RMSE", "MAE"]:
            if k in metrics:
                assert isinstance(metrics[k], (int, float))
