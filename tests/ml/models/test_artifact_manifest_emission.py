"""Unit tests for prediction artifact manifest emission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ml.models.artifact_manifest_emission import (
    build_prediction_artifact_manifest,
    emit_prediction_artifact_manifest,
)


class TestBuildPredictionArtifactManifest:
    """Unit tests for build_prediction_artifact_manifest."""

    def test_builds_valid_manifest_with_relative_local_path(
        self,
        tmp_path: Path,
    ):
        prediction_path = tmp_path / "data/final/Sebastopol_N-S/y_full.csv"
        prediction_path.parent.mkdir(parents=True)
        prediction_path.write_text("value\n1\n", encoding="utf-8")
        processed_path = tmp_path / "data/processed/Sebastopol_N-S/input.csv"
        processed_path.parent.mkdir(parents=True)
        processed_path.write_text("value\n1\n", encoding="utf-8")

        manifest = build_prediction_artifact_manifest(
            prediction_path=prediction_path,
            processed_path=processed_path,
            sub_dir="Sebastopol_N-S",
            repository_root=tmp_path,
            run_id="manual-run",
            counter_id="counter-1",
            dataset_version="dataset-v1",
            model_version="model-v1",
            producer_service="ml-models-test",
            producer_image="ml-models:test",
        )

        assert manifest.run_id == "manual-run"
        assert manifest.counter_id == "counter-1"
        assert manifest.artifact_type == "predictions"
        assert manifest.status == "validated"
        assert manifest.source.raw_file_name == "input.csv"
        assert manifest.source.dataset_version == "dataset-v1"
        assert manifest.source.model_version == "model-v1"
        assert manifest.producer.service == "ml-models-test"
        assert manifest.producer.image == "ml-models:test"
        assert manifest.storage.primary_backend == "local"
        assert manifest.storage.local_path == (
            "data/final/Sebastopol_N-S/y_full.csv"
        )
        assert manifest.storage.checksum_sha256 is not None
        assert len(manifest.storage.checksum_sha256) == 64

    def test_raises_when_absolute_prediction_path_is_outside_repository(
        self,
        tmp_path: Path,
    ):
        prediction_path = tmp_path / "outside.csv"
        prediction_path.write_text("value\n1\n", encoding="utf-8")
        repository_root = tmp_path / "repo"
        repository_root.mkdir()

        with pytest.raises(ValueError, match="inside repository_root"):
            build_prediction_artifact_manifest(
                prediction_path=prediction_path,
                processed_path=prediction_path,
                sub_dir="Sebastopol_N-S",
                repository_root=repository_root,
                run_id="manual-run",
            )


class TestEmitPredictionArtifactManifest:
    """Unit tests for emit_prediction_artifact_manifest."""

    def test_returns_none_when_manifest_root_is_missing(self, tmp_path: Path):
        prediction_path = tmp_path / "prediction.csv"
        prediction_path.write_text("value\n1\n", encoding="utf-8")

        manifest = emit_prediction_artifact_manifest(
            manifest_root=None,
            prediction_path=prediction_path,
            processed_path=prediction_path,
            sub_dir="Sebastopol_N-S",
            repository_root=tmp_path,
        )

        assert manifest is None

    def test_promotes_manifest_and_current_pointer(self, tmp_path: Path):
        prediction_path = tmp_path / "data/final/Sebastopol_N-S/y_full.csv"
        prediction_path.parent.mkdir(parents=True)
        prediction_path.write_text("value\n1\n", encoding="utf-8")
        processed_path = tmp_path / "data/processed/Sebastopol_N-S/input.csv"
        processed_path.parent.mkdir(parents=True)
        processed_path.write_text("value\n1\n", encoding="utf-8")
        manifest_root = tmp_path / "artifacts/manifests"

        manifest = emit_prediction_artifact_manifest(
            manifest_root=manifest_root,
            prediction_path=prediction_path,
            processed_path=processed_path,
            sub_dir="Sebastopol_N-S",
            repository_root=tmp_path,
            run_id="manual-run",
            counter_id="counter-1",
            model_version="model-v1",
            promote=True,
        )

        assert manifest is not None
        run_manifest_path = (
            manifest_root
            / "runs"
            / "manual-run"
            / "predictions-manifest.json"
        )
        current_path = manifest_root / "current.json"
        assert run_manifest_path.exists()
        assert current_path.exists()
        assert manifest.counter_id == "counter-1"
        assert json.loads(current_path.read_text(encoding="utf-8"))["run_id"] == (
            "manual-run"
        )
