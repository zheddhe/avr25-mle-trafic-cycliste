"""Unit tests for prediction artifact manifest emission."""

from __future__ import annotations

from pathlib import Path

from src.ml.models.artifact_manifest_emission import (
    build_prediction_artifact_manifest,
)


class TestBuildPredictionArtifactManifest:
    """Unit tests for build_prediction_artifact_manifest."""

    def test_builds_prediction_artifact_manifest(self, tmp_path: Path) -> None:
        prediction_path = tmp_path / "data/final/counter-1/y_full.csv"
        prediction_path.parent.mkdir(parents=True)
        prediction_path.write_text("value\n1\n", encoding="utf-8")
        processed_path = tmp_path / "data/processed/counter-1/input.csv"
        processed_path.parent.mkdir(parents=True)
        processed_path.write_text("value\n1\n", encoding="utf-8")

        manifest = build_prediction_artifact_manifest(
            prediction_path=prediction_path,
            processed_path=processed_path,
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
            model_version="model-v1",
            producer_service="ml-models-test",
        )

        assert manifest.artifact_type == "predictions"
        assert manifest.source.raw_file_name == "input.csv"
        assert manifest.source.model_version == "model-v1"
        assert manifest.producer.service == "ml-models-test"
