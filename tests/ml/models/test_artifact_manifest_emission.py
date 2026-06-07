"""Unit tests for prediction artifact manifest emission."""

from __future__ import annotations

from artifacts.manifest_store import read_current_manifest, read_manifest
from artifacts.schemas import ArtifactStatus, ArtifactType, StorageBackend
from src.ml.models.artifact_manifest_emission import (
    build_prediction_artifact_manifest,
    emit_prediction_artifact_manifest,
)


class TestPredictionArtifactManifestEmission:
    """Unit tests for prediction artifact manifest helpers."""

    def test_build_prediction_manifest_includes_checksum(self, tmp_path):
        prediction_path = tmp_path / "data/final/counter-a/y_full.csv"
        prediction_path.parent.mkdir(parents=True)
        prediction_path.write_text("y_true,y_pred\n1,1.2\n", encoding="utf-8")
        processed_path = tmp_path / "data/processed/counter-a/features.csv"
        processed_path.parent.mkdir(parents=True)
        processed_path.write_text("x,y\n1,2\n", encoding="utf-8")

        manifest = build_prediction_artifact_manifest(
            prediction_path=prediction_path,
            processed_path=processed_path,
            sub_dir="counter-a",
            repository_root=tmp_path,
            run_id="run-001",
            dataset_version="dataset-v1",
            model_version="model-v1",
            producer_service="ml-models-test",
        )

        assert manifest.artifact_type == ArtifactType.PREDICTIONS
        assert manifest.status == ArtifactStatus.VALIDATED
        assert manifest.run_id == "run-001"
        assert manifest.counter_id == "counter-a"
        assert manifest.source.raw_file_name == "features.csv"
        assert manifest.source.dataset_version == "dataset-v1"
        assert manifest.source.model_version == "model-v1"
        assert manifest.producer.service == "ml-models-test"
        assert manifest.storage.primary_backend == StorageBackend.LOCAL
        assert manifest.storage.local_path == "data/final/counter-a/y_full.csv"
        assert manifest.storage.checksum_sha256 is not None

    def test_emit_prediction_manifest_promotes_current_manifest(self, tmp_path):
        prediction_path = tmp_path / "data/final/counter-a/y_full.csv"
        prediction_path.parent.mkdir(parents=True)
        prediction_path.write_text("y_true,y_pred\n1,1.2\n", encoding="utf-8")
        processed_path = tmp_path / "data/processed/counter-a/features.csv"
        processed_path.parent.mkdir(parents=True)
        processed_path.write_text("x,y\n1,2\n", encoding="utf-8")
        manifest_root = tmp_path / "artifacts/manifests"

        manifest = emit_prediction_artifact_manifest(
            manifest_root=manifest_root,
            prediction_path=prediction_path,
            processed_path=processed_path,
            sub_dir="counter-a",
            repository_root=tmp_path,
            run_id="run-001",
            model_version="model-v1",
        )

        current_manifest = read_current_manifest(manifest_root)
        run_manifest = read_manifest(
            manifest_root / "runs/run-001/predictions-manifest.json"
        )
        assert manifest is not None
        assert current_manifest.run_id == manifest.run_id
        assert run_manifest.storage.checksum_sha256 == manifest.storage.checksum_sha256

    def test_emit_prediction_manifest_is_disabled_without_root(self, tmp_path):
        prediction_path = tmp_path / "data/final/counter-a/y_full.csv"
        prediction_path.parent.mkdir(parents=True)
        prediction_path.write_text("y_true,y_pred\n1,1.2\n", encoding="utf-8")
        processed_path = tmp_path / "data/processed/counter-a/features.csv"
        processed_path.parent.mkdir(parents=True)
        processed_path.write_text("x,y\n1,2\n", encoding="utf-8")

        manifest = emit_prediction_artifact_manifest(
            manifest_root=None,
            prediction_path=prediction_path,
            processed_path=processed_path,
            sub_dir="counter-a",
            repository_root=tmp_path,
        )

        assert manifest is None
