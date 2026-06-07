"""Unit tests for ML pipeline artifact manifest coherence."""

from __future__ import annotations

from pathlib import Path

from src.artifacts.schemas import ArtifactType
from src.ml.features.artifact_manifest_emission import (
    build_feature_dataset_artifact_manifest,
)
from src.ml.ingest.artifact_manifest_emission import (
    build_interim_dataset_artifact_manifest,
)
from src.ml.models.artifact_manifest_emission import build_prediction_artifact_manifest


class TestMlPipelineArtifactManifestCoherence:
    """Unit tests for manifest coherence across ingest, features, and models."""

    def test_pipeline_steps_build_non_colliding_manifest_scopes(
        self,
        tmp_path: Path,
    ):
        raw_path = tmp_path / "data/raw/bike-counts.csv"
        interim_path = tmp_path / "data/interim/counter-1/initial.csv"
        processed_path = tmp_path / "data/processed/counter-1/initial_with_feats.csv"
        prediction_path = tmp_path / "data/final/counter-1/y_full.csv"
        for path in (raw_path, interim_path, processed_path, prediction_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("value\n1\n", encoding="utf-8")

        common = {
            "sub_dir": "counter-1",
            "repository_root": tmp_path,
            "run_id": "run-001",
            "counter_id": "counter-1",
        }
        ingest_manifest = build_interim_dataset_artifact_manifest(
            payload_path=interim_path,
            source_file_name=raw_path,
            producer_service="ml-ingest-test",
            **common,
        )
        feature_manifest = build_feature_dataset_artifact_manifest(
            payload_path=processed_path,
            source_file_name=interim_path,
            dataset_version=ingest_manifest.run_id,
            producer_service="ml-features-test",
            **common,
        )
        prediction_manifest = build_prediction_artifact_manifest(
            prediction_path=prediction_path,
            processed_path=processed_path,
            model_version="model-run-001",
            producer_service="ml-models-test",
            **common,
        )

        assert ingest_manifest.storage.local_path == "data/interim/counter-1/initial.csv"
        assert feature_manifest.storage.local_path == (
            "data/processed/counter-1/initial_with_feats.csv"
        )
        assert prediction_manifest.storage.local_path == "data/final/counter-1/y_full.csv"
        assert feature_manifest.source.raw_file_name == "initial.csv"
        assert prediction_manifest.source.raw_file_name == "initial_with_feats.csv"
        assert feature_manifest.source.dataset_version == ingest_manifest.run_id
        assert prediction_manifest.source.model_version == "model-run-001"
        assert {
            ingest_manifest.artifact_type,
            feature_manifest.artifact_type,
            prediction_manifest.artifact_type,
        } == {
            ArtifactType.INTERIM_DATASET,
            ArtifactType.FEATURE_DATASET,
            ArtifactType.PREDICTIONS,
        }
