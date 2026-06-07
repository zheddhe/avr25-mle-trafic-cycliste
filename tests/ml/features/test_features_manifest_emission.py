"""Unit tests for feature artifact manifest emission."""

from __future__ import annotations

from pathlib import Path

from src.artifacts.schemas import ArtifactType
from src.ml.features.artifact_manifest_emission import (
    build_feature_dataset_artifact_manifest,
    emit_feature_dataset_artifact_manifest,
)


class TestBuildFeatureDatasetArtifactManifest:
    """Unit tests for build_feature_dataset_artifact_manifest."""

    def test_builds_feature_dataset_manifest_with_default_producer(
        self,
        tmp_path: Path,
    ):
        interim_path = tmp_path / "data/interim/counter-1/initial.csv"
        interim_path.parent.mkdir(parents=True)
        interim_path.write_text("value\n1\n", encoding="utf-8")
        payload_path = tmp_path / "data/processed/counter-1/initial_with_feats.csv"
        payload_path.parent.mkdir(parents=True)
        payload_path.write_text("value\n1\n", encoding="utf-8")

        manifest = build_feature_dataset_artifact_manifest(
            payload_path=payload_path,
            source_file_name=interim_path,
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
            dataset_version="interim-run-001",
        )

        assert manifest.artifact_type == ArtifactType.FEATURE_DATASET
        assert manifest.producer.service == "ml-features"
        assert manifest.source.raw_file_name == "initial.csv"
        assert manifest.source.dataset_version == "interim-run-001"
        assert manifest.storage.local_path == (
            "data/processed/counter-1/initial_with_feats.csv"
        )


class TestEmitFeatureDatasetArtifactManifest:
    """Unit tests for emit_feature_dataset_artifact_manifest."""

    def test_promotes_feature_dataset_manifest(self, tmp_path: Path):
        payload_path = tmp_path / "data/processed/counter-1/initial_with_feats.csv"
        payload_path.parent.mkdir(parents=True)
        payload_path.write_text("value\n1\n", encoding="utf-8")
        manifest_root = tmp_path / "artifacts/manifests"

        manifest = emit_feature_dataset_artifact_manifest(
            manifest_root=manifest_root,
            payload_path=payload_path,
            source_file_name="initial.csv",
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
        )

        assert manifest is not None
        assert manifest.artifact_type == ArtifactType.FEATURE_DATASET
        assert (
            manifest_root / "feature_dataset" / "counter-1" / "current.json"
        ).exists()
