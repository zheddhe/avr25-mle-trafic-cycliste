"""Unit tests for feature dataset artifact manifest emission."""

from __future__ import annotations

from pathlib import Path

from src.ml.features.artifact_manifest_emission import (
    build_feature_dataset_artifact_manifest,
)


class TestBuildFeatureDatasetArtifactManifest:
    """Unit tests for build_feature_dataset_artifact_manifest."""

    def test_builds_feature_dataset_manifest(self, tmp_path: Path) -> None:
        interim_path = tmp_path / "data/interim/counter-1/initial.csv"
        interim_path.parent.mkdir(parents=True)
        interim_path.write_text("value\n1\n", encoding="utf-8")
        processed_path = tmp_path / "data/processed/counter-1/initial_with_feats.csv"
        processed_path.parent.mkdir(parents=True)
        processed_path.write_text("value,hour\n1,12\n", encoding="utf-8")

        manifest = build_feature_dataset_artifact_manifest(
            payload_path=processed_path,
            source_file_name=interim_path,
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
            dataset_version="interim-run-001",
            producer_service="ml-features-test",
        )

        assert manifest.artifact_type == "feature_dataset"
        assert manifest.source.raw_file_name == "initial.csv"
        assert manifest.source.dataset_version == "interim-run-001"
        assert manifest.producer.service == "ml-features-test"
