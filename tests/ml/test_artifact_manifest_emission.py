"""Unit tests for ML pipeline artifact manifest emission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts.schemas import ArtifactType
from src.ml.artifact_manifest_emission import (
    build_dataset_artifact_manifest,
    emit_dataset_artifact_manifest,
)
from src.ml.models.artifact_manifest_emission import build_prediction_artifact_manifest


class TestBuildDatasetArtifactManifest:
    """Unit tests for build_dataset_artifact_manifest."""

    def test_builds_valid_interim_dataset_manifest(self, tmp_path: Path):
        raw_path = tmp_path / "data/raw/bike-counts.csv"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text("timestamp,value\n2026-06-07,1\n", encoding="utf-8")
        interim_path = tmp_path / "data/interim/counter-1/initial.csv"
        interim_path.parent.mkdir(parents=True)
        interim_path.write_text("timestamp,value\n2026-06-07,1\n", encoding="utf-8")

        manifest = build_dataset_artifact_manifest(
            artifact_type=ArtifactType.INTERIM_DATASET,
            payload_path=interim_path,
            source_file_name=raw_path,
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
            producer_service="ml-ingest-test",
            producer_image="ml-ingest:test",
        )

        assert manifest.artifact_type == ArtifactType.INTERIM_DATASET
        assert manifest.status == "validated"
        assert manifest.source.raw_file_name == "bike-counts.csv"
        assert manifest.producer.service == "ml-ingest-test"
        assert manifest.storage.local_path == "data/interim/counter-1/initial.csv"
        assert manifest.storage.checksum_sha256 is not None

    def test_builds_valid_feature_dataset_manifest(self, tmp_path: Path):
        interim_path = tmp_path / "data/interim/counter-1/initial.csv"
        interim_path.parent.mkdir(parents=True)
        interim_path.write_text("timestamp,value\n2026-06-07,1\n", encoding="utf-8")
        processed_path = tmp_path / "data/processed/counter-1/initial_with_feats.csv"
        processed_path.parent.mkdir(parents=True)
        processed_path.write_text("timestamp,value,hour\n2026-06-07,1,12\n", encoding="utf-8")

        manifest = build_dataset_artifact_manifest(
            artifact_type=ArtifactType.FEATURE_DATASET,
            payload_path=processed_path,
            source_file_name=interim_path,
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
            dataset_version="interim-run-001",
            producer_service="ml-features-test",
        )

        assert manifest.artifact_type == ArtifactType.FEATURE_DATASET
        assert manifest.source.raw_file_name == "initial.csv"
        assert manifest.source.dataset_version == "interim-run-001"
        assert manifest.storage.local_path == (
            "data/processed/counter-1/initial_with_feats.csv"
        )

    def test_raises_when_absolute_payload_path_is_outside_repository(
        self,
        tmp_path: Path,
    ):
        outside_path = tmp_path / "outside.csv"
        outside_path.write_text("value\n1\n", encoding="utf-8")
        repository_root = tmp_path / "repo"
        repository_root.mkdir()

        with pytest.raises(ValueError, match="inside repository_root"):
            build_dataset_artifact_manifest(
                artifact_type=ArtifactType.INTERIM_DATASET,
                payload_path=outside_path,
                source_file_name="raw.csv",
                sub_dir="counter-1",
                repository_root=repository_root,
                run_id="run-001",
            )


class TestEmitDatasetArtifactManifest:
    """Unit tests for emit_dataset_artifact_manifest."""

    def test_returns_none_when_manifest_root_is_missing(self, tmp_path: Path):
        payload_path = tmp_path / "data/interim/counter-1/initial.csv"
        payload_path.parent.mkdir(parents=True)
        payload_path.write_text("value\n1\n", encoding="utf-8")

        manifest = emit_dataset_artifact_manifest(
            manifest_root=None,
            artifact_type=ArtifactType.INTERIM_DATASET,
            payload_path=payload_path,
            source_file_name="raw.csv",
            sub_dir="counter-1",
            repository_root=tmp_path,
        )

        assert manifest is None

    def test_promotes_dataset_manifest_and_current_pointer(self, tmp_path: Path):
        payload_path = tmp_path / "data/interim/counter-1/initial.csv"
        payload_path.parent.mkdir(parents=True)
        payload_path.write_text("value\n1\n", encoding="utf-8")
        manifest_root = tmp_path / "artifacts/manifests"

        manifest = emit_dataset_artifact_manifest(
            manifest_root=manifest_root,
            artifact_type=ArtifactType.INTERIM_DATASET,
            payload_path=payload_path,
            source_file_name="raw.csv",
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
            promote=True,
        )

        assert manifest is not None
        run_manifest_path = (
            manifest_root
            / "interim_dataset"
            / "counter-1"
            / "run-001"
            / "manifest.json"
        )
        current_path = manifest_root / "interim_dataset" / "counter-1" / "current.json"
        assert run_manifest_path.exists()
        assert current_path.exists()
        assert json.loads(current_path.read_text(encoding="utf-8"))["run_id"] == (
            "run-001"
        )


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
        ingest_manifest = build_dataset_artifact_manifest(
            artifact_type=ArtifactType.INTERIM_DATASET,
            payload_path=interim_path,
            source_file_name=raw_path,
            producer_service="ml-ingest-test",
            **common,
        )
        feature_manifest = build_dataset_artifact_manifest(
            artifact_type=ArtifactType.FEATURE_DATASET,
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
