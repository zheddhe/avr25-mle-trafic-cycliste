"""Unit tests for ingest artifact manifest emission."""

from __future__ import annotations

from pathlib import Path

from src.artifacts.schemas import ArtifactType
from src.ml.ingest.artifact_manifest_emission import (
    build_interim_dataset_artifact_manifest,
    emit_interim_dataset_artifact_manifest,
)


class TestBuildInterimDatasetArtifactManifest:
    """Unit tests for build_interim_dataset_artifact_manifest."""

    def test_builds_interim_dataset_manifest_with_default_producer(
        self,
        tmp_path: Path,
    ):
        raw_path = tmp_path / "data/raw/bike-counts.csv"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text("value\n1\n", encoding="utf-8")
        payload_path = tmp_path / "data/interim/counter-1/initial.csv"
        payload_path.parent.mkdir(parents=True)
        payload_path.write_text("value\n1\n", encoding="utf-8")

        manifest = build_interim_dataset_artifact_manifest(
            payload_path=payload_path,
            source_file_name=raw_path,
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
        )

        assert manifest.artifact_type == ArtifactType.INTERIM_DATASET
        assert manifest.producer.service == "ml-ingest"
        assert manifest.source.raw_file_name == "bike-counts.csv"
        assert manifest.storage.local_path == "data/interim/counter-1/initial.csv"


class TestEmitInterimDatasetArtifactManifest:
    """Unit tests for emit_interim_dataset_artifact_manifest."""

    def test_promotes_interim_dataset_manifest(self, tmp_path: Path):
        payload_path = tmp_path / "data/interim/counter-1/initial.csv"
        payload_path.parent.mkdir(parents=True)
        payload_path.write_text("value\n1\n", encoding="utf-8")
        manifest_root = tmp_path / "artifacts/manifests"

        manifest = emit_interim_dataset_artifact_manifest(
            manifest_root=manifest_root,
            payload_path=payload_path,
            source_file_name="bike-counts.csv",
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
        )

        assert manifest is not None
        assert manifest.artifact_type == ArtifactType.INTERIM_DATASET
        assert (
            manifest_root / "interim_dataset" / "counter-1" / "current.json"
        ).exists()
