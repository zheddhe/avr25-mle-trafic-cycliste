"""Unit tests for interim dataset artifact manifest emission."""

from __future__ import annotations

from pathlib import Path

from src.ml.ingest.artifact_manifest_emission import (
    build_interim_dataset_artifact_manifest,
)


class TestBuildInterimDatasetArtifactManifest:
    """Unit tests for build_interim_dataset_artifact_manifest."""

    def test_builds_interim_dataset_manifest(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "data/raw/bike-counts.csv"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text("value\n1\n", encoding="utf-8")
        interim_path = tmp_path / "data/interim/counter-1/initial.csv"
        interim_path.parent.mkdir(parents=True)
        interim_path.write_text("value\n1\n", encoding="utf-8")

        manifest = build_interim_dataset_artifact_manifest(
            payload_path=interim_path,
            source_file_name=raw_path,
            sub_dir="counter-1",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
            producer_service="ml-ingest-test",
        )

        assert manifest.artifact_type == "interim_dataset"
        assert manifest.source.raw_file_name == "bike-counts.csv"
        assert manifest.producer.service == "ml-ingest-test"
        assert manifest.storage.local_path == "data/interim/counter-1/initial.csv"
