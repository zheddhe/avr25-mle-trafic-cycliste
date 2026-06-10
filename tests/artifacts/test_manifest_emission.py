"""Unit tests for canonical artifact manifest emission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts.manifest_emission import build_artifact_manifest, emit_artifact_manifest
from src.artifacts.schemas import ArtifactType


class TestBuildArtifactManifest:
    """Unit tests for build_artifact_manifest."""

    def test_builds_valid_interim_dataset_manifest(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "data/raw/bike-counts.csv"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text("timestamp,value\n2026-06-07,1\n", encoding="utf-8")
        interim_path = tmp_path / "data/interim/counter-1/initial.csv"
        interim_path.parent.mkdir(parents=True)
        interim_path.write_text("timestamp,value\n2026-06-07,1\n", encoding="utf-8")

        manifest = build_artifact_manifest(
            artifact_type=ArtifactType.INTERIM_DATASET,
            payload_path=interim_path,
            source_file_name=raw_path,
            sub_dir="counter-1",
            default_producer_service="ml-ingest-test",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
            producer_image="ml-ingest:test",
        )

        assert manifest.artifact_type == ArtifactType.INTERIM_DATASET
        assert manifest.status == "validated"
        assert manifest.source.raw_file_name == "bike-counts.csv"
        assert manifest.producer.service == "ml-ingest-test"
        assert manifest.storage.local_path == "data/interim/counter-1/initial.csv"
        assert manifest.storage.checksum_sha256 is not None

    def test_builds_valid_feature_dataset_manifest(self, tmp_path: Path) -> None:
        interim_path = tmp_path / "data/interim/counter-1/initial.csv"
        interim_path.parent.mkdir(parents=True)
        interim_path.write_text("timestamp,value\n2026-06-07,1\n", encoding="utf-8")
        processed_path = tmp_path / "data/processed/counter-1/initial_with_feats.csv"
        processed_path.parent.mkdir(parents=True)
        processed_path.write_text(
            "timestamp,value,hour\n2026-06-07,1,12\n",
            encoding="utf-8",
        )

        manifest = build_artifact_manifest(
            artifact_type=ArtifactType.FEATURE_DATASET,
            payload_path=processed_path,
            source_file_name=interim_path,
            sub_dir="counter-1",
            default_producer_service="ml-features-test",
            repository_root=tmp_path,
            run_id="run-001",
            counter_id="counter-1",
            dataset_version="interim-run-001",
        )

        assert manifest.artifact_type == ArtifactType.FEATURE_DATASET
        assert manifest.source.raw_file_name == "initial.csv"
        assert manifest.source.dataset_version == "interim-run-001"
        assert manifest.storage.local_path == (
            "data/processed/counter-1/initial_with_feats.csv"
        )

    def test_raises_when_payload_is_outside_repository(
        self,
        tmp_path: Path,
    ) -> None:
        outside_path = tmp_path / "outside.csv"
        outside_path.write_text("value\n1\n", encoding="utf-8")
        repository_root = tmp_path / "repo"
        repository_root.mkdir()

        with pytest.raises(ValueError, match="inside repository_root"):
            build_artifact_manifest(
                artifact_type=ArtifactType.INTERIM_DATASET,
                payload_path=outside_path,
                source_file_name="raw.csv",
                sub_dir="counter-1",
                repository_root=repository_root,
                run_id="run-001",
            )


class TestEmitArtifactManifest:
    """Unit tests for emit_artifact_manifest."""

    def test_returns_none_when_manifest_root_is_missing(
        self,
        tmp_path: Path,
    ) -> None:
        payload_path = tmp_path / "data/interim/counter-1/initial.csv"
        payload_path.parent.mkdir(parents=True)
        payload_path.write_text("value\n1\n", encoding="utf-8")

        manifest = emit_artifact_manifest(
            manifest_root=None,
            artifact_type=ArtifactType.INTERIM_DATASET,
            payload_path=payload_path,
            source_file_name="raw.csv",
            sub_dir="counter-1",
            repository_root=tmp_path,
        )

        assert manifest is None

    def test_promotes_dataset_manifest_and_current_pointer(
        self,
        tmp_path: Path,
    ) -> None:
        payload_path = tmp_path / "data/interim/counter-1/initial.csv"
        payload_path.parent.mkdir(parents=True)
        payload_path.write_text("value\n1\n", encoding="utf-8")
        manifest_root = tmp_path / "artifacts/manifests"

        manifest = emit_artifact_manifest(
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
