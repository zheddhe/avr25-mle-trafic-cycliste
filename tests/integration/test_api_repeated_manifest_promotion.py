from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.api.config import ApiSettings
from src.api.serving import load_predictions_from_manifests
from src.artifacts.manifest_store import promote_manifest
from src.artifacts.schemas import (
    SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactProducer,
    ArtifactSource,
    ArtifactStatus,
    ArtifactStorage,
    ArtifactType,
    StorageBackend,
)


@pytest.mark.integration
class TestApiRepeatedManifestPromotion:
    def test_loads_latest_manifest_after_repeated_promotions(
        self,
        tmp_path: Path,
    ) -> None:
        manifest_root = tmp_path / "manifests"
        first_payload = _write_prediction_payload(tmp_path, "first.csv", 1.0)
        second_payload = _write_prediction_payload(tmp_path, "second.csv", 2.0)
        first_manifest = _build_manifest(tmp_path, first_payload, "counter-a", "run-a")
        second_manifest = _build_manifest(tmp_path, second_payload, "counter-a", "run-b")

        promote_manifest(
            first_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )
        promote_manifest(
            second_manifest,
            manifest_root=manifest_root,
            repository_root=tmp_path,
        )

        result = load_predictions_from_manifests(
            ApiSettings(
                manifest_root=manifest_root,
                repository_root=tmp_path,
                counter_ids=("counter-a",),
            )
        )

        dataframe = result.predictions["counter-a"]
        assert result.artifacts["counter-a"].run_id == "run-b"
        assert dataframe.iloc[0]["y_pred"] == 2.0


def _write_prediction_payload(tmp_path: Path, name: str, y_pred: float) -> Path:
    payload_path = tmp_path / name
    dataframe = pd.DataFrame(
        [
            {
                "date_et_heure_de_comptage_local": "2026-01-01T00:00:00+01:00",
                "date_et_heure_de_comptage_utc": "2025-12-31T23:00:00+00:00",
                "y_true": 1,
                "y_pred": y_pred,
                "forecast_mode": False,
            }
        ]
    )
    dataframe.to_csv(payload_path)
    return payload_path


def _build_manifest(
    repository_root: Path,
    payload_path: Path,
    counter_id: str,
    run_id: str,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        artifact_type=ArtifactType.PREDICTIONS,
        status=ArtifactStatus.VALIDATED,
        run_id=run_id,
        counter_id=counter_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        producer=ArtifactProducer(service="integration-test"),
        source=ArtifactSource(raw_file_name="raw.csv"),
        storage=ArtifactStorage(
            primary_backend=StorageBackend.LOCAL,
            local_path=payload_path.relative_to(repository_root).as_posix(),
            checksum_sha256=hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        ),
    )
