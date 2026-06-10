"""Integration tests for repository test topology rules."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

IGNORED_SOURCE_NAMES = {"__init__.py"}
NON_UNIT_TEST_DIRS = {"acceptance", "integration", "performance"}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*.py")
        if path.name not in IGNORED_SOURCE_NAMES
    }


def _expected_unit_test_path(source_file: Path) -> Path:
    return (
        Path("tests")
        / source_file.parent.relative_to("src")
        / f"test_{source_file.name}"
    )


def _is_non_unit_test(test_path: Path) -> bool:
    return len(test_path.parts) > 1 and test_path.parts[1] in NON_UNIT_TEST_DIRS


def test_source_files_have_mirrored_unit_test_files() -> None:
    repo_root = _repository_root()
    source_files = _python_files(repo_root / "src")
    test_files = _python_files(repo_root / "tests")
    unit_test_files = {
        Path("tests") / test_file
        for test_file in test_files
        if not _is_non_unit_test(Path("tests") / test_file)
    }

    expected_unit_test_files = {
        _expected_unit_test_path(Path("src") / source_file)
        for source_file in source_files
    }

    missing_files = sorted(expected_unit_test_files - unit_test_files)

    assert missing_files == []


def test_integration_and_acceptance_tests_use_registered_markers() -> None:
    repo_root = _repository_root()
    marker_by_dir = {
        "integration": "pytest.mark.integration",
        "acceptance": "pytest.mark.acceptance",
    }

    for directory_name, expected_marker in marker_by_dir.items():
        test_dir = repo_root / "tests" / directory_name
        for test_file in test_dir.rglob("test_*.py"):
            content = test_file.read_text(encoding="utf-8")
            assert expected_marker in content, test_file.as_posix()
