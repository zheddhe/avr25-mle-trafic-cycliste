from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Resolve project root as the parent of tests/
TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
SRC = ROOT / "src"


def _add_to_sys_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# Make repository root and src/ importable for tests
_add_to_sys_path(ROOT)
_add_to_sys_path(SRC)


@pytest.fixture(scope="session", autouse=True)
def _chdir_to_repo_root():
    """
    Ensure tests run with repository root as the current working directory.
    Keeps relative paths stable for integration tests and CLI invocations.
    """
    prev = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(prev)
