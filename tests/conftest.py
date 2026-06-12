from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Resolve project root as the parent of tests/
TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
SRC = ROOT / "src"
ENV_LOADED = False


def _add_to_sys_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# Make repository root and src/ importable for tests
_add_to_sys_path(ROOT)
_add_to_sys_path(SRC)


def pytest_configure(config: pytest.Config) -> None:
    """Load local pytest environment before test modules are collected."""

    load_test_environment()


def env_file_path() -> Path:
    """Return the dotenv file used by pytest runtime tests."""

    env_file = Path(os.getenv("ENV_FILE", ".env"))
    if env_file.is_absolute():
        return env_file

    return ROOT / env_file


def load_test_environment() -> None:
    """Load .env values as local pytest source of truth."""

    global ENV_LOADED

    if ENV_LOADED:
        return

    env_file = env_file_path()
    if env_file.is_file():
        load_dotenv(dotenv_path=env_file, override=True)

    ENV_LOADED = True


def get_optional_env(name: str) -> str | None:
    """Return an optional test environment variable after dotenv loading."""

    load_test_environment()
    return os.getenv(name) or None


def get_required_env(name: str) -> str:
    """Return a required test environment variable or fail explicitly."""

    value = get_optional_env(name)
    if value:
        return value

    pytest.fail(
        f"Missing required test environment variable {name}. "
        f"Define it in {env_file_path()} or export it in the shell."
    )
    raise AssertionError


@pytest.fixture(scope="session")
def required_env_var() -> Callable[[str], str]:
    """Provide mandatory environment lookup to tests."""

    return get_required_env


@pytest.fixture(scope="session")
def optional_env_var() -> Callable[[str], str | None]:
    """Provide optional environment lookup to tests."""

    return get_optional_env


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
