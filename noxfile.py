# -*- coding: utf-8 -*-
import nox  # type: ignore
import shutil
import stat
import time
import os
from pathlib import Path
import glob

PYTHON_VERSION = "3.12"


def _make_writable(func, path, exc_info):
    """Clear read-only bit then retry (Windows-friendly)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except FileNotFoundError:
        # Already gone: fine for cleaning.
        return


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except PermissionError:
        # Try once after clearing RO bit.
        _make_writable(os.unlink, str(path), None)


def _safe_rmtree(path: Path, retries: int = 3, delay: float = 0.25) -> None:
    """Robust rmtree that tolerates transient FS races on Windows."""
    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_make_writable)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _iter_targets(patterns):
    for pattern in patterns:
        for target in glob.glob(pattern):
            yield Path(target)


def _remove_paths(session: nox.Session, patterns) -> None:
    """Remove files/dirs matching given glob patterns, robustly."""
    # Do not delete the active Nox venv of this session.
    active_nox_dir = None
    if getattr(session, "virtualenv", None):
        active_nox_dir = Path(session.virtualenv.location)

    for path in _iter_targets(patterns):
        if not path.exists():
            continue
        # Skip the active env of the current session.
        if active_nox_dir and (
            path == active_nox_dir or active_nox_dir in path.parents
        ):
            session.log(f"Skip active nox venv: {path}")
            continue

        if path.is_dir():
            _safe_rmtree(path)
        else:
            _safe_unlink(path)
        session.log(f"Removed {path}")
    # Best-effort cleanup for scattered artifacts
    for pyc in Path(".").rglob("*.pyc"):
        _safe_unlink(pyc)
    for cover in Path(".").rglob("*,cover"):
        _safe_unlink(cover)
    for cache in Path(".").rglob("__pycache__"):
        if cache.exists():
            _safe_rmtree(cache)


@nox.session(python=PYTHON_VERSION,
             reuse_venv=True,
             name="cleanall")
def cleanall(session: nox.Session) -> None:
    """Remove temporary project files and build artifacts."""
    patterns = [
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "build",
        "dist",
        "*.egg-info",
        # Optional: clean other Nox envs, but not the current one.
        ".nox/*",
        # Optional: project venv if you also use uv/poetry, etc.
        ".venv",
    ]
    _remove_paths(session, patterns)
    session.log("Project cleaned.")


@nox.session(python=PYTHON_VERSION,
             venv_backend="uv",
             name="build",
             reuse_venv=True)
def build(session):
    """Run code linting and full test suite with coverage and HTML report."""
    # Synchronisation de l'environnement avec uv.lock
    # Point uv to the Nox-managed venv (instead of creating .venv)
    session.env["UV_PROJECT_ENVIRONMENT"] = str(session.virtualenv.location)
    # Sync locked deps + extras into the Nox venv
    session.run("uv", "sync",
                "--extra", "test",
                "--extra", "dev",
                "--extra", "py312",
                external=True)
    session.run("flake8")
    session.run("pytest")
    session.log("Build session complete. Coverage report in htmlcov/index.html")
