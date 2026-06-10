#!/usr/bin/env python
"""Scan source files for French-language residuals.

The scan is intentionally heuristic. It flags likely French words in code,
documentation, and tests while allowing external Paris dataset column names that
remain part of the current ETL contract.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ALLOWED_TERMS = {
    "date_et_heure_de_comptage",
    "comptage_horaire",
    "nom_du_site_de_comptage",
    "orientation_compteur",
}

DEFAULT_ROOTS = ("src", "tests", "docker", "docs", "scripts")

FRENCH_PATTERNS = (
    r"\bVérif(?:ie|iez|ication)\b",
    r"\bdroits?\b",
    r"\bécriture\b",
    r"\blecture\b",
    r"\binaccessible\b",
    r"\bcompteurs?\b",
    r"\bprédictions?\b",
    r"\binstallés?\b",
    r"\bdepuis\b",
    r"\buniquement\b",
)

SKIPPED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "runtime",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Scan repository text files for French residuals.",
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Repository paths to scan.",
    )
    return parser.parse_args()


def iter_text_files(roots: list[str]) -> list[Path]:
    """Return text files to scan from the requested roots."""

    files: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        if root_path.is_file():
            files.append(root_path)
            continue
        for path in root_path.rglob("*"):
            if any(part in SKIPPED_DIRS for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                files.append(path)
    return sorted(files)


def scan_file(path: Path, pattern: re.Pattern[str]) -> list[str]:
    """Scan one file and return formatted findings."""

    findings: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line_number, line in enumerate(lines, start=1):
        sanitized_line = line
        for allowed_term in ALLOWED_TERMS:
            sanitized_line = sanitized_line.replace(allowed_term, "")
        if pattern.search(sanitized_line):
            findings.append(f"{path}:{line_number}: {line.strip()}")
    return findings


def main() -> int:
    """Run the scan and return a shell-compatible status code."""

    args = parse_args()
    pattern = re.compile("|".join(FRENCH_PATTERNS), flags=re.IGNORECASE)
    findings: list[str] = []
    for path in iter_text_files(args.roots):
        findings.extend(scan_file(path, pattern))

    if findings:
        print("\n".join(findings))
        return 1

    print("No French residual candidate found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
