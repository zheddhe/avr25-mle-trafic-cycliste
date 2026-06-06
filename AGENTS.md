# Project-specific agent rules

## Project context

- Treat this repository as a Python/MLOps platform for Paris bike traffic
  prediction.
- Keep a Machine Learning Engineer / MLOps perspective: reproducibility,
  artifact traceability, orchestration, monitoring, API serving, and runtime
  boundaries matter.
- Work from `main` and create one branch per issue, story, or bugfix.
- Open PRs against `main`; never merge without human review.

## Code rules

- Use English for code, variables, functions, classes, logs, comments,
  docstrings, tests, and API payloads.
- Keep Python code compatible with Python 3.12, PEP8, Ruff, and the current
  `pyproject.toml`.
- Prefer small, cohesive commits grouped by logical intent, not one commit per
  file.
- Put reusable Python logic in `src/`; put tests under `tests/`.
- Keep framework-neutral contracts independent from FastAPI, Airflow, Docker,
  or CLI internals unless the story explicitly targets integration code.

## Tests and validation

- Follow the existing class-based pytest style when adding unit tests.
- Add or update tests for every functional contract change.
- Prefer targeted validation first, then broader checks when relevant:
  `make lint`, `make tests`, `make checks`.
- Mention commands actually run in the PR body. Do not claim unrun validation.

## Documentation

- Keep the root `README.md` concise and project-oriented.
- Put runtime commands and workspace ownership in
  `docs/current-runtime-and-operations/`.
- Put cross-runtime architecture rules in `docs/architecture-references/`.
- Put Phase 8 target designs and implementation notes in
  `docs/next-phase-design/`.
- When a design becomes implemented, update wording from future-state to
  current-state and leave remaining gaps explicit.

## Runtime and artifacts

- Preserve the dev/prod split: root `data/`, `models/`, and `logs/` are
  development/DVC workspaces; `docker/prod/runtime/` is local production-like
  runtime space.
- Prefer explicit artifact manifests over implicit filesystem discovery.
- Do not add secrets, local `.env` values, generated artifacts, or runtime
  payloads to Git.
- Do not widen Docker mounts, exposed ports, or network access without a story
  explaining the runtime impact.
