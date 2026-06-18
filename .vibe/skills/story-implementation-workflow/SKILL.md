---
name: story-implementation-workflow
description: >
  Use for implementing one GitHub story end-to-end in this repository, from
  issue reading and project-context review to branch work, tests, documentation,
  and pull-request handoff.
license: MIT
compatibility: Python 3.12+
user-invocable: true
allowed-tools:
  - read_file
  - write_file
  - search_replace
  - grep
  - glob
  - bash
  - task
  - ask_user_question
---

# Story Implementation Workflow

## Purpose

Use this project-specific skill when implementing a single GitHub story or
issue in `zheddhe/avr25-mle-trafic-cycliste`.

This skill specializes the global `feature-dev-workflow` defaults for this
repository. It keeps the same exploration, implementation, independent review,
validation, documentation, and PR handoff discipline, but adds the local MLOps
rules and story-completion expectations.

Do not use this workflow for trivial typo fixes, purely mechanical edits, or
work that is not attached to a clear story or issue.

## Required project context

Before planning or editing files, read the project agent rules:

1. Read `AGENTS.md` at the repository root.
2. Treat `AGENTS.md` as the canonical project rule file when both files exist
   unless the user explicitly says otherwise.
3. Read `docs/README.md` before documentation or architecture work.

The current `AGENTS.md` rules are valid for this skill: keep a Machine Learning
Engineer / MLOps perspective, use one branch per story, open PRs against `main`,
write code and tests in English, follow Python 3.12 and project lint settings,
run targeted validation first, and never merge without human review.

## Story intake

Start from one explicit story number, issue URL, or issue title.

Collect and restate:

- the story goal;
- acceptance criteria;
- out-of-scope items;
- impacted runtime area;
- expected user or pipeline behavior;
- validation expected by the story.

If the issue is ambiguous, inspect repository evidence before asking questions.
Ask the user only when the ambiguity blocks safe implementation.

## Workflow

### Phase 1 — Confirm scope and branch

- Confirm the target repository is `zheddhe/avr25-mle-trafic-cycliste`.
- Work from `main`.
- Create or use one focused branch for the story.
- Do not mix unrelated stories in the same branch.
- Identify whether the story is documentation-only, code-only, test-only, or
  mixed.

### Phase 2 — Explore before implementation

Delegate to `code-explorer` before editing files.

The exploration must identify:

- relevant files and directories;
- execution path or documentation ownership path;
- analogous project patterns;
- existing tests and fixtures;
- project constraints from `AGENTS.md`, `docs/README.md`, and `pyproject.toml`;
- mismatches between the issue wording and the repository state.

Explicitly tell the subagent to avoid implementation changes during this phase.
Do not assume subagents automatically apply repository skills or rules.

### Phase 3 — Plan a minimal story-sized change

Create a short implementation plan that maps issue acceptance criteria to files.

Keep the plan reviewable:

- prefer focused diffs;
- avoid broad cleanup;
- preserve current behavior unless the story explicitly changes it;
- protect the dev/prod runtime split;
- protect manifest-first artifact handoff;
- avoid widening Docker mounts, ports, or network exposure without story scope.

### Phase 4 — Implement with repository guardrails

Delegate implementation to `developer` when code changes are needed.

Explicitly require these skills in the delegated request:

- `developer-implementation-guardrails`;
- `python-ruff-check`.

Implementation rules:

- use English for code, variables, functions, classes, logs, comments,
  docstrings, tests, and API payloads;
- keep Python compatible with Python 3.12 and the current `pyproject.toml`;
- use lazy interpolation for logging calls;
- put reusable logic under `src/`;
- put tests under `tests/`;
- keep framework-neutral contracts independent from FastAPI, Airflow, Docker,
  or CLI internals unless the story targets integration code;
- do not commit secrets, generated artifacts, local `.env` values, runtime
  payloads, or large raw data files.

### Phase 5 — Add or update tests independently

Delegate independent review and test work to `test-engineer` after the
implementation exists.

Explicitly require these skills in the delegated request:

- `review-python-change`;
- `pytest-branch-coverage`.

Testing expectations:

- follow the repository's class-based pytest style;
- add or update tests for each functional contract change;
- prefer small deterministic fixtures;
- mock external sources and network calls in unit tests;
- cover error paths, boundary cases, and manifest or lineage behavior when
  relevant;
- avoid sleeps, broad mocks, or order-dependent tests.

### Phase 6 — Validate from focused to broad

Run the narrowest useful checks first, then broaden when the change warrants it.

Preferred order:

1. File syntax checks for documentation, YAML, JSON, shell, or config changes.
2. Focused lint for changed Python files.
3. Focused tests for changed behavior.
4. Coverage checks for modified logic when practical.
5. Broader repository checks such as `make lint`, `make tests`, or
   `make checks` when the impact is broad.

Report every command actually run. Do not claim unrun validation.
Separate introduced failures from pre-existing failures.

### Phase 7 — Update documentation only when justified

Update documentation when behavior, commands, architecture, setup, runtime
ownership, or operating procedures changed.

Use repository documentation ownership rules:

- current implemented operations belong under
  `docs/current-runtime-and-operations/`;
- implemented architecture rules belong under `docs/architecture-references/`;
- future-state design targets belong under `docs/remaining-work/`;
- story-level details belong in GitHub issues and pull requests.

When editing Markdown, apply `markdownlint-aware-writing`. Do not document
future behavior as current state before it is implemented.

### Phase 8 — Prepare PR handoff and stop

Apply `prepare-pr-with-human-stop`.

Return:

- implementation summary;
- files changed;
- acceptance criteria coverage;
- tests and validation commands with results;
- documentation impact;
- risks and follow-ups;
- suggested commit message;
- PR title and body draft;
- explicit statement that human review is required before merge.

Never merge directly to `main`, `master`, or `head`.

## Expected final output

The final response for a completed story must include:

- story reference;
- branch name;
- concise implementation summary;
- acceptance criteria status;
- validation report;
- documentation updates;
- PR handoff text;
- unresolved risks or blockers;
- a clear human stop before merge.
