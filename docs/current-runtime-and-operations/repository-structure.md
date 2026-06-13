# Repository structure and runtime ownership

This document explains how repository paths are owned now that the project has
aligned local `docker/dev` and local production-like `docker/prod` runtimes.

## Guiding rules

- Use `src/` for reusable application, API, ML, metrics, artifacts, and runner
  logic.
- Keep deployment-specific Airflow DAGs close to their runtime assets.
- Treat root `data`, `logs`, and `models` as local experimentation and DVC
  workspaces.
- Treat `docker/dev/runtime` and `docker/prod/runtime` as ignored runtime
  workspaces.
- Use promoted artifact manifests as the runtime handoff contract.
- Keep local environment files and runtime payloads out of Git.
- Use runtime-specific Make targets or explicit `docker compose -f` commands.

## Top-level responsibilities

| Path | Responsibility |
| ---- | -------------- |
| `README.md` | Project entrypoint and links to detailed documentation. |
| `Makefile` | Repository setup, validation, DVC, and runtime target inclusion. |
| `.env.template` | Versioned local runtime variable template. |
| `.env` | Local-only runtime values created from `.env.template`. |
| `pyproject.toml` | Python project, dependency groups, pytest, coverage, and Ruff. |
| `uv.lock` | uv lockfile for reproducible local validation. |
| `src/` | API, runner, ML pipeline, artifacts, metrics, and shared helpers. |
| `tests/` | Unit, integration, regression, and acceptance tests. |
| `docker/` | Dev/prod Compose architecture, Dockerfiles, runtime configs, and DAGs. |
| `docs/` | Current runtime, architecture, assets, and remaining work documentation. |
| `data/` | Local experimentation and DVC data workspace. |
| `models/` | Local experimentation and DVC model artifact workspace. |
| `logs/` | Local non-Compose developer logs. |
| `references/` | Static diagrams, exports, and explanatory material. |
| `.dvc/` | DVC metadata and ignored local DVC state. |
| `dvc.yaml` | Versioned DVC pipeline definition. |
| `params.yaml` | Versioned DVC and ML pipeline parameters. |

## Ownership table

| Path | Versioned | Generated | Runtime-mounted | DVC-managed | Local-only |
| ---- | --------- | --------- | --------------- | ----------- | ---------- |
| `README.md`, `Makefile`, `pyproject.toml`, `uv.lock` | Yes | No | No | No | No |
| `.env.template` | Yes | No | No | No | No |
| `.env`, `.env.local`, `.env.dagshub` | No | No | Yes | No | Yes |
| `src/` | Yes | No | Build context | No | No |
| `tests/` | Yes | No | No | No | No |
| `docker/dev/` | Yes | No | Partly | No | No |
| `docker/dev/runtime/` | No | Yes | Yes | No | Yes |
| `docker/prod/` | Yes | No | Partly | No | No |
| `docker/prod/runtime/` | No | Yes | Yes | No | Yes |
| `docs/` | Yes | No | No | No | No |
| `data/raw/` | Metadata or placeholders | Restored locally | Selected read-only inputs | Yes | Partly |
| `data/interim/`, `data/processed/`, `data/final/` | Metadata only | Yes | Local/DVC only | Yes | Partly |
| `models/` | Metadata only | Yes | Local/DVC only | Yes | Partly |
| `logs/` | No | Yes | Local/DVC only | No | Yes |
| `.dvc/config` | Yes | No | No | No | No |
| `.dvc/config.local`, `.dvc/cache/`, `.dvc/tmp/` | No | Yes | No | Local cache | Yes |

`Partly` means a path can contain tracked placeholders or metadata while runtime
payloads remain ignored, restored, or reproduced.

## Source code and runtime integration

`src/` is the right place for reusable Python logic that should be importable and
testable outside a specific Docker Compose service. This includes FastAPI code,
ML ingestion, feature engineering, model training, metrics, artifacts, the runner
API, and shared helpers.

The runner API lives under `src/job_runner/` because it is reusable application
logic with API tests. It uses framework-neutral ML job contracts rather than
Airflow-specific or Docker-specific payloads.

Airflow DAGs under `docker/dev/airflow/dags/` and
`docker/prod/airflow/dags/` are runtime integration assets. They are coupled to
runtime service names, Airflow connections, mounted paths, and local operator
choices. They should not be moved into `src/` without a packaging story.

## Docker runtime layout

### `docker/dev`

`docker/dev` is the local development Compose runtime. It optimizes for host
visibility, debugging, demos, and fast iteration.

| Development path | Purpose |
| ---------------- | ------- |
| `docker/dev/runtime/data` | Generated runtime data. |
| `docker/dev/runtime/models` | Runtime model artifacts. |
| `docker/dev/runtime/logs` | Airflow, API, runner, and ML service logs. |
| `docker/dev/runtime/artifacts` | Manifest-first artifact handoff root. |

### `docker/prod`

`docker/prod` is the local production-like Compose runtime. It optimizes for
reduced host exposure, explicit runtime ownership, and narrower service
boundaries.

| Production-like path | Purpose |
| -------------------- | ------- |
| `docker/prod/runtime` | Local view and helpers for the production-like runtime. |
| `prod-runtime:/data` | Generated production-like data workspace. |
| `prod-runtime:/models` | Production-like model artifacts. |
| `prod-runtime:/logs` | Production-like runtime logs. |
| `prod-runtime:/artifacts` | Manifest-first artifact handoff root. |

The only current root data dependency for prod-like execution is the required
source CSV under `data/raw`, seeded into the runtime volume by the init service.

## Data and DVC expectations

DVC describes reproducible development data and model outputs while keeping large
payloads out of normal Git history.

Current `dvc.yaml` ownership:

1. `import_raw_data` reads raw input and writes `data/interim/<scenario>`.
2. `build_features` reads interim data and writes `data/processed/<scenario>`.
3. `train_and_predict` reads processed data and writes `models/<scenario>` and
   `data/final/<scenario>`.

Root `data`, `models`, and `logs` remain available for local Python commands,
DVC reproduction, notebooks, and explicit developer scripts. They should not be
used as the default write target for Airflow-driven Compose operations.

## Logging and runtime configuration

Business code uses shared helpers from `src/common/`:

- service entrypoints call `configure_logging(...)` once;
- internal modules obtain loggers with `get_logger(__name__)`;
- runtime configuration is read through `src.common.env`;
- production-like containers rely on process environment variables only;
- local `.env` loading is limited to test or developer harnesses before runtime
  code starts.

Mandatory runtime variables must fail fast with an explicit configuration error.
Optional variables must use an explicit default or return `None`.

Runtime logging details are owned by
[`runtime-logging.md`](runtime-logging.md). In short, API, runner, and ML service
files use `<service_name>_<hostname>.log`, while `run_id`, `trace_id`, `job_id`,
`job_type`, and `counter_id` stay inside log records for traceability.

## Documentation ownership

The documentation level rules are defined in [`../README.md`](../README.md).

| Area | Responsibility |
| ---- | -------------- |
| `docs/current-runtime-and-operations/` | Implemented commands, workspaces, service exposure, logging, dependencies, and runtime ownership. |
| `docs/architecture-references/` | Implemented cross-runtime boundaries, networks, communication paths, and runtime guardrails. |
| `docs/assets/` | Documentation-only icons and rendered diagrams. |
| `docs/remaining-work/` | Future improvement axes and not-yet-implemented design targets. |

Current-state docs should describe validated behavior only. Future-state design
or open gaps belong under `docs/remaining-work/` until implemented.

## Scripts and helper ownership

A top-level `scripts/` folder, if introduced, should contain developer or
operational helpers only. Runtime-specific helpers should stay near the runtime
that executes them, for example under `docker/dev/airflow/scripts/` or
`docker/prod/airflow/scripts/`.

Reusable business logic should live in `src/` with tests rather than in shell
scripts or deployment folders.

## Local-only files

- `.env.template` is versioned and contains placeholders or safe defaults.
- `.env`, `.env.local`, `.env.dagshub`, and `.dvc/config.local` are local-only.
- `docker/dev/runtime/` and `docker/prod/runtime/` are local-only and ignored
  except for placeholder files.
- Runtime config files under `docker/dev/airflow/config/` and
  `docker/prod/airflow/config/` may contain local placeholders, but should not
  contain personal runtime values.

## Ignore-rule review

The current ignore strategy is consistent with this structure:

- root `.gitignore` ignores local Python caches, virtual environments, local env
  files, `logs`, `mlruns`, and `mlflow.db`;
- `.dvc/.gitignore` ignores local DVC config, cache, and temporary state;
- `data/*/.gitignore` and `models/.gitignore` keep generated scenario outputs
  out of normal Git commits;
- runtime folders under `docker/dev` and `docker/prod` keep Compose outputs out
  of Git and DVC ownership.

There is no root `.dockerignore` at the time of this review. Adding one requires
validation because current image build contexts rely on repository root
visibility.
