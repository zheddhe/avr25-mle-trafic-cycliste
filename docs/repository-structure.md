# Repository structure and runtime ownership

This document clarifies how contributors should read and modify the repository
while Phase 6 keeps the current local development runtime stable and prepares a
future local production-like runtime.

The goal is documentation and repository hygiene only. This story does not move
large folders, restructure packages, or implement `docker/prod`.

## Guiding rules

- Keep the current `docker/dev` Compose workflow stable.
- Use `src/` for reusable application, API, ML, metrics, and pipeline logic.
- Keep deployment-specific Airflow DAGs close to their runtime assets.
- Treat `data`, `logs`, and `models` as runtime workspaces, not normal source.
- Keep local environment files out of Git.
- Introduce `docker/prod` only in a later production-like runtime story.

## Top-level responsibilities

| Path | Responsibility |
| ---- | -------------- |
| `README.md` | Project entry point and links to detailed documentation. |
| `Makefile` | Local setup, validation, DVC, and Docker Compose operation targets. |
| `.env.template` | Versioned template for local runtime variables. |
| `.env` | Local-only runtime values created from `.env.template`. |
| `pyproject.toml` | Local Python dependency groups and tooling configuration. |
| `uv.lock` | Versioned uv lockfile for reproducible local validation. |
| `src/` | Reusable FastAPI, ML, feature, model, metrics, and pipeline code. |
| `tests/` | Unit, integration, and regression tests. |
| `docker/` | Dockerfiles, runtime config, Airflow DAG wiring, and service helpers. |
| `docs/` | Architecture, operations, dependency, runtime, and repository docs. |
| `data/` | Data workspace shared with local containers and DVC stages. |
| `models/` | Generated model and forecast artifact workspace. |
| `logs/` | Generated local runtime logs. |
| `references/` | Static diagrams, exports, and explanatory material. |
| `.dvc/` | DVC metadata and ignored local DVC state. |
| `dvc.yaml` | Versioned DVC pipeline definition. |
| `params.yaml` | Versioned DVC and ML pipeline parameters. |

## Ownership table

| Path | Versioned | Generated | Mounted | DVC-managed | Local-only | Static ref |
| ---- | --------- | --------- | ------- | ----------- | ---------- | ---------- |
| `README.md` | Yes | No | No | No | No | No |
| `Makefile` | Yes | No | No | No | No | No |
| `.env.template` | Yes | No | No | No | No | No |
| `.env` | No | No | Yes | No | Yes | No |
| `pyproject.toml` | Yes | No | No | No | No | No |
| `uv.lock` | Yes | No | No | No | No | No |
| `src/` | Yes | No | Build context | No | No | No |
| `tests/` | Yes | No | No | No | No | No |
| `docker-compose.yaml` | Yes | No | No | No | No | No |
| `docker/dev/` | Yes | No | Partly | No | No | No |
| `docker/prod/` | Planned | No | Planned | No | No | No |
| `docker/common/` | Optional | No | Optional | No | No | No |
| `docs/` | Yes | No | No | No | No | No |
| `data/raw/` | Metadata or placeholders | Restored locally | Yes | Yes | Partly | No |
| `data/interim/` | Metadata only | Yes | Yes | Yes | Partly | No |
| `data/processed/` | Metadata only | Yes | Yes | Yes | Partly | No |
| `data/final/` | Metadata only | Yes | Yes | Yes | Partly | No |
| `models/` | Metadata only | Yes | Yes | Yes | Partly | No |
| `logs/` | No | Yes | Yes | No | Yes | No |
| `references/` | Yes | No | No | No | No | Yes |
| `.dvc/config` | Yes | No | No | No | No | No |
| `.dvc/config.local` | No | No | No | No | Yes | No |
| `.dvc/cache/` | No | Yes | No | Local cache | Yes | No |

`Partly` means a path can contain tracked placeholders or DVC metadata while
large data payloads and runtime outputs remain ignored, restored, or reproduced.

## Source code and runtime integration

`src/` is the right place for reusable Python logic that should be importable and
testable outside a specific Docker Compose service. This includes FastAPI code,
ML ingestion, feature engineering, model training, prediction, metrics, and
shared pipeline helpers.

Airflow DAGs under `docker/dev/airflow/dags/` are runtime integration assets.
They are coupled to local Airflow variables, connections, DockerOperator
settings, service names, mounted paths, and the current worker model. They should
not be moved into `src/` unless a separate packaging decision is made.

DAG placement may differ between `docker/dev` and a future `docker/prod` area if
operators, worker pools, queue names, image names, or artifact handoff mechanisms
diverge.

## Docker runtime layout

### `docker/dev`

`docker/dev` is the current local development runtime. It optimizes for host
visibility, debugging, demos, and fast iteration.

It may use broad bind mounts such as `./data`, `./logs`, and `./models`, expose
local UIs, and keep Airflow DAGs, config files, and helper scripts close to the
Airflow runtime that consumes them.

### `docker/prod`

`docker/prod` is the expected future location for local production-like Compose
assets. It should optimize for least privilege, reduced host mounts, stable
service discovery, explicit artifact handoff, and narrower runtime boundaries.

It is intentionally not implemented by this story.

### `docker/common`

`docker/common` may be introduced only when dev and prod runtimes share enough
Dockerfiles, scripts, or templates to justify a common layer. Until then,
runtime-specific clarity is preferable to premature abstraction.

## Documentation areas

| Document or area | Responsibility |
| ---------------- | -------------- |
| `docs/repository-structure.md` | Repository paths, runtime ownership, artifacts, and dev/prod split. |
| `docs/runtime-communication-matrix.md` | Current service-to-service communication and mount coupling. |
| `docs/runtime-security-boundaries.md` | Runtime identities and boundary design for Phase 6 and Phase 7. |
| `docs/ports-and-services.md` | Host-exposed ports, local URLs, and internal-only services. |
| `docs/dependency-strategy.md` | uv, image, dependency, and healthcheck strategy. |
| `references/` | Static diagrams, exports, and explanatory assets. |

Architecture documentation, operations documentation, dependency strategy,
runtime communication, runtime boundaries, ports, and static references should
remain distinguishable so contributors know where to update each decision.

## Data and DVC expectations

DVC describes reproducible data and model outputs while keeping large or
generated payloads out of normal Git history.

Current `dvc.yaml` ownership:

1. `import_raw_data` reads raw input and writes `data/interim/<scenario>`.
2. `build_features` reads interim data and writes `data/processed/<scenario>`.
3. `train_and_predict` reads processed data and writes `models/<scenario>` and
   `data/final/<scenario>`.

Directory expectations:

- `data/raw/` contains input data restored locally or tracked through DVC
  metadata when needed.
- `data/interim/`, `data/processed/`, and `data/final/` are generated pipeline
  outputs restored, reproduced, or mounted by local runtime services.
- `models/` is generated by training and prediction workflows.
- `logs/` is generated by local services and batch jobs.
- `.dvc/config` is versioned; `.dvc/config.local`, `.dvc/cache/`, and
  `.dvc/tmp/` are local-only.

The current development runtime intentionally mounts `./data`, `./logs`, and
`./models` for local visibility. A future production-like runtime should replace
that broad visibility with explicit data release and artifact handoff contracts.

## Scripts and helper ownership

A top-level `scripts/` folder, if introduced, should contain developer or
operational helpers only. Runtime-specific helpers should stay near the runtime
that executes them, for example under `docker/dev/airflow/scripts/`.

Reusable business logic should live in `src/` with tests rather than in shell
scripts or deployment folders.

## Local-only files

- `.env.template` is versioned and contains placeholders or safe defaults.
- `.env`, `.env.local`, `.env.dagshub`, and `.dvc/config.local` are local-only.
- Runtime config files under `docker/dev/airflow/config/` may contain local IDs
  or placeholders, but should not contain personal runtime values.

## Ignore-rule review

The current ignore strategy is consistent with this structure:

- root `.gitignore` ignores local Python caches, virtual environments, local env
  files, `logs`, `mlruns`, and `mlflow.db`;
- `.dvc/.gitignore` ignores local DVC config, cache, and temporary state;
- `data/*/.gitignore` and `models/.gitignore` keep generated scenario outputs
  out of normal Git commits.

There is no root `.dockerignore` at the time of this review. Adding one should
be validated separately because current image build contexts rely on repository
root visibility.

No ignore-rule change is required in this story.

## Migration strategy

Follow-up work should use this sequence:

1. Keep `docker/dev` as the stable local development runtime.
2. Design `docker/prod` as a separate local production-like runtime area.
3. Define artifact handoff before reducing `data`, `logs`, or `models` mounts.
4. Decide whether Airflow DAGs need separate dev and prod placement.
5. Introduce `docker/common` only when concrete shared assets justify it.
6. Update diagrams and operations docs after the target runtime is validated.
