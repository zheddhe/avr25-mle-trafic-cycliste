# Repository structure and runtime ownership

This document clarifies how contributors should read and modify the repository
now that the project has symmetric local Compose runtime entrypoints for
development and local production-like validation.

## Guiding rules

- Use `src/` for reusable application, API, ML, metrics, runner, and pipeline
  logic.
- Keep deployment-specific Airflow DAGs close to their runtime assets.
- Treat root `data`, `logs`, and `models` as local experimentation and DVC
  workspaces.
- Treat `docker/dev/runtime` as the ignored host-visible development Compose
  runtime workspace.
- Treat the Docker volume `prod-runtime` as the local production-like runtime
  workspace.
- Use promoted artifact manifests as the runtime handoff contract.
- Keep local environment files out of Git.
- Keep dev and prod Compose files structurally comparable.
- Use runtime-specific Make targets or explicit `docker compose -f` commands;
  no root-level Compose entrypoint is supported.

## Top-level responsibilities

| Path | Responsibility |
| ---- | -------------- |
| `README.md` | Project entry point and links to detailed documentation. |
| `Makefile` | Repository setup, validation, DVC, environment helper, and runtime Makefile inclusion. |
| `.env.template` | Versioned template for local runtime variables. |
| `.env` | Local runtime values created from `.env.template`; not tracked. |
| `pyproject.toml` | Local Python dependency groups and tooling configuration. |
| `uv.lock` | Versioned uv lockfile for reproducible local validation. |
| `src/` | Reusable FastAPI, ML, feature, model, metrics, runner, and pipeline code. |
| `tests/` | Unit, integration, and regression tests. |
| `docker/` | Dockerfiles, runtime config, Airflow DAG wiring, Compose files, and service helpers. |
| `docs/` | Current runtime docs, architecture references, and active design docs. |
| `data/` | Local experimentation and DVC data workspace. |
| `models/` | Local experimentation and DVC model artifact workspace. |
| `logs/` | Local experimentation and non-Compose developer logs. |
| `references/` | Static diagrams, exports, and explanatory material. |
| `.dvc/` | DVC metadata and ignored local DVC state. |
| `dvc.yaml` | Versioned DVC pipeline definition. |
| `params.yaml` | Versioned DVC and ML pipeline parameters. |

## Runtime ownership table

| Resource | Versioned | Runtime use | Owner |
| -------- | --------- | ----------- | ----- |
| `src/` | Yes | Built into API, runner, and ML service images. | Reusable application code. |
| `tests/` | Yes | Local validation. | Test suite. |
| `docker/dev/` | Yes | Development Dockerfiles, DAGs, config, and Compose wiring. | Development runtime. |
| `docker/dev/runtime/` | No | Host-visible data, models, logs, and artifact manifests. | Development Compose runtime. |
| `docker/prod/` | Yes | Production-like Dockerfiles, DAGs, config, and Compose wiring. | Production-like runtime. |
| `prod-runtime` Docker volume | No | Production-like data, models, logs, and artifact manifests. | Production-like Compose runtime. |
| `data/` | Metadata or DVC outputs | Local Python, notebooks, DVC, and source CSV seed for runtime init. | Local experimentation and DVC. |
| `models/` | Metadata or generated outputs | Local Python, notebooks, and DVC model outputs. | Local experimentation and DVC. |
| `logs/` | No | Local non-Compose logs. | Local experimentation. |
| `.env` | No | Runtime environment values. | Local operator. |

## Source code and runtime integration

`src/` is the right place for reusable Python logic that should be importable and
testable outside a specific Docker Compose service. This includes FastAPI code,
ML ingestion, feature engineering, model training, prediction, metrics, the
runner API, and shared pipeline helpers.

The runner API lives under `src/job_runner/` because it is reusable application
logic with API tests. It uses framework-neutral contracts instead of redefining
job request and status schemas in runtime-specific folders.

Airflow DAGs under `docker/dev/airflow/dags/` and `docker/prod/airflow/dags/`
are runtime integration assets. They are coupled to runtime connection IDs,
service DNS names, mounted paths, Airflow configuration, and operational tags.
They should not be moved into `src/` unless a separate packaging decision is
made.

The current dev and prod-like DAGs are intentionally aligned: both submit typed
ML jobs to `job-runner-api`, then refresh the prediction API from promoted
manifests. Differences should remain explicit and runtime-scoped.

## Docker runtime layout

### `docker/dev`

`docker/dev` is the canonical local development Compose runtime. It optimizes for
host visibility, debugging, demos, and fast iteration while using the same
functional runner/gateway/ML-service execution path as production-like runtime.

Development runtime outputs are host bind-mounted under:

```text
docker/dev/runtime/artifacts
docker/dev/runtime/data
docker/dev/runtime/logs
docker/dev/runtime/models
```

`docker/dev/Makefile` prepares these host directories and seeds the raw CSV from
root `data/raw` into `docker/dev/runtime/data/raw` before Compose configuration
validation.

Root `data`, `models`, and `logs` remain available for local Python commands,
DVC reproduction, notebooks, and explicit developer scripts. They should not be
used as the default write target for Airflow-driven development operations.

### `docker/prod`

`docker/prod` is the local production-like Compose runtime. It optimizes for
least privilege, reduced host mounts, stable service discovery, explicit
workspace ownership, and narrower runtime boundaries.

Production-like runtime outputs are stored in the named Docker volume:

```text
prod-runtime
```

The `init-volumes` service creates the expected runtime subdirectories, seeds the
raw CSV from root `data/raw`, applies ownership, and fixes permissions before
runtime services start. A host `docker/prod/runtime` directory is no longer the
normal runtime workspace.

Inspect the production-like runtime volume with:

```bash
make prod-dir-runtime
```

Consumers in `docker/prod` should use promoted manifests rather than scanning
runtime folders for the newest files.

## Documentation areas

The documentation level rules are defined in [`../README.md`](../README.md).

| Area | Responsibility |
| ---- | -------------- |
| `docs/current-runtime-and-operations/` | Runtime operation, port inventory, dependency policy, and repository ownership. |
| `docs/architecture-references/` | Communication, security boundaries, and implemented network topology. |
| `docs/remaining-work/` | Future improvement axes and not-yet-implemented design targets. |
| `references/` | Static diagrams, exports, and explanatory assets. |

## Data and DVC expectations

DVC describes reproducible data and model outputs while keeping large or
generated payloads out of normal Git history.

Current `dvc.yaml` ownership:

1. `import_raw_data` reads raw input and writes `data/interim/<scenario>`.
2. `build_features` reads interim data and writes `data/processed/<scenario>`.
3. `train_and_predict` reads processed data and writes `models/<scenario>` and
   `data/final/<scenario>`.

Directory expectations:

- `data/raw/` contains input data restored locally or tracked through DVC metadata when needed.
- `data/interim/`, `data/processed/`, and `data/final/` are local experimentation and DVC outputs restored or reproduced outside Compose runtime ownership.
- `models/` is generated by local experimentation, DVC, and training workflows outside Compose runtime ownership.
- `logs/` is generated by local developer commands and non-Compose experiments.
- `docker/dev/runtime/data`, `docker/dev/runtime/models`, `docker/dev/runtime/logs`, and `docker/dev/runtime/artifacts` are Airflow-driven development runtime outputs and are not DVC-managed.
- `prod-runtime:/data`, `prod-runtime:/models`, `prod-runtime:/logs`, and `prod-runtime:/artifacts` are production-like runtime outputs and are not DVC-managed.
- `.dvc/config` is versioned; `.dvc/config.local`, `.dvc/cache/`, and `.dvc/tmp/` are local workstation state.

The development and production-like Compose runtimes intentionally use their own
ignored runtime storage. Root `data`, `models`, and `logs` stay reserved for
DVC/local experimentation, comparisons, and explicit developer commands.

## Scripts and helper ownership

A top-level `scripts/` folder, if introduced, should contain developer or
operational helpers only. Runtime-specific helpers should stay near the runtime
that executes them, for example under `docker/dev/airflow/scripts/` or
`docker/prod/airflow/scripts/`.

Reusable business logic should live in `src/` with tests rather than in shell
scripts or deployment folders.

## Ignore-rule review

The current ignore strategy is consistent with this structure:

- root `.gitignore` ignores local Python caches, virtual environments, local env
  files, `logs`, `mlruns`, and `mlflow.db`;
- `.dvc/.gitignore` ignores local DVC config, cache, and temporary state;
- `data/*/.gitignore` and `models/.gitignore` keep generated scenario outputs
  out of normal Git commits;
- `docker/dev/runtime/.gitignore` keeps development Compose runtime outputs out
  of Git and DVC ownership.

There is no root `.dockerignore` at the time of this review. Adding one requires
validation because current image build contexts rely on repository root
visibility.
