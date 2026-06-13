# Repository structure and runtime ownership

This document clarifies how contributors should read and modify the repository
now that the project has symmetric local Compose runtime entrypoints for
development and local production-like validation.

## Guiding rules

- Use `src/` for reusable application, API, ML, metrics, and pipeline logic.
- Keep deployment-specific Airflow DAGs close to their runtime assets.
- Treat root `data`, `logs`, and `models` as local experimentation and DVC
  workspaces.
- Treat `docker/dev/runtime` as the ignored Airflow-driven development runtime
  workspace.
- Treat `docker/prod/runtime` as the ignored local production-like runtime
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
| `Makefile` | Repository setup, validation, DVC, and runtime Makefile inclusion. |
| `.env.template` | Versioned template for local runtime variables. |
| `.env` | Local-only runtime values created from `.env.template`. |
| `pyproject.toml` | Local Python dependency groups and tooling configuration. |
| `uv.lock` | Versioned uv lockfile for reproducible local validation. |
| `src/` | Reusable FastAPI, ML, feature, model, metrics, runner, and pipeline code. |
| `tests/` | Unit, integration, and regression tests. |
| `docker/` | Dockerfiles, runtime config, Airflow DAG wiring, and service helpers. |
| `docs/` | Current runtime docs, architecture references, and active design docs. |
| `data/` | Local experimentation and DVC data workspace. |
| `models/` | Local experimentation and DVC model artifact workspace. |
| `logs/` | Local experimentation and non-Compose developer logs. |
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
| `docker/dev/` | Yes | No | Partly | No | No | No |
| `docker/dev/runtime/` | No | Yes | Yes | No | Yes | No |
| `docker/prod/` | Yes | No | Partly | No | No | No |
| `docker/prod/runtime/` | No | Yes | Yes | No | Yes | No |
| `docker/prod/runtime/artifacts/` | No | Yes | Yes | No | Yes | No |
| `docs/` | Yes | No | No | No | No | No |
| `data/raw/` | Metadata or placeholders | Restored locally | Selected read-only inputs | Yes | Partly | No |
| `data/interim/` | Metadata only | Yes | Local/DVC only | Yes | Partly | No |
| `data/processed/` | Metadata only | Yes | Local/DVC only | Yes | Partly | No |
| `data/final/` | Metadata only | Yes | Local/DVC only | Yes | Partly | No |
| `models/` | Metadata only | Yes | Local/DVC only | Yes | Partly | No |
| `logs/` | No | Yes | Local/DVC only | No | Yes | No |
| `references/` | Yes | No | No | No | No | Yes |
| `.dvc/config` | Yes | No | No | No | No | No |
| `.dvc/config.local` | No | No | No | No | Yes | No |
| `.dvc/cache/` | No | Yes | No | Local cache | Yes | No |

`Partly` means a path can contain tracked placeholders or runtime configuration
while large data payloads and runtime outputs remain ignored, restored, or
reproduced.

## Source code and runtime integration

`src/` is the right place for reusable Python logic that should be importable and
testable outside a specific Docker Compose service. This includes FastAPI code,
ML ingestion, feature engineering, model training, prediction, metrics, the
runner API, and shared pipeline helpers.

The runner API lives under `src/job_runner/` because it is reusable application
logic with API tests. It uses the framework-neutral pipeline contracts from
`src/pipeline/contracts/` instead of redefining job request and status schemas.

Airflow DAGs under `docker/dev/airflow/dags/` are runtime integration assets.
They are coupled to local Airflow variables, connections, DockerOperator
settings, service names, mounted paths, and the current worker model. They should
not be moved into `src/` unless a separate packaging decision is made.

DAG placement may differ between `docker/dev` and `docker/prod` when operators,
worker pools, queue names, image names, or artifact handoff mechanisms diverge.

## Docker runtime layout

### `docker/dev`

`docker/dev` is the canonical local development Compose runtime. It optimizes for
host visibility, debugging, demos, and fast iteration while keeping
Airflow-driven operational outputs under `docker/dev/runtime`.

The development Airflow/DockerOperator path mounts `docker/dev/runtime/data`,
`docker/dev/runtime/models`, and `docker/dev/runtime/logs` into ML task
containers. This keeps ops-style Compose runs separate from root DVC and notebook
experimentation outputs.

Root `data`, `models`, and `logs` remain available for local Python commands,
DVC reproduction, notebooks, and explicit developer scripts. They should not be
used as the default write target for Airflow-driven development operations.

### `docker/prod`

`docker/prod` is the local production-like Compose runtime. It optimizes for
least privilege, reduced host mounts, stable service discovery, explicit
workspace ownership, and narrower runtime boundaries.

It writes generated operational data under `docker/prod/runtime`, which is
ignored by Git and not DVC-managed. The only current root data dependency is the
required source CSV mounted read-only from `data/raw` into the ingestion service.

Runtime artifact promotion is defined by
[`../next-phase-design/artifact-handoff-strategy.md`](../next-phase-design/artifact-handoff-strategy.md).
Consumers in `docker/prod` should use promoted manifests rather than scanning
runtime folders for the newest files.

## Documentation areas

The documentation level rules are defined in [`../README.md`](../README.md).

| Area | Responsibility |
| ---- | -------------- |
| `docs/current-runtime-and-operations/` | Runtime operation, port inventory, dependency policy, and repository ownership. |
| `docs/architecture-references/` | Communication, security, and implemented network topology. |
| `docs/next-phase-design/` | Active design notes and open implementation coordination. |
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
- `data/interim/`, `data/processed/`, and `data/final/` are local
  experimentation and DVC outputs restored or reproduced outside Compose runtime
  ownership.
- `models/` is generated by local experimentation, DVC, and training workflows
  outside Compose runtime ownership.
- `logs/` is generated by local developer commands and non-Compose experiments.
- `docker/dev/runtime/data`, `docker/dev/runtime/models`, and
  `docker/dev/runtime/logs` are Airflow-driven development runtime outputs and
  are not DVC-managed.
- `docker/prod/runtime/data`, `docker/prod/runtime/models`, and
  `docker/prod/runtime/logs` are production-like runtime outputs and are not
  DVC-managed.
- `docker/prod/runtime/artifacts` is the expected local root for promoted
  manifest files and artifact payloads in the first manifest-first runtime.
- `.dvc/config` is versioned; `.dvc/config.local`, `.dvc/cache/`, and
  `.dvc/tmp/` are local-only.

The development and production-like Compose runtimes intentionally use their own
ignored runtime folders. Root `data`, `models`, and `logs` stay reserved for
DVC/local experimentation, comparisons, and explicit developer commands.

## Scripts and helper ownership

A top-level `scripts/` folder, if introduced, should contain developer or
operational helpers only. Runtime-specific helpers should stay near the runtime
that executes them, for example under `docker/dev/airflow/scripts/` or
`docker/prod/airflow/scripts/`.

Reusable business logic should live in `src/` with tests rather than in shell
scripts or deployment folders.

## Logging and runtime environment conventions

Business code uses shared helpers from `src/common/`:

- service and CLI entrypoints call `configure_logging(...)` once;
- internal modules obtain loggers with `get_logger(__name__)`;
- runtime configuration is read through `src.common.env`;
- production-like containers rely on process environment variables only;
- local `.env` loading is limited to test or developer harnesses before code
  starts.

Mandatory runtime variables must fail fast with an explicit configuration error.
Optional variables must use an explicit default or return `None`.

Runtime service logs are written below repository-local `logs/` paths:

- the serving API writes project logs to `logs/api/main.log`;
- the job runner API writes project logs to `logs/job-runner/main.log`;
- typed ML service jobs write to
  `logs/ml/<step>/<service_instance_id>_<job_id>.log`, where `<step>` is
  `ingest`, `features`, or `models`. The service instance identifier
  comes from `ML_SERVICE_INSTANCE_ID`, `SERVICE_INSTANCE_ID`, `HOSTNAME`,
  or a process-local fallback.

Direct CLI launches keep console-only project logging. This keeps ad-hoc unit
runs visible in the terminal or container logs without creating local files.
The job runner API logs `job_id`, `run_id`, `job_type`, and `counter_id`
so operators can correlate `logs/job-runner/main.log` with ML job files.

Log levels should remain operational: `DEBUG` for diagnostics, `INFO` for
lifecycle milestones, `WARNING` for recoverable abnormal states, and `ERROR`
for failed operations.

Logger calls should use lazy interpolation by passing values as logger
arguments, for example `LOGGER.info("Loaded %s rows", row_count)`. Regular
non-logging strings should use f-strings when interpolation improves
readability.

## Local-only files

- `.env.template` is versioned and contains placeholders or safe defaults.
- `.env`, `.env.local`, `.env.dagshub`, and `.dvc/config.local` are local-only.
- `docker/dev/runtime/` and `docker/prod/runtime/` are local-only and ignored
  except for their `.gitignore` placeholders.
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
- `docker/dev/runtime/.gitignore` and `docker/prod/runtime/.gitignore` keep
  Compose runtime outputs out of Git and DVC ownership.

There is no root `.dockerignore` at the time of this review. Adding one requires
validation because current image build contexts rely on repository root
visibility.
