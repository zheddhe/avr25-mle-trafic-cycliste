# Runtime logging and traceability

This document describes the implemented runtime logging contract for local
`docker/dev`, local `docker/prod`, and direct local commands.

The goal is operational traceability without coupling services to the Docker API.
Services identify themselves from process environment values and write project
logs to deterministic files under the runtime log root.

## Runtime log roots

| Context | Runtime log root | Notes |
| ------- | ---------------- | ----- |
| Direct local Python commands | `logs/` | Console output is preferred for ad-hoc CLI runs. |
| Development Compose runtime | `docker/dev/runtime/logs` | Host bind mount mapped to `/app/logs` in services. |
| Production-like Compose runtime | `prod-runtime:/logs` | Named volume mounted as `/app/logs` in services. |

Airflow keeps its own task logs under the runtime Airflow log tree. Application
services use the project logger tree and do not rely on `uvicorn` log files for
business evidence.

## Service log files

Runtime service logs use one file per service instance. `run_id`, `trace_id`,
`job_id`, `job_type`, and `counter_id` are written inside log records rather than
in the file name. This preserves chronological readability when the same process
handles several jobs sequentially.

| Service family | File pattern | Default identity |
| -------------- | ------------ | ---------------- |
| Prediction API | `logs/api/<service_name>_<hostname>.log` | `api_local.log` |
| Job runner API | `logs/job-runner/<service_name>_<hostname>.log` | `job-runner-api_local.log` |
| ML ingest | `logs/ml/ingest/<service_name>_<hostname>.log` | `ml-ingest_local.log` |
| ML features | `logs/ml/features/<service_name>_<hostname>.log` | `ml-features_local.log` |
| ML models | `logs/ml/models/<service_name>_<hostname>.log` | `ml-models_local.log` |

`service_name` comes from `SERVICE_NAME` when configured. `hostname` comes from
`HOSTNAME` when available. Missing values fall back to the defaults above.

The project intentionally does not inspect the Docker daemon to recover container
names. That would require Docker socket access from runtime services and would
weaken the current isolation boundary.

## End-to-end job traceability

Airflow, the runner, and ML services share the following identifiers:

| Field | Origin | Purpose |
| ----- | ------ | ------- |
| `run_id` | Airflow DAG preparation | Logical pipeline run across steps. |
| `trace_id` | Currently equal to `run_id` | Stable cross-service trace field. |
| `job_id` | `job-runner-api` | Runner job identity forwarded to ML services. |
| `job_type` | Typed request contract | One of `ingest`, `features`, or `models`. |
| `counter_id` | DAG configuration | Counter processed by the job. |
| `metrics_reference` | ML service result | Log file path returned in `JobResult.metrics`. |

The expected correlation path is:

```text
Airflow task log
  -> job-runner-api service log
  -> ML service instance log
  -> JobResult.metrics.metrics_reference
```

Runner-dispatched ML jobs must log at least job start, failure, and success with
these fields. The job runner API must log the same `job_id`, `run_id`,
`job_type`, and `counter_id` so that operators can join the runner decision with
the ML service execution evidence.

## Direct CLI behavior

Direct host-side CLI launches keep console-only project logging by default. This
avoids creating local files for one-off development commands and keeps container
logs useful when a CLI is executed inside an ephemeral shell.

Runner-dispatched jobs are the path that creates ML service log files.

## Logging conventions

Project modules should use:

```python
LOGGER = get_logger(__name__)
```

Service and CLI entrypoints configure logging once with `configure_logging(...)`.
Logger calls should use lazy interpolation:

```python
LOGGER.info("Loaded %s rows", row_count)
```

Use f-strings for regular non-logging strings.

## Validation hints

Useful static checks for this logging contract are:

```bash
grep -RInE 'basicConfig|getLogger|print\(' src/
grep -RInE 'os\.getenv|os\.environ|environ\[|load_dotenv' src/
```

Remaining matches should stay restricted to centralized helpers or explicitly
justified framework integration points.
