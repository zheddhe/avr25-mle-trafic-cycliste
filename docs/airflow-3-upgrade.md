# Airflow 3 upgrade strategy

This document tracks the Airflow part of story #51.

The goal is to move the local orchestration runtime from the previous Airflow 2
stack to an Airflow 3 baseline without requiring backward compatibility with the
existing local Airflow metadata database.

## Scope

Target baseline:

- Airflow: `3.2.2`
- Python runtime: managed by the official Airflow image
- Executor: CeleryExecutor
- Broker: Redis
- Metadata database: Postgres

This document focuses only on Airflow. MLflow, Prometheus, and Grafana are
handled in later increments of the same story.

## Why this change is needed

The previous Compose stack used `apache/airflow:2.8.1`. The runtime observed
inside the container was Python 3.8. That is no longer aligned with the project
code style and Ruff target, because DAG helper modules now use modern Python type
annotations such as `dict[str, ...]`, `list[...]`, and `tuple[...]`.

The current business microservices already use Python 3.12 images. The Airflow
upgrade should therefore modernize the orchestration runtime instead of forcing
DAG code back to Python 3.8-compatible annotations.

## Migration approach

No production Airflow deployment or metadata database has to be preserved.
Therefore the migration can be treated as a clean local runtime replacement:

1. Stop the previous Airflow services.
2. Remove the previous local Airflow Postgres volume.
3. Start the Airflow 3 Compose workspace.
4. Validate DAG import and representative task execution.
5. Fold the validated Airflow 3 model back into the main Compose file.

The temporary migration Compose file is:

```bash
docker-compose.airflow3.yaml
```

It deliberately introduces the Airflow 3 service split before replacing the main
runtime:

- `airflow-api-server`
- `airflow-scheduler`
- `airflow-dag-processor`
- `airflow-worker`
- `airflow-triggerer`
- `airflow-init`
- `airflow-postgres`
- `airflow-redis`
- `airflow-flower`

## Current validation commands

Render the Airflow 3 Compose model:

```bash
docker compose -f docker-compose.airflow3.yaml --profile airflow3 config
```

Reset the local Airflow metadata database when needed:

```bash
docker compose -f docker-compose.airflow3.yaml --profile airflow3 down --volumes
```

Start the Airflow 3 stack:

```bash
docker compose -f docker-compose.airflow3.yaml --profile airflow3 up -d
```

Include Flower if needed:

```bash
docker compose -f docker-compose.airflow3.yaml --profile flower up -d
```

Follow scheduler logs:

```bash
docker compose -f docker-compose.airflow3.yaml logs -f airflow-scheduler
```

Follow DAG processor logs:

```bash
docker compose -f docker-compose.airflow3.yaml logs -f airflow-dag-processor
```

## Smoke test checklist

- [ ] `docker compose -f docker-compose.airflow3.yaml --profile airflow3 config`
      succeeds.
- [ ] Airflow metadata database initializes from a clean volume.
- [ ] Airflow API server is reachable on `AIRFLOW_HOST_PORT`.
- [ ] Airflow UI/authentication works with the configured local user.
- [ ] `airflow-dag-processor` imports all project DAG files.
- [ ] No DAG import error remains for `common/utils.py`.
- [ ] `bike_traffic_pipeline_dag.py` appears in the Airflow UI.
- [ ] `bike_traffic_orchestrator_dag.py` appears in the Airflow UI.
- [ ] Airflow variables are imported from `variables.json`.
- [ ] Airflow connections are imported from `connections.json`.
- [ ] The `sequential_counters` pool exists.
- [ ] The worker can access the Docker socket for DockerOperator tasks.
- [ ] A representative DAG run can trigger business ML containers.
- [ ] Logs are written to the host `./logs` mount.

## Known follow-up decisions

The Airflow 3 Compose workspace is an intermediate artifact. Once validated, the
main `docker-compose.yaml` should be updated so the project has only one primary
runtime definition.

The Docker socket remains a local development privilege. It should be revisited
in the later runtime identity and permissions story.

The Airflow worker entrypoint still uses a root bootstrap to align Docker socket
access. This is acceptable for the current local migration but not a production
security model.
