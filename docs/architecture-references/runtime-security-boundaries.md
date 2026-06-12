# Runtime security boundaries

This document describes current local runtime boundaries for `docker/dev` and
`docker/prod`. It is not the final security-hardening contract; that work remains
separate. The goal here is to document the implemented boundaries after the dev
and prod-like topology alignment.

## Current boundary summary

| Boundary | Development runtime | Local production-like runtime |
| -------- | ------------------- | ----------------------------- |
| Normal ML execution | Airflow DAGs submit typed jobs to `job-runner-api`; the runner dispatches through `ml-gateway`. | Same functional path as development. |
| Docker socket | Airflow does not mount `/var/run/docker.sock` for normal ML execution. | Airflow does not mount `/var/run/docker.sock` for normal ML execution. |
| Docker socket exception | cAdvisor mounts the Docker socket read-only for container metrics. | cAdvisor mounts the Docker socket read-only for container metrics. |
| Runtime workspace | Host bind-mounted `docker/dev/runtime`. | Named Docker volume `prod-runtime`, initialized by `init-volumes`. |
| Host exposure | Broad and intentionally visible for local debugging. | Reduced to API, Airflow, MLflow, and Grafana. |
| Service dispatch | Runner reaches only `ml-gateway`; gateway reaches ML step services. | Same functional path as development. |
| MLflow backend | Internal PostgreSQL and MinIO services; MinIO console exposed only in dev. | Internal PostgreSQL and MinIO services; MinIO is not host-exposed. |
| Runtime users | Configured through `AIRFLOW_UID`, `AIRFLOW_GID`, `APP_UID`, and `APP_GID`. | Same user variables, with production-like security options on business services. |

## Docker socket boundary

The normal Airflow pipeline execution path no longer depends on DockerOperator or
on `/var/run/docker.sock`.

Expected normal flow:

```text
Airflow task
  -> HTTP request to job-runner-api
  -> HTTP request to ml-gateway
  -> HTTP request to one ML step service
```

The Docker socket must not be reintroduced into Airflow scheduler, worker,
triggerer, DAG processor, API server, runner, API, gateway, or ML step services
for normal pipeline execution.

The current exception is cAdvisor:

```text
/var/run/docker.sock:/var/run/docker.sock:ro
```

That exception exists only for container observability. It should not be reused
as an execution boundary.

## Runner API boundary

`job-runner-api` is intentionally internal-only in both runtimes. It must stay
inside the pipeline runtime network and should not publish a host port.

Allowed responsibilities:

- accept typed `ingest`, `features`, and `models` job submissions;
- dispatch each job to a configured service URL;
- enforce in-flight job limits;
- expose health and job status for Airflow polling;
- write runtime runner logs.

Disallowed responsibilities:

- arbitrary command execution;
- Docker socket access;
- host filesystem control;
- direct access to Airflow metadata storage;
- direct access to MLflow backend storage unless a future story explicitly adds
  such a need.

## Gateway boundary

`ml-gateway` is internal-only and connected only to the pipeline runtime network.
It provides stable routing between `job-runner-api` and scaled ML step services.

The runner should target gateway paths, not individual service replicas:

```text
http://ml-gateway:10090/ingest
http://ml-gateway:10090/features
http://ml-gateway:10090/models
```

The gateway should not expose a host port. It should not join tracking backend,
observability, support, or Airflow orchestration networks.

## Runtime workspace boundary

Development prioritizes inspectability:

```text
docker/dev/runtime/data
docker/dev/runtime/models
docker/dev/runtime/logs
docker/dev/runtime/artifacts
```

These paths are bind-mounted from the host and intentionally easy to inspect.
They are local-only and ignored by Git.

Production-like prioritizes runtime ownership and reduced host coupling:

```text
prod-runtime:/data
prod-runtime:/models
prod-runtime:/logs
prod-runtime:/artifacts
```

The `init-volumes` service owns first-time directory creation, raw CSV seeding,
ownership, and permissions for the `prod-runtime` volume. Other services depend
on it where they need runtime paths.

Root `data`, `models`, and `logs` remain local/DVC experimentation workspaces and
must not become implicit Compose runtime write targets.

## Host exposure boundary

Development host exposure is intentionally broad and visible. It supports
OpenAPI inspection, Airflow debugging, MLflow inspection, MinIO console access,
metrics debugging, local alert routing, and demo workflows.

Production-like host exposure is reduced to operator-facing services:

- FastAPI prediction API;
- Airflow UI/API;
- MLflow UI/API;
- Grafana UI.

Internal-only in production-like runtime:

- `job-runner-api`;
- `ml-gateway`;
- all ML step services;
- MinIO API and console;
- Prometheus;
- Pushgateway;
- cAdvisor;
- Alertmanager;
- MailHog;
- Airflow metadata services;
- MLflow backend services.

## Network boundary

Both runtimes use functional network domains with runtime-specific prefixes.
The production-like runtime uses:

- `prod_orchestration_net`;
- `prod_pipeline_runtime_net`;
- `prod_tracking_client_net`;
- `prod_tracking_backend_net`;
- `prod_observability_net`;
- `prod_support_net`.

The development runtime mirrors the same functional split with `dev_` prefixes.
The previous broad local `mlops_net` is no longer the documented current runtime
contract.

Expected service placement rules:

- Airflow metadata services stay in orchestration networks.
- Airflow workers bridge orchestration and pipeline runtime networks because DAG
  tasks call runtime services.
- Runner APIs and gateways stay in pipeline runtime networks only.
- ML step services join pipeline runtime and observability networks.
- Model services additionally join MLflow tracking networks.
- MLflow backend databases and object stores stay in tracking backend networks.
- Monitoring services stay in observability networks unless a support path is
  explicitly required.

## Environment and secrets boundary

`.env.template` contains safe defaults and placeholders only. `.env` contains
local runtime values and must stay untracked.

Current runtime variables separate:

- dev/prod host port ranges;
- runtime user IDs and group IDs;
- image tags;
- ML scale-out settings;
- MLflow local, Compose, and DagsHub views;
- Airflow keys and metadata database credentials;
- monitoring credentials.

Secrets hardening, credential rotation, stronger authentication, and production
configuration management remain future security-hardening work.

## Validation expectations

Configuration checks:

```bash
make dev-compose-config
make prod-compose-config
```

Runtime checks:

```bash
make dev-start DEV_PROFILE=ptf
make prod-start PROD_PROFILE=ptf
make dev-ps
make prod-ps
```

Expected security-boundary properties:

- no Airflow service mounts `/var/run/docker.sock`;
- no runner, gateway, API, or ML service mounts `/var/run/docker.sock`;
- only cAdvisor has the Docker socket observability exception;
- `job-runner-api`, `ml-gateway`, and ML step services are not host-published;
- production-like MinIO, Prometheus, Pushgateway, cAdvisor, Alertmanager, and
  MailHog are not host-published;
- production-like runtime payloads live in `prod-runtime`, not host runtime
  folders;
- development runtime payloads live in host-visible `docker/dev/runtime`.
