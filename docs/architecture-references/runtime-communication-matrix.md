# Runtime communication matrix

This document describes the implemented service-to-service communication model for
`docker/dev` and `docker/prod`.

Both runtimes now use the same functional pipeline path:

```text
Airflow DAG tasks
  -> job-runner-api
  -> ml-gateway
  -> ML step service replicas
  -> promoted artifact manifests
  -> API refresh
```

Runtime-specific differences are limited to service suffixes, network names,
host exposure, workspace mounts, image tags, and runtime ownership.

## Runtime network domains

Development uses prefixed networks to keep it runnable in parallel with
production-like validation:

| Dev network | Responsibility |
| ----------- | -------------- |
| `dev_orchestration_net` | Airflow API server, scheduler, DAG processor, triggerer, worker, metadata database, and Redis broker. |
| `dev_pipeline_runtime_net` | API, Airflow worker, runner API, gateway, and ML step service control flow. |
| `dev_tracking_client_net` | Model workloads and MLflow tracking API. |
| `dev_tracking_backend_net` | MLflow server, PostgreSQL backend, MinIO artifact backend, and MinIO initialization. |
| `dev_observability_net` | Metrics push, metrics scrape, dashboards, cAdvisor, and alerting. |
| `dev_support_net` | Local development support services such as MailHog. |

Production-like uses the same functional split with `prod_` prefixes:

| Prod-like network | Responsibility |
| ----------------- | -------------- |
| `prod_orchestration_net` | Airflow API server, scheduler, DAG processor, triggerer, worker, metadata database, and Redis broker. |
| `prod_pipeline_runtime_net` | API, Airflow worker, runner API, gateway, and ML step service control flow. |
| `prod_tracking_client_net` | Model workloads and MLflow tracking API. |
| `prod_tracking_backend_net` | MLflow server, PostgreSQL backend, MinIO artifact backend, and MinIO initialization. |
| `prod_observability_net` | Metrics push, metrics scrape, dashboards, cAdvisor, and alerting. |
| `prod_support_net` | Local support services that are not part of the pipeline core. |

The previous broad local `mlops_net` model is no longer the implemented runtime
contract for `docker/dev`.

## Cross-runtime communication matrix

| Source | Target | Dev DNS | Prod-like DNS | Port | Runtime network | Reason |
| ------ | ------ | ------- | ------------- | ---- | --------------- | ------ |
| Host browser/client | Prediction API | `localhost:${API_HOST_PORT_DEV}` | `localhost:${API_HOST_PORT_PROD}` | `10000` inside container | Host publication | Local API usage and OpenAPI docs. |
| Host browser/client | Airflow API/UI | `localhost:${AIRFLOW_HOST_PORT_DEV}` | `localhost:${AIRFLOW_HOST_PORT_PROD}` | `8080` inside container | Host publication | DAG operations and inspection. |
| Host browser/client | MLflow UI | `localhost:${MLFLOW_HOST_PORT_DEV}` | `localhost:${MLFLOW_HOST_PORT_PROD}` | `5000` inside container | Host publication | ML tracking inspection. |
| Host browser/client | Grafana UI | `localhost:${GRAFANA_HOST_PORT_DEV}` | `localhost:${GRAFANA_HOST_PORT_PROD}` | `3000` inside container | Host publication | Dashboard inspection. |
| Host browser/client | Dev-only support UIs | Dev-only host ports | Internal-only | service-specific | Host publication in dev only | Debugging MinIO, Prometheus, Pushgateway, cAdvisor, Alertmanager, and MailHog. |
| Airflow services | Airflow PostgreSQL | `airflow-postgres` | `airflow-postgres` | `5432` | orchestration | Metadata DB and result backend. |
| Airflow services | Airflow Redis | `airflow-redis` | `airflow-redis` | `6379` | orchestration | Celery broker. |
| Airflow services | Airflow API server | `airflow-api-server` | `airflow-api-server` | `8080` | orchestration | Internal Airflow execution API. |
| Airflow DAG tasks | Runner API | `job-runner-api` | `job-runner-api` | `10080` | pipeline runtime | Typed ML job submission and status reads. |
| Airflow DAG tasks | Prediction API | `api-dev` | `api-prod` | `10000` | pipeline runtime | Authenticated prediction refresh after successful DAG runs. |
| Runner API | ML gateway | `ml-gateway` | `ml-gateway` | `10090` | pipeline runtime | Route typed job requests to ML step service replicas. |
| ML gateway | Ingestion service | `ml-ingest-dev` | `ml-ingest-prod` | `10081` | pipeline runtime | Execute ingestion jobs. |
| ML gateway | Feature service | `ml-features-dev` | `ml-features-prod` | `10082` | pipeline runtime | Execute feature jobs. |
| ML gateway | Model service | `ml-models-dev` | `ml-models-prod` | `10083` | pipeline runtime | Execute training and prediction jobs. |
| ML step services | Pushgateway | `monitoring-pushgateway` | `monitoring-pushgateway` | `9091` | observability | Push batch job metrics. |
| Prometheus | Prediction API | `api-dev` | `api-prod` | `10000` | observability | Scrape FastAPI metrics. |
| Prometheus | Pushgateway | `monitoring-pushgateway` | `monitoring-pushgateway` | `9091` | observability | Scrape batch job metrics. |
| Prometheus | cAdvisor | `monitoring-cadvisor` | `monitoring-cadvisor` | `8080` | observability | Scrape container metrics. |
| Grafana | Prometheus | `monitoring-prometheus` | `monitoring-prometheus` | `9090` | observability | Provisioned datasource. |
| Alertmanager | MailHog | `monitoring-mailhog` | `monitoring-mailhog` | `1025` | support | Local alert email capture when enabled. |
| Model service | MLflow server | `mlflow-server` | `mlflow-server` | `5000` | tracking client | Log runs, metrics, params, model metadata, and model registry evidence. |
| Model service | MLflow MinIO | `mlflow-minio` | `mlflow-minio` | `9000` | tracking backend | Artifact backend reachability when MLflow artifact locations are resolved by clients. |
| MLflow server | MLflow PostgreSQL | `mlflow-postgres` | `mlflow-postgres` | `5432` | tracking backend | MLflow backend store. |
| MLflow server | MLflow MinIO | `mlflow-minio` | `mlflow-minio` | `9000` | tracking backend | MLflow artifact store. |
| MinIO init | MLflow MinIO | `mlflow-minio` | `mlflow-minio` | `9000` | tracking backend | Bootstrap the MLflow bucket. |

## Runner execution boundary

`job-runner-api` is the stable execution boundary for Airflow ML work in both
runtimes.

It is responsible for:

- accepting typed job submissions;
- mapping each job type to one service endpoint;
- limiting in-flight service dispatch;
- exposing health and status endpoints;
- writing runner logs under the runtime log root.

It is not responsible for:

- Airflow scheduling or DAG dependency ordering;
- arbitrary shell execution;
- Docker socket orchestration;
- direct filesystem discovery of latest outputs;
- bypassing `ml-gateway` to reach scaled ML service replicas.

## Gateway routing

`ml-gateway` is an internal Nginx service in both runtimes. It listens on
`10090` and routes stable paths to service-specific `/jobs` endpoints:

| Gateway path | Dev backend | Prod-like backend |
| ------------ | ----------- | ----------------- |
| `/ingest/jobs` | `ml-ingest-dev:10081/jobs` | `ml-ingest-prod:10081/jobs` |
| `/features/jobs` | `ml-features-dev:10082/jobs` | `ml-features-prod:10082/jobs` |
| `/models/jobs` | `ml-models-dev:10083/jobs` | `ml-models-prod:10083/jobs` |
| `/health` | Gateway health response | Gateway health response |

This keeps runner configuration stable when ML service replicas are scaled by
`make dev-scale-ml` or `make prod-scale-ml`.

## Runtime mount coupling

Development uses host bind mounts:

| Dev mount | Consumers | Purpose |
| --------- | --------- | ------- |
| `./runtime/data:/app/data` | ML step services | Read/write runtime data. |
| `./runtime/models:/app/models` | Model service | Write runtime model artifacts. |
| `./runtime/logs:/app/logs` | API, ML services, runner | Host-visible runtime logs. |
| `./runtime/artifacts:/app/artifacts` | ML services, API, Airflow | Manifest-first artifact handoff. |

Production-like uses the named Docker volume `prod-runtime`, initialized by
`init-volumes`:

| Prod-like volume subpath | Consumers | Purpose |
| ------------------------ | --------- | ------- |
| `data` | ML step services and API final-data reads | Runtime data handoff. |
| `models` | Model service | Runtime model artifacts. |
| `logs` | API, Airflow, ML services, runner | Runtime logs. |
| `artifacts` | ML services, API, Airflow | Manifest-first artifact handoff. |

`init-volumes` also seeds the production-like raw CSV from root `data/raw` into
`prod-runtime:/data/raw` before dependent services start.

## Manifest handoff path

The current artifact handoff is manifest-first:

```text
ML step service
  -> writes data/model payloads
  -> writes artifact manifest under /app/artifacts/manifests
  -> promotes current.json for prediction outputs
  -> API refresh reads promoted manifests
```

API serving resolves prediction payloads from promoted manifests and
`ARTIFACT_REPOSITORY_ROOT`; it does not select files by scanning runtime folders.

## Validation

Expected checks:

```bash
make dev-compose-config
make prod-compose-config
make dev-start DEV_PROFILE=ptf
make prod-start PROD_PROFILE=ptf
make dev-scale-ml ML_INGEST_REPLICAS=2 ML_FEATURES_REPLICAS=2 ML_MODELS_REPLICAS=2
make prod-scale-ml ML_INGEST_REPLICAS=2 ML_FEATURES_REPLICAS=2 ML_MODELS_REPLICAS=2
```

The rendered Compose configuration should show:

- Airflow workers connected to orchestration, pipeline runtime, and support
  networks;
- runner APIs connected only to the pipeline runtime network;
- gateways connected only to the pipeline runtime network;
- ML services connected to pipeline runtime and observability networks;
- model services additionally connected to MLflow tracking networks;
- no Docker socket mount on Airflow workers;
- cAdvisor as the only Docker socket observability exception.
