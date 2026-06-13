# Global remaining work

This document lists improvement axes that remain outside the validated local
runtime baseline.

The current baseline is documented in:

- [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md);
- [`../current-runtime-and-operations/runtime-logging.md`](../current-runtime-and-operations/runtime-logging.md);
- [`../architecture-references/runtime-communication-matrix.md`](../architecture-references/runtime-communication-matrix.md);
- [`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md).

## Current validated baseline

The local runtime validates this functional path:

```text
Airflow DAG task
  -> job-runner-api
  -> ml-gateway
  -> ML step service replica
  -> promoted prediction manifest
  -> authenticated API refresh
  -> FastAPI serving from the promoted manifest payload
  -> Prometheus and Grafana observability
```

Acceptance coverage targets the production-like Compose contract,
internal-only ML services, manifest-first API serving, promoted prediction
payload loading, and monitoring wiring.

## Security hardening

The local production-like runtime is not yet a hardened production security
baseline. Future work should cover:

- secret management and rotation;
- separated runtime identities;
- stricter network exposure reviews;
- authentication and authorization hardening;
- production-grade configuration management;
- container user, volume permission, and capability reviews.

Security hardening should preserve the current Docker socket boundary: normal
pipeline execution must not require Docker socket access from Airflow, runner,
API, gateway, or ML step services.

## Scale-out execution

The runtime now has the structural pieces needed for local bounded scale-out:
Airflow submits typed jobs, `job-runner-api` dispatches through `ml-gateway`, and
ML step services can be replicated behind the gateway.

Remaining work should focus on proving and hardening the scale-out behavior:

- bounded multi-counter fan-out with conservative defaults;
- retry and idempotency policies under larger local workloads;
- resource limits and backpressure behavior per service family;
- acceptance validation for scaled ML service replicas;
- dashboard and alert coverage for queueing, failures, and slow jobs;
- concurrency review for manifest promotion and API refresh behavior.

Distributed orchestration remains future work when local Compose is no longer
enough.

## Full ETL source chain

The current pipeline still starts from an ML-ready raw CSV file. Future work
should connect ingestion and enrichment to the real external Paris bike counter
sources before feature generation and model execution.

The existing Paris dataset column names remain part of the current ETL contract
until that source-chain redesign explicitly changes them.

## Object-storage-first artifact handoff

Artifact manifests may record optional `s3://` metadata, but serving remains
local manifest-first today. Future work should cover:

- object upload from ML step services;
- object checksum verification;
- object download or streaming in the API;
- object-storage-first API serving;
- credential scoping for artifact access.

## Remote deployment and production operations

The current scope does not include remote deployment or production operations.
Future tracks may cover:

- Kubernetes or another remote workload runtime;
- durable queues or distributed workers;
- production ingress and TLS;
- performance benchmarks and load testing;
- backup, restore, and retention policies;
- operational SLOs and incident runbooks.

## Observability hardening

The current runtime records traceable identifiers across Airflow, runner, and ML
service logs. Future observability work should cover:

- alert rules aligned with model and API business impact;
- dashboard review for multi-counter scale;
- metric cardinality control;
- log retention and aggregation strategy;
- trace export beyond local log files;
- production retention strategy.
