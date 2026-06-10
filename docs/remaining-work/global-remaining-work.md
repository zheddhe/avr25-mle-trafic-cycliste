# Global remaining work

This document lists improvement axes that remain outside the validated local
production-like baseline.

The current baseline is documented in:

- [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md);
- [`../architecture-references/runtime-communication-matrix.md`](../architecture-references/runtime-communication-matrix.md);
- [`../architecture-references/local-prod-network-topology.md`](../architecture-references/local-prod-network-topology.md).

## Current validated baseline

The local production-like runtime validates this path:

```text
Airflow DAG task
  -> job-runner-api
  -> ml-ingest-prod / ml-features-prod / ml-models-prod
  -> promoted prediction manifest
  -> authenticated API refresh
  -> FastAPI serving from the promoted manifest payload
  -> Prometheus and Grafana observability
```

Acceptance tests cover the production-like Compose contract, internal-only ML
services, manifest-first API serving, promoted prediction payload loading, and
monitoring wiring.

## Security hardening

The local production-like runtime is not yet a hardened production security
baseline. Future work should cover:

- secret management and rotation;
- separated runtime identities;
- stricter network exposure reviews;
- authentication and authorization hardening;
- production-grade configuration management;
- container user, volume permission, and service capability reviews.

Security hardening will be reassessed after Phase 9 scale-out work clarifies the
runtime concurrency and resource model.

## Scale-out execution

The current multi-counter orchestration remains intentionally sequential for the
local production-like path.

Phase 9 starts from the bounded local scale-out contract documented in
[`phase-9-bounded-scale-out-contract.md`](phase-9-bounded-scale-out-contract.md).
That contract defines the future-state scope before implementation stories
change the current runtime behavior.

Phase 9 should address:

- safe bounded parallel execution across counters;
- concurrency control for manifests and promoted `current.json` files;
- retry and idempotency policies for larger local workloads;
- per-service resource limits and backpressure strategy;
- acceptance validation and observability for bounded scale-out.

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

The current production-like runtime validates Pushgateway, Prometheus, and
Grafana wiring. Future observability work should cover:

- alert rules aligned with model and API business impact;
- dashboard review for multi-counter scale;
- metric cardinality control;
- production retention strategy;
- run-level traceability from Airflow to API artifact state.
