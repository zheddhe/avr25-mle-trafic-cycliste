# Global remaining work

This document is the durable backlog of cross-cutting improvement axes that
remain outside the validated local runtime baseline.

It receives the residual work of completed phases. A closed phase document must
not remain as a second backlog once its implemented behaviour is documented as
current state and its unresolved cross-cutting work is captured here.

The current baseline is documented in:

- [`../current-runtime-and-operations/local-prod-runtime.md`](../current-runtime-and-operations/local-prod-runtime.md);
- [`../current-runtime-and-operations/runtime-logging.md`](../current-runtime-and-operations/runtime-logging.md);
- [`../architecture-references/runtime-communication-matrix.md`](../architecture-references/runtime-communication-matrix.md);
- [`../architecture-references/execution-and-artifact-promotion-contract.md`](../architecture-references/execution-and-artifact-promotion-contract.md);
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

The baseline includes local bounded multi-counter execution, explicit Airflow and
runner concurrency limits, manifest-first promotion, and traceable job evidence.

Acceptance coverage targets the production-like Compose contract,
internal-only ML services, manifest-first API serving, promoted prediction
payload loading, bounded local execution, and monitoring wiring.

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

## Distributed execution and larger-scale operations

Phase 9 completed the bounded local scale-out baseline. Remaining work begins
where local Compose limits are no longer sufficient:

- remote workload orchestration or a distributed worker runtime;
- durable queues and distributed worker pools;
- capacity planning and load testing beyond the local Compose environment;
- resource isolation and scheduling policy for larger workloads;
- operational SLOs and failure recovery across remote execution domains.

## Full ETL source chain

The current pipeline still starts from an ML-ready raw CSV file. Future work
should connect ingestion and enrichment to the real external Paris bike counter
sources before feature generation and model execution.

The existing Paris dataset column names remain part of the current ETL contract
until that source-chain redesign explicitly changes them.

The target source lifecycle is defined in
[`phase-10-etl-source-chain.md`](phase-10-etl-source-chain.md).

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

- production ingress and TLS;
- backup, restore, and retention policies;
- deployment promotion and rollback procedures;
- incident runbooks and operational ownership;
- environment-specific configuration and secret delivery.

## Observability hardening

The current runtime records traceable identifiers across Airflow, runner, and ML
service logs. Future observability work should cover:

- alert rules aligned with model and API business impact;
- dashboard review for larger-scale and remote execution;
- metric cardinality control;
- log retention and aggregation strategy;
- trace export beyond local log files;
- production retention strategy.
