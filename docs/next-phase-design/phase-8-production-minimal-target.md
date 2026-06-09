# Phase 8 remaining improvement axes

Phase 8 now targets closure through the production-like acceptance harness in
issue #72. Implemented runtime behavior is documented in current runtime and
architecture references, not in next-phase design notes.

This document keeps only the future-facing axes that remain deliberately out of
scope for the Phase 8 closure PR.

## Security hardening

The local production-like runtime is not yet a hardened production security
baseline. Future work should cover:

- secret management and rotation;
- separated runtime identities;
- stricter network exposure reviews;
- authentication and authorization hardening;
- production-grade configuration management.

## Scale-out execution

The current multi-counter orchestration remains intentionally sequential for the
local production-like path. Future work should address:

- safe parallel execution across counters;
- distributed orchestration when local Compose is no longer enough;
- concurrency control for manifests and promoted `current.json` files;
- retry and idempotency policies for larger workloads.

## Full ETL source chain

The current pipeline still starts from an ML-ready raw CSV file. Future work
should connect ingestion and enrichment to the real external Paris bike counter
sources before feature generation and model execution.

The existing Paris dataset column names remain part of the current ETL contract
until that source-chain redesign explicitly changes them.

## Object-storage-first artifact handoff

Artifact manifests may record optional `s3://` metadata, but serving remains
local manifest-first today. Future work should cover object upload, object
checksum verification, object download, and object-storage-first API serving.

## Remote deployment and production operations

The Phase 8 closure does not include remote deployment or production operations.
Future tracks may cover Kubernetes, cloud execution, durable queues, performance
benchmarks, load testing, and operational SLOs.
