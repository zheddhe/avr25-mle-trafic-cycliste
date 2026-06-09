# Project documentation index

This folder is organized so contributors can distinguish current operation,
implemented architecture, and active next-phase design work.

The root `README.md` stays concise and project-oriented. Detailed runtime,
architecture, and implementation design decisions live here.

## Documentation level rules

Use each documentation area at one level of abstraction:

| Area | Scope | Rule |
| ---- | ----- | ---- |
| `current-runtime-and-operations/` | Implemented local commands, workspaces, service exposure, dependencies, and runtime ownership. | Describe what exists on `main` and how to operate it. Do not use this area for backlog, future behavior, or story-specific acceptance notes. |
| `architecture-references/` | Implemented cross-runtime boundaries, networks, communication paths, and security rules. | Describe the current architecture and its guardrails. Do not add roadmap tables or planned service-to-service paths here. |
| `next-phase-design/` | Active design and implementation coordination for the ongoing phase. | Keep only remaining design gaps and phase coordination. Do not keep separate implementation-outcome notes once stable wording has been folded into current runtime or architecture docs. |

Story-level details belong in GitHub issues and PRs. Documentation may keep
cross-cutting status only when it helps contributors understand the phase, and it
should link to the appropriate source document instead of repeating the same
information in several places.

When a design becomes implemented, move the stable current-state wording to
`current-runtime-and-operations/` or `architecture-references/`. Keep only the
remaining design gaps under `next-phase-design/`.

## Current runtime and operations

Use these documents to run or validate what exists on `main`.

| Document | Purpose |
| -------- | ------- |
| [`current-runtime-and-operations/local-prod-runtime.md`](current-runtime-and-operations/local-prod-runtime.md) | Current `docker/dev` and `docker/prod` runtime guide, workspace ownership, service exposure, runner API behavior, manifest-first API serving, and validation entrypoints. |
| [`current-runtime-and-operations/ports-and-services.md`](current-runtime-and-operations/ports-and-services.md) | Host-exposed ports, local URLs, and internal-only services for dev and production-like runtimes. |
| [`current-runtime-and-operations/dependency-strategy.md`](current-runtime-and-operations/dependency-strategy.md) | uv groups, custom images, upstream runtime images, healthchecks, and dependency upgrade policy. |
| [`current-runtime-and-operations/repository-structure.md`](current-runtime-and-operations/repository-structure.md) | Repository ownership rules, generated artifact expectations, DVC boundaries, and dev/prod runtime placement. |

## Architecture references

Use these documents to understand the implemented local MLOps architecture and the
rules that guide runtime changes.

| Document | Purpose |
| -------- | ------- |
| [`architecture-references/runtime-communication-matrix.md`](architecture-references/runtime-communication-matrix.md) | Service-to-service communication, runner execution boundary, manifest handoff paths, mount coupling, and current network traffic. |
| [`architecture-references/runtime-security-boundaries.md`](architecture-references/runtime-security-boundaries.md) | Runtime identities, Docker socket boundary, host exposure, and service privilege rules. |
| [`architecture-references/local-prod-network-topology.md`](architecture-references/local-prod-network-topology.md) | Implemented `docker/prod` functional network topology and current service placement. |

## Next Phase design

Use these documents when implementing or reviewing the active MLOps phase. Keep
this list short: each document should own an active coordination topic rather than
one completed implementation slice.

| Document | Purpose |
| -------- | ------- |
| [`next-phase-design/phase-8-production-minimal-target.md`](next-phase-design/phase-8-production-minimal-target.md) | Phase 8 closure target: minimal `docker/prod` execution path without Airflow Docker socket, using internal ML step services and manifest-first API serving. |
| [`next-phase-design/artifact-handoff-strategy.md`](next-phase-design/artifact-handoff-strategy.md) | Consolidated manifest-first artifact handoff contract, implemented coverage, and remaining artifact-serving gaps. |
| [`next-phase-design/airflow-job-runner-strategy.md`](next-phase-design/airflow-job-runner-strategy.md) | Remaining production-like Airflow orchestration design that chains typed runner jobs. |

## Reading order

For runtime work, read:

1. [`current-runtime-and-operations/local-prod-runtime.md`](current-runtime-and-operations/local-prod-runtime.md)
2. [`current-runtime-and-operations/repository-structure.md`](current-runtime-and-operations/repository-structure.md)
3. [`architecture-references/runtime-communication-matrix.md`](architecture-references/runtime-communication-matrix.md)
4. [`architecture-references/runtime-security-boundaries.md`](architecture-references/runtime-security-boundaries.md)

For active phase implementation, read:

1. [`next-phase-design/phase-8-production-minimal-target.md`](next-phase-design/phase-8-production-minimal-target.md)
2. [`next-phase-design/artifact-handoff-strategy.md`](next-phase-design/artifact-handoff-strategy.md)
3. [`next-phase-design/airflow-job-runner-strategy.md`](next-phase-design/airflow-job-runner-strategy.md)
4. [`architecture-references/local-prod-network-topology.md`](architecture-references/local-prod-network-topology.md)
5. The GitHub issue for the current implementation task.
