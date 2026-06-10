# Project documentation index

This folder separates current operations, implemented architecture,
documentation assets, and global remaining work.

The root `README.md` stays concise and project-oriented. Detailed runtime,
architecture, and implementation decisions live here.

## Documentation level rules

| Area | Scope | Rule |
| ---- | ----- | ---- |
| `current-runtime-and-operations/` | Implemented local commands, workspaces, service exposure, dependencies, and runtime ownership. | Describe what exists and how to operate it. |
| `architecture-references/` | Implemented cross-runtime boundaries, networks, communication paths, and runtime guardrails. | Describe the current architecture. |
| `assets/` | Documentation-only icons and rendered diagrams. | Store visuals used by Markdown docs only. |
| `remaining-work/` | Future improvement axes, active phase contracts, and not-yet-implemented design targets outside the validated local production-like baseline. | Keep global remaining work and future-state contracts here until implementation is validated. |

Story-level details belong in GitHub issues and pull requests. When a design
becomes implemented, move stable current-state wording to current runtime or
architecture docs. Remaining gaps belong under `remaining-work/`.

## Current runtime and operations

| Document | Purpose |
| -------- | ------- |
| [`current-runtime-and-operations/local-prod-runtime.md`](current-runtime-and-operations/local-prod-runtime.md) | Current `docker/dev` and `docker/prod` runtime guide, workspace ownership, service exposure, runner API behavior, manifest-first API serving, and validation entrypoints. |
| [`current-runtime-and-operations/ports-and-services.md`](current-runtime-and-operations/ports-and-services.md) | Host-exposed ports, local URLs, and internal-only services for dev and production-like runtimes. |
| [`current-runtime-and-operations/dependency-strategy.md`](current-runtime-and-operations/dependency-strategy.md) | uv groups, custom images, upstream runtime images, healthchecks, and dependency upgrade policy. |
| [`current-runtime-and-operations/repository-structure.md`](current-runtime-and-operations/repository-structure.md) | Repository ownership rules, generated artifact expectations, DVC boundaries, and dev/prod runtime placement. |

## Architecture references

| Document | Purpose |
| -------- | ------- |
| [`architecture-references/runtime-communication-matrix.md`](architecture-references/runtime-communication-matrix.md) | Service-to-service communication, runner execution boundary, manifest handoff paths, mount coupling, and current network traffic. |
| [`architecture-references/runtime-security-boundaries.md`](architecture-references/runtime-security-boundaries.md) | Runtime identities, Docker socket boundary, host exposure, and service privilege rules. |
| [`architecture-references/local-prod-network-topology.md`](architecture-references/local-prod-network-topology.md) | Implemented `docker/prod` functional network topology and current service placement. |

## Documentation assets

| Document | Purpose |
| -------- | ------- |
| [`assets/README.md`](assets/README.md) | Asset ownership rules for icons and rendered diagrams. |
| [`assets/diagrams/local-prod-architecture-overview.png`](assets/diagrams/local-prod-architecture-overview.png) | Rendered architecture overview used for onboarding. Mermaid diagrams and service tables remain the maintained architecture contract. |

## Remaining work

| Document | Purpose |
| -------- | ------- |
| [`remaining-work/global-remaining-work.md`](remaining-work/global-remaining-work.md) | Security, scale-out, full ETL source chain, object-storage-first handoff, remote operations, and observability hardening. |
| [`remaining-work/phase-9-bounded-scale-out-contract.md`](remaining-work/phase-9-bounded-scale-out-contract.md) | Future-state Phase 9 contract for bounded local production-like scale-out before implementation changes current runtime behavior. |

## Reading order

For runtime work, read:

1. [`current-runtime-and-operations/local-prod-runtime.md`](current-runtime-and-operations/local-prod-runtime.md)
2. [`current-runtime-and-operations/repository-structure.md`](current-runtime-and-operations/repository-structure.md)
3. [`architecture-references/runtime-communication-matrix.md`](architecture-references/runtime-communication-matrix.md)
4. [`architecture-references/runtime-security-boundaries.md`](architecture-references/runtime-security-boundaries.md)

For future improvement planning, read:

1. [`remaining-work/global-remaining-work.md`](remaining-work/global-remaining-work.md)
2. [`remaining-work/phase-9-bounded-scale-out-contract.md`](remaining-work/phase-9-bounded-scale-out-contract.md)
3. [`architecture-references/local-prod-network-topology.md`](architecture-references/local-prod-network-topology.md)
