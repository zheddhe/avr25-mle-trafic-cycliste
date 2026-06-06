# Project documentation index

This folder is organized so contributors can distinguish:

1. current runtime and operations;
2. architecture references;
3. next-phase target designs.

The root `README.md` stays concise and project-oriented. Detailed runtime,
architecture, and Phase 8 design decisions live here.

## Current runtime and operations

Use these documents to run or validate what exists on `main`.

| Document | Purpose |
| -------- | ------- |
| [`current-runtime-and-operations/local-prod-runtime.md`](current-runtime-and-operations/local-prod-runtime.md) | Current `docker/dev` and `docker/prod` runtime guide, workspace ownership, remaining exceptions, and validation entrypoints. |
| [`current-runtime-and-operations/ports-and-services.md`](current-runtime-and-operations/ports-and-services.md) | Host-exposed ports, local URLs, and internal-only services for dev and production-like runtimes. |
| [`current-runtime-and-operations/dependency-strategy.md`](current-runtime-and-operations/dependency-strategy.md) | uv groups, custom images, upstream runtime images, healthchecks, and dependency upgrade policy. |
| [`current-runtime-and-operations/repository-structure.md`](current-runtime-and-operations/repository-structure.md) | Repository ownership rules, generated artifact expectations, DVC boundaries, and dev/prod runtime placement. |

## Architecture references

Use these documents to understand the current local MLOps architecture and the
rules that should guide future runtime changes.

| Document | Purpose |
| -------- | ------- |
| [`architecture-references/runtime-communication-matrix.md`](architecture-references/runtime-communication-matrix.md) | Service-to-service communication, mount coupling, and Phase 8 additions. |
| [`architecture-references/runtime-security-boundaries.md`](architecture-references/runtime-security-boundaries.md) | Runtime identities, Docker socket boundary, host exposure, and security hardening direction. |
| [`architecture-references/local-prod-network-topology.md`](architecture-references/local-prod-network-topology.md) | Implemented `docker/prod` functional network topology and rules for future service placement. |

## Next Phase design

Use these documents when implementing Phase 8 stories.

| Document | Purpose |
| -------- | ------- |
| [`next-phase-design/artifact-handoff-strategy.md`](next-phase-design/artifact-handoff-strategy.md) | Hybrid manifest-first artifact promotion contract using local runtime paths and optional MinIO object URIs. |
| [`next-phase-design/airflow-job-runner-strategy.md`](next-phase-design/airflow-job-runner-strategy.md) | Target runner-based Airflow execution model that replaces DockerOperator in `docker/prod`. |

## Reading order

For runtime work, read:

1. [`current-runtime-and-operations/local-prod-runtime.md`](current-runtime-and-operations/local-prod-runtime.md)
2. [`current-runtime-and-operations/repository-structure.md`](current-runtime-and-operations/repository-structure.md)
3. [`architecture-references/runtime-communication-matrix.md`](architecture-references/runtime-communication-matrix.md)
4. [`architecture-references/runtime-security-boundaries.md`](architecture-references/runtime-security-boundaries.md)

For Phase 8 implementation, read:

1. [`next-phase-design/artifact-handoff-strategy.md`](next-phase-design/artifact-handoff-strategy.md)
2. [`next-phase-design/airflow-job-runner-strategy.md`](next-phase-design/airflow-job-runner-strategy.md)
3. [`architecture-references/local-prod-network-topology.md`](architecture-references/local-prod-network-topology.md)
4. The issue body for the current Phase 8 story.

## Documentation rules

- Keep the root `README.md` concise and project-oriented.
- Put runtime commands and workspace ownership in `current-runtime-and-operations/`.
- Put cross-runtime architecture rules in `architecture-references/`.
- Put implementation roadmap details in Phase 8 story issues, not in the root README.
- When a target design becomes implemented, move its wording from future tense to
  current-state wording and keep only remaining gaps in a dedicated section.
