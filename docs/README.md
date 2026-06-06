# Project documentation index

This folder is organized so contributors can distinguish current operating
instructions, architecture references, and Phase 8 target designs.

## Current runtime and operations

Use these documents to run or validate what exists on `main` now.

| Document | Purpose |
| -------- | ------- |
| [`local-prod-runtime.md`](local-prod-runtime.md) | Current `docker/dev` and `docker/prod` runtime guide, workspace ownership, remaining exceptions, and validation entrypoints. |
| [`ports-and-services.md`](ports-and-services.md) | Host-exposed ports, local URLs, and internal-only services for dev and production-like runtimes. |
| [`dependency-strategy.md`](dependency-strategy.md) | uv groups, custom images, upstream runtime images, healthchecks, and dependency upgrade policy. |
| [`repository-structure.md`](repository-structure.md) | Repository ownership rules, generated artifact expectations, DVC boundaries, and dev/prod runtime placement. |

## Architecture references

Use these documents to understand the current local MLOps architecture and the
rules that should guide future runtime changes.

| Document | Purpose |
| -------- | ------- |
| [`runtime-communication-matrix.md`](runtime-communication-matrix.md) | Service-to-service communication, mount coupling, and Phase 8 additions. |
| [`runtime-security-boundaries.md`](runtime-security-boundaries.md) | Runtime identities, Docker socket boundary, host exposure, and security hardening direction. |
| [`local-prod-network-topology.md`](local-prod-network-topology.md) | Implemented `docker/prod` functional network topology and rules for future service placement. |

## Phase 8 target designs

Use these documents when implementing Phase 8 stories.

| Document | Purpose |
| -------- | ------- |
| [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md) | Hybrid manifest-first artifact promotion contract using local runtime paths and optional MinIO object URIs. |
| [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md) | Target runner-based Airflow execution model that replaces DockerOperator in `docker/prod`. |

## Reading order

For runtime work, read:

1. [`local-prod-runtime.md`](local-prod-runtime.md)
2. [`repository-structure.md`](repository-structure.md)
3. [`runtime-communication-matrix.md`](runtime-communication-matrix.md)
4. [`runtime-security-boundaries.md`](runtime-security-boundaries.md)

For Phase 8 implementation, read:

1. [`artifact-handoff-strategy.md`](artifact-handoff-strategy.md)
2. [`airflow-job-runner-strategy.md`](airflow-job-runner-strategy.md)
3. [`local-prod-network-topology.md`](local-prod-network-topology.md)
4. The issue body for the current Phase 8 story.

## Documentation rules

- Keep the root `README.md` concise and project-oriented.
- Put operational runtime details in `local-prod-runtime.md`.
- Put host port inventories in `ports-and-services.md`.
- Put implementation roadmap details in Phase 8 story issues, not in the root
  README.
- When a target design becomes implemented, move its wording from future tense to
  current-state wording and keep only the remaining gaps in a dedicated section.
