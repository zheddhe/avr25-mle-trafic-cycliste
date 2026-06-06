# Local development Docker Compose runtime

This folder contains the canonical local development runtime.

It mirrors the `docker/prod` folder structure so contributors can compare the
development and local production-like runtimes without reading unrelated project
setup targets.

The root `docker-compose.yaml` is kept as a compatibility entrypoint for now.
New operational targets should prefer `docker/dev/docker-compose.yaml` through
`docker/dev/Makefile` or the root Makefile aliases.

## Validate

```bash
make dev-compose-config
```

Equivalent explicit command:

```bash
make -f docker/dev/Makefile dev-compose-config
```

## Start

```bash
make dev-ops
```

The default development profile is `ptf`. Override it when needed:

```bash
make dev-ops DEV_PROFILE=api
```

## Stop

```bash
make dev-stop
make dev-down
```

## Design constraints

- Keep broad host visibility for local debugging.
- Keep current DockerOperator-based Airflow ML execution available.
- Keep `data`, `logs`, and `models` as root-level local/DVC workspaces.
- Keep development UIs and debug ports available.
- Keep comments, sections, and property ordering close to `docker/prod` so diffs
  remain easy to review.

See `docs/local-prod-runtime.md` for the production-like counterpart and the
expected divergence points.
