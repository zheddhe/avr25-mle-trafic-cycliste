# Local development Docker Compose runtime

This folder contains the canonical local development runtime.

It mirrors the `docker/prod` folder structure so contributors can compare the
development and local production-like runtimes without reading unrelated project
setup targets.

Use the runtime through Make targets or an explicit Compose file:

```bash
make dev-compose-config
make dev-start

docker compose --env-file .env -f docker/dev/docker-compose.yaml config
```

## Validate

```bash
make dev-compose-config
```

## Start

```bash
make dev-start
```

The default development profile is `ptf`. Override it when needed:

```bash
make dev-start DEV_PROFILE=api
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

See `docs/current-runtime-and-operations/local-prod-runtime.md` for the
production-like counterpart and the expected divergence points.
