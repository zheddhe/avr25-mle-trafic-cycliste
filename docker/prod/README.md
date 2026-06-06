# Local production-like Docker Compose runtime

This folder contains the local production-like runtime introduced for Phase 7.

It is intentionally separate from the deprecated root `docker-compose.yaml`
compatibility entrypoint and the `docker/dev` assets. The development runtime
remains the correct target for interactive debugging, broad host visibility, and
current DockerOperator-based Airflow ML jobs.

## Validate

```bash
make prod-compose-config
```

## Start

```bash
make prod-start
```

The default production-like profile is `ptf`. Override it when needed:

```bash
make prod-start PROD_PROFILE=monitoring
```

## Stop

```bash
make prod-stop
make prod-down
```

## Design constraints

- Do not mount `/var/run/docker.sock` in Airflow.
- Do not reuse the broad `mlops_net` development network.
- Keep only local operator-facing services published to the host.
- Prefer read-only runtime configuration mounts.
- Keep production-like generated artifacts under `docker/prod/runtime` and use
  the Phase 8 manifest handoff contract for promoted artifacts.
- Run custom API and ML containers as a non-root application user.

See `docs/current-runtime-and-operations/local-prod-runtime.md` for the full
operating model and limitations.
