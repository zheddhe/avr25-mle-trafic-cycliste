# Local production-like Docker Compose runtime

This folder contains the local production-like runtime introduced for Phase 7.

It is intentionally separate from the root `docker-compose.yaml` and the
`docker/dev` assets. The development runtime remains the correct target for
interactive debugging, broad host visibility, and current DockerOperator-based
Airflow ML jobs.

## Validate

```bash
make -f docker/prod/Makefile prod-compose-config
```

## Start

```bash
make -f docker/prod/Makefile prod-ops
```

The default production-like profile is `ptf`. Override it when needed:

```bash
make -f docker/prod/Makefile prod-ops PROD_PROFILE=monitoring
```

## Stop

```bash
make -f docker/prod/Makefile prod-stop
make -f docker/prod/Makefile prod-down
```

## Design constraints

- Do not mount `/var/run/docker.sock` in Airflow.
- Do not reuse the broad `mlops_net` development network.
- Keep only local operator-facing services published to the host.
- Prefer read-only runtime configuration mounts.
- Keep temporary data, model, and log bind mounts documented until an artifact
  handoff story replaces them.
- Run custom API and ML containers as a non-root application user.

See `docs/local-prod-runtime.md` for the full operating model and limitations.
