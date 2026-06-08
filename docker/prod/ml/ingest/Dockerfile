FROM python:3.12-slim

ARG APP_GID=1000
ARG APP_UID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY docker/prod/ml/ingest/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN groupadd --gid "${APP_GID}" app && \
    useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --home-dir /app \
        --shell /usr/sbin/nologin \
        app && \
    mkdir -p \
        /app/artifacts \
        /app/data \
        /app/logs \
        /app/src/artifacts \
        /app/src/metrics \
        /app/src/ml/ingest \
        /app/src/ml/jobs \
        /app/src/ml/services && \
    chown -R app:app /app

COPY src/artifacts /app/src/artifacts
COPY src/metrics /app/src/metrics
COPY src/ml/jobs /app/src/ml/jobs
COPY src/ml/services /app/src/ml/services
COPY src/ml/ingest/artifact_manifest_emission.py \
     /app/src/ml/ingest/artifact_manifest_emission.py
COPY src/ml/ingest/ingest_utils.py /app/src/ml/ingest/ingest_utils.py
COPY src/ml/ingest/import_raw_data.py /app/src/ml/ingest/import_raw_data.py

USER app

CMD [
    "uvicorn",
    "src.ml.services.ingest_main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "10081"
]
