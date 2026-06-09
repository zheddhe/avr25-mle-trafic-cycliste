FROM python:3.12-slim

ARG APP_GID=1000
ARG APP_UID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY docker/prod/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN groupadd --gid "${APP_GID}" app && \
    useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --home-dir /app \
        --shell /usr/sbin/nologin \
        app && \
    mkdir -p /app/src/api /app/src/artifacts /app/logs/api && \
    mkdir -p /app/data/final /app/artifacts/manifests && \
    chown -R app:app /app

COPY src/api /app/src/api
COPY src/artifacts /app/src/artifacts

USER app

EXPOSE 10000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "10000"]
