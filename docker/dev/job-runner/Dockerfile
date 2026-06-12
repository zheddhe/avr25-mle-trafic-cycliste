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

COPY docker/dev/job-runner/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN groupadd --gid "${APP_GID}" app && \
    useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --home-dir /app \
        --shell /usr/sbin/nologin \
        app && \
    mkdir -p /app/src/artifacts /app/src/ml/jobs /app/src/job_runner && \
    mkdir -p /app/logs/job-runner

COPY src/artifacts /app/src/artifacts
COPY src/ml/jobs /app/src/ml/jobs
COPY src/job_runner /app/src/job_runner

RUN chown -R app:app /app

USER app

EXPOSE 10080
CMD ["uvicorn", "src.job_runner.main:app", "--host", "0.0.0.0", "--port", "10080"]
