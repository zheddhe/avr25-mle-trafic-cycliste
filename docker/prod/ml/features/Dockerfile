FROM python:3.12-slim

ARG APP_GID=1000
ARG APP_UID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

COPY docker/dev/ml/features/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN groupadd --gid "${APP_GID}" app && \
    useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --home-dir /app \
        --shell /usr/sbin/nologin \
        app && \
    mkdir -p /app/src/metrics /app/src/ml/features /app/data /app/logs && \
    chown -R app:app /app

COPY src/metrics /app/src/metrics
COPY src/ml/features/features_utils.py /app/src/ml/features/features_utils.py
COPY src/ml/features/build_features.py /app/src/ml/features/build_features.py

USER app

ENTRYPOINT ["python", "-m", "src.ml.features.build_features"]
