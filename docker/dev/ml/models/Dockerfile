FROM python:3.12-slim

ARG APP_GID=1000
ARG APP_UID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY docker/dev/ml/models/requirements.txt /tmp/requirements.txt
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
        /app/models \
        /app/src/artifacts \
        /app/src/metrics \
        /app/src/ml/models \
        /app/src/ml/jobs \
        /app/src/ml/services

COPY src/artifacts /app/src/artifacts
COPY src/common /app/src/common
COPY src/metrics /app/src/metrics
COPY src/ml/jobs /app/src/ml/jobs
COPY src/ml/services /app/src/ml/services
COPY src/ml/models/artifact_manifest_emission.py \
     /app/src/ml/models/artifact_manifest_emission.py
COPY src/ml/models/models_utils.py /app/src/ml/models/models_utils.py
COPY src/ml/models/mlflow_tracking.py /app/src/ml/models/mlflow_tracking.py
COPY src/ml/models/train_and_predict.py \
     /app/src/ml/models/train_and_predict.py

RUN chown -R app:app /app

USER app

CMD ["uvicorn", "src.ml.services.models_main:app", "--host", "0.0.0.0", "--port", "10083"]
