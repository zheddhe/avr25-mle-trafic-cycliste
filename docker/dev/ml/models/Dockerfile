FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    GIT_PYTHON_REFRESH="quiet"

WORKDIR /app

COPY docker/dev/ml/models/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /app/src/artifacts /app/src/metrics /app/src/ml/models
COPY src/artifacts /app/src/artifacts
COPY src/metrics /app/src/metrics
COPY src/ml/models/artifact_manifest_emission.py \
     /app/src/ml/models/artifact_manifest_emission.py
COPY src/ml/models/models_utils.py \
     /app/src/ml/models/models_utils.py
COPY src/ml/models/mlflow_tracking.py \
     /app/src/ml/models/mlflow_tracking.py
COPY src/ml/models/train_and_predict.py \
     /app/src/ml/models/train_and_predict.py

ENTRYPOINT ["python", "-m", "src.ml.models.train_and_predict"]
