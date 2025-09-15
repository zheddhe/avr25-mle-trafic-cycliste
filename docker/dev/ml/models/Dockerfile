FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Déps pour entraînement + tracking
COPY docker/dev/ml/models/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Code strictement nécessaire (pas de tests)
RUN mkdir -p /app/src/ml/models
COPY src/ml/models/models_utils.py \
     /app/src/ml/models/models_utils.py
COPY src/ml/models/mlflow_tracking.py \
     /app/src/ml/models/mlflow_tracking.py
COPY src/ml/models/train_and_predict.py \
     /app/src/ml/models/train_and_predict.py

ENTRYPOINT ["python", "-m", "src.ml.models.train_and_predict"]