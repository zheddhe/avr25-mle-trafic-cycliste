FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Déps pour transformations légères
COPY docker/dev/ml/features/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Code strictement nécessaire (pas de tests)
RUN mkdir -p /app/src/ml/features
COPY src/ml/features/features_utils.py \
     /app/src/ml/features/features_utils.py
COPY src/ml/features/build_features.py \
     /app/src/ml/features/build_features.py

ENTRYPOINT ["python", "-m", "src.ml.features.build_features"]