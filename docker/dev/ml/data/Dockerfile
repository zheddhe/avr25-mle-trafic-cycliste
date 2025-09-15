FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Déps minimales pour l'ingestion
COPY docker/dev/ml/data/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Code strictement nécessaire (pas de tests)
RUN mkdir -p /app/src/ml/data
COPY src/ml/data/data_utils.py /app/src/ml/data/data_utils.py
COPY src/ml/data/import_raw_data.py /app/src/ml/data/import_raw_data.py

# Par défaut on lance la CLI ; compose fournira les options
ENTRYPOINT ["python", "-m", "src.ml.data.import_raw_data"]