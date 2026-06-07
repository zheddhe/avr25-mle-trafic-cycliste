FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY docker/dev/ml/ingest/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /app/src/artifacts /app/src/metrics /app/src/ml/ingest
COPY src/artifacts /app/src/artifacts
COPY src/metrics /app/src/metrics
COPY src/ml/artifact_manifest_emission.py /app/src/ml/artifact_manifest_emission.py
COPY src/ml/ingest/ingest_utils.py /app/src/ml/ingest/ingest_utils.py
COPY src/ml/ingest/import_raw_data.py /app/src/ml/ingest/import_raw_data.py

ENTRYPOINT ["python", "-m", "src.ml.ingest.import_raw_data"]
