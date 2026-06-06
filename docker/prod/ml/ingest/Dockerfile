FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY docker/dev/ml/ingest/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN addgroup --system app && \
    adduser --system --ingroup app --home /app app && \
    mkdir -p /app/src/ml/ingest /app/data /app/logs && \
    chown -R app:app /app

COPY src/ml/ingest/ingest_utils.py /app/src/ml/ingest/ingest_utils.py
COPY src/ml/ingest/import_raw_data.py /app/src/ml/ingest/import_raw_data.py

USER app

ENTRYPOINT ["python", "-m", "src.ml.ingest.import_raw_data"]
