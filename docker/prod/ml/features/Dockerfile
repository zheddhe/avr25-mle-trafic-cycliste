FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY docker/dev/ml/features/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN addgroup --system app && \
    adduser --system --ingroup app --home /app app && \
    mkdir -p /app/src/ml/features /app/data /app/logs && \
    chown -R app:app /app

COPY src/ml/features/features_utils.py /app/src/ml/features/features_utils.py
COPY src/ml/features/build_features.py /app/src/ml/features/build_features.py

USER app

ENTRYPOINT ["python", "-m", "src.ml.features.build_features"]
