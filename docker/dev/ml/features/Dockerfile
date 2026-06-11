FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY docker/dev/ml/features/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/logs /app/artifacts /app/models \
    && chown -R app:app /app

COPY --chown=app:app src/artifacts /app/src/artifacts
COPY --chown=app:app src/metrics /app/src/metrics
COPY --chown=app:app src/ml/jobs /app/src/ml/jobs
COPY --chown=app:app src/ml/services /app/src/ml/services
COPY --chown=app:app src/ml/features /app/src/ml/features

USER app

EXPOSE 10082

CMD ["uvicorn", "src.ml.services.features_main:app", "--host", "0.0.0.0", "--port", "10082"]
