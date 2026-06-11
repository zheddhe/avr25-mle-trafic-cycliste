FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    GIT_PYTHON_REFRESH="quiet"

WORKDIR /app

COPY docker/dev/ml/models/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/logs /app/artifacts /app/models \
    && chown -R app:app /app

COPY --chown=app:app src/artifacts /app/src/artifacts
COPY --chown=app:app src/metrics /app/src/metrics
COPY --chown=app:app src/ml/jobs /app/src/ml/jobs
COPY --chown=app:app src/ml/services /app/src/ml/services
COPY --chown=app:app src/ml/models /app/src/ml/models

USER app

EXPOSE 10083

CMD ["uvicorn", "src.ml.services.models_main:app", "--host", "0.0.0.0", "--port", "10083"]
