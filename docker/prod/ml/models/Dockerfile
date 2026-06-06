FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY docker/dev/ml/models/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN addgroup --system app && \
    adduser --system --ingroup app --home /app app && \
    mkdir -p /app/src/ml/models /app/data /app/models /app/logs && \
    chown -R app:app /app

COPY src/ml/models/models_utils.py /app/src/ml/models/models_utils.py
COPY src/ml/models/mlflow_tracking.py /app/src/ml/models/mlflow_tracking.py
COPY src/ml/models/train_and_predict.py /app/src/ml/models/train_and_predict.py

USER app

ENTRYPOINT ["python", "-m", "src.ml.models.train_and_predict"]
