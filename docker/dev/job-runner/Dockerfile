FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY docker/dev/job-runner/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/logs \
    && chown -R app:app /app

COPY --chown=app:app src/artifacts /app/src/artifacts
COPY --chown=app:app src/ml/jobs /app/src/ml/jobs
COPY --chown=app:app src/job_runner /app/src/job_runner

USER app

EXPOSE 10080

CMD ["uvicorn", "src.job_runner.main:app", "--host", "0.0.0.0", "--port", "10080"]
