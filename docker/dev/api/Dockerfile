FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY docker/dev/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/src/api /app/src/artifacts /app/data/final \
    /app/artifacts/manifests /app/logs/api
COPY src/api/main.py /app/src/api/main.py
COPY src/artifacts /app/src/artifacts

EXPOSE 10000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "10000"]
