FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONWARNINGS="ignore:pkg_resources"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "setuptools<81" && \
    pip install \
      "mlflow==2.22.1" \
      psycopg2-binary \
      boto3

EXPOSE 5000

ENTRYPOINT ["mlflow"]
CMD ["--help"]
