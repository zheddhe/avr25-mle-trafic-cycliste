FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY docker/dev/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TODO: préferer inclure les fichiers unitairement un à un (liste blanche)
COPY src/ ./src/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]