# Image légère Python
FROM python:3.11-slim

# Variables d'environnement pour limiter les warnings Python
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Installer dépendances système nécessaires à psycopg2 et scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer MLflow avec extras + connecteurs nécessaires
RUN pip install --upgrade pip && \
    pip install "setuptools<81" && \
    pip install \
      "mlflow[extras]==2.22.1" \
      psycopg2-binary \
      boto3

ENV PYTHONWARNINGS="ignore:pkg_resources"

# Exposer le port par défaut du serveur MLflow
EXPOSE 5000

# Commande par défaut (sera surchargée dans docker-compose si besoin)
ENTRYPOINT ["mlflow"]
CMD ["--help"]