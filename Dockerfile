FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first
COPY pyproject.toml .
COPY README.md .

# Copy source code BEFORE install (because setuptools needs src/)
COPY src/ ./src/

# Install your package + dependencies
RUN pip install --no-cache-dir .

# Copy runtime files
COPY main.py .
COPY config/ ./config/

ENV PYTHONPATH="/app"
ENV MLFLOW_ALLOW_FILE_STORE=true


CMD ["python", "main.py"]


