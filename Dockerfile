# syntax=docker/dockerfile:1
FROM python:3.14-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    FASTEMBED_CACHE_DIR=/app/models/fastembed \
    FLASHRANK_CACHE_DIR=/app/models/flashrank

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy package dependency definition
COPY requirements.txt ./

# Install dependencies using uv into the system python environment
# --mount=type=cache persists uv's wheel cache on the Docker host between builds
# so unchanged packages are never re-downloaded.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Pre-download and bake FastEmbed and FlashRank model weights during image building
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/models/fastembed'); TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/models/fastembed')"
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/models/flashrank')"

# Copy the rest of the application source code
COPY . /app

# Ensure persistent data and model directories exist and are writable
RUN mkdir -p /app/data /app/models && chmod 777 /app/data /app/models

# Create and switch to a non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
