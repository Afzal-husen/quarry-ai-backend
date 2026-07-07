# syntax=docker/dockerfile:1
FROM python:3.14-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy package definition files
COPY pyproject.toml ./

# Install dependencies using uv into the system python environment
# --mount=type=cache persists uv's wheel cache on the Docker host between builds
# so unchanged packages are never re-downloaded.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml

# Copy the rest of the application source code
COPY . /app

# Ensure persistent data directories exist and are writable
RUN mkdir -p /app/data && chmod 777 /app/data

# Create and switch to a non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
