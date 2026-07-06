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
COPY pyproject.toml uv.lock /app/

# Install dependencies using uv into the system python environment
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy the rest of the application source code
COPY . /app

# Ensure persistent data directories exist and are writable
RUN mkdir -p /app/data && chmod 777 /app/data

# Create and switch to a non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uv", "run", "main.py"]
