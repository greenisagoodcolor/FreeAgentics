# Multi-stage build for FreeAgentics
FROM python:3.11.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app user with specific UID/GID
RUN groupadd -g 1001 app && \
    useradd -r -u 1001 -g app --create-home --shell /bin/bash app

# Development stage
FROM base as development

WORKDIR /app

# Copy requirements files
COPY requirements-core.txt requirements-dev.txt requirements.txt ./

# Create requirements-docker.txt without git dependencies
RUN grep -v "^-e git+" requirements.txt > requirements-docker.txt 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Development command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

WORKDIR /app

# Copy production requirements
COPY requirements-production.txt ./

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy source code (exclude dev files)
COPY agents/ ./agents/
COPY api/ ./api/
COPY inference/ ./inference/
COPY world/ ./world/
COPY coalitions/ ./coalitions/
COPY main.py ./
COPY pyproject.toml ./

# Change ownership to app user
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
