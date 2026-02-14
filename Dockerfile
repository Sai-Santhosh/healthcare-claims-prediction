# =============================================================================
# Dockerfile for Medical Claims Data Engineering Pipeline
# Multi-stage build for optimized production image
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production
# -----------------------------------------------------------------------------
FROM python:3.10-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r pipeline && useradd -r -g pipeline pipeline

# Copy application code
COPY --chown=pipeline:pipeline src/ /app/src/
COPY --chown=pipeline:pipeline config/ /app/config/
COPY --chown=pipeline:pipeline dbt/ /app/dbt/

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/bronze /app/data/silver /app/data/gold \
    /app/logs /app/reports && \
    chown -R pipeline:pipeline /app

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT="production"

# Switch to non-root user
USER pipeline

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "src.etl.run_pipeline"]

# -----------------------------------------------------------------------------
# Stage 3: Development
# -----------------------------------------------------------------------------
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

# Install DBT
RUN pip install --no-cache-dir dbt-core dbt-postgres

USER pipeline

CMD ["python", "-m", "pytest", "tests/", "-v"]

# -----------------------------------------------------------------------------
# Stage 4: Lambda (for AWS Lambda deployment)
# -----------------------------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.10 as lambda

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY config/ ${LAMBDA_TASK_ROOT}/config/

# Set Python path
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}"

# Lambda handler
CMD ["src.aws.lambda.etl_handler.handler"]
