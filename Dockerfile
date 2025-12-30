# Aegis Insight - Dockerfile
# ==========================
#
# Multi-stage build for optimized container size
#

# ============================================================
# Stage 1: Build stage
# ============================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================
# Stage 2: Runtime stage
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    netcat-openbsd \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Pre-download HuggingFace model for offline use
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create directories
RUN mkdir -p /app/data /app/logs /app/demo-data /app/scripts /app/web

# Copy application code
COPY *.py ./

# Copy src module
COPY src/ ./src/

# Copy web frontend
COPY web/ ./web/

# Copy scripts
COPY scripts/ ./scripts/
RUN chmod +x /app/scripts/*.sh /app/scripts/*.py 2>/dev/null || true

# Copy demo data (if present)
COPY demo-data/ ./demo-data/

# Copy entrypoint
COPY docker-entrypoint.sh .
RUN chmod +x /app/docker-entrypoint.sh

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/api/health/simple || exit 1

# Entry point
ENTRYPOINT ["/app/docker-entrypoint.sh"]
