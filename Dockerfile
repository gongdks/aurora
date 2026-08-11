# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies first (cache layer)
COPY pyproject.toml ./
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# ---- Runtime stage ----
FROM python:3.12-slim

WORKDIR /app

# Install only runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY agent/ ./agent/
COPY app_tk.py ./

# Create volumes for persistent data
RUN mkdir -p /app/agent_notes /app/agent_workspace

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

# Default: API mode. Override for GUI.
CMD ["python", "-m", "agent.api"]
