FROM python:3.11-slim

WORKDIR /app

# System deps for git operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps (no browser/local-model for server)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    openai>=1.66.0 \
    requests \
    dulwich \
    starlette \
    "uvicorn[standard]" \
    websockets

COPY . .

# Data volume for persistent state
VOLUME /data

ENV OUROBOROS_DATA_DIR=/data \
    OUROBOROS_REPO_DIR=/app \
    OUROBOROS_HOST=0.0.0.0 \
    OUROBOROS_SERVER_PORT=8765 \
    OUROBOROS_MAX_WORKERS=2 \
    PYTHONUNBUFFERED=1

EXPOSE 8765

CMD ["python", "server.py"]
