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

# Git config for auto-rescue commits inside container
RUN git config --global user.email "ouroboros@basilisk" && \
    git config --global user.name "Ouroboros" && \
    git config --global --add safe.directory /app

# Data volume for persistent state
VOLUME /data

# Seed memory files to avoid startup warnings
RUN mkdir -p /data/memory /data/state /data/logs && \
    echo "# Ouroboros Server Instance\nI am Ouroboros running on a remote VPS.\nMy owner communicates in Russian." > /data/memory/identity.md && \
    echo "# Working Memory" > /data/memory/scratchpad.md && \
    echo "# Environment\n- Platform: Linux VPS (Docker)\n- Owner language: Russian" > /data/memory/WORLD.md

ENV OUROBOROS_DATA_DIR=/data \
    OUROBOROS_REPO_DIR=/app \
    OUROBOROS_HOST=0.0.0.0 \
    OUROBOROS_SERVER_PORT=8765 \
    OUROBOROS_MAX_WORKERS=2 \
    PYTHONUNBUFFERED=1

EXPOSE 8765

CMD ["python", "server.py"]
