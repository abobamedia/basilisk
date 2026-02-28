FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 (for Claude Code CLI — needed by claude_code_edit tool)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

COPY . .

# Git config for local commits
RUN git config --global user.email "e2e@test" \
    && git config --global user.name "E2E Test" \
    && git config --global init.defaultBranch main

CMD ["pytest", "tests/e2e/", "-v", "--tb=short", "-x"]
