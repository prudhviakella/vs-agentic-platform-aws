FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first — cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent package into /app/agent/ so Python resolves:
#   from agent.graph import build_agent        → /app/agent/graph.py
#   from agent.middleware.pii import ...       → /app/agent/middleware/pii.py
#   from agent.core.agent import MiddlewareAgent → /app/agent/core/agent.py
# Everything is self-contained — no files copied from other repos.
COPY . ./agent/

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8080/ping || exit 1

# -m flag: Python's sys.path includes /app/ so `from agent.*` imports work
CMD ["python", "-m", "agent.agent"]
