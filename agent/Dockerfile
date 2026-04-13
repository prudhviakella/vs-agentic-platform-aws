FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Copy shared middleware from local packages
# These are copied from vs-agentic-platform before docker build
# See scripts/deploy.sh for the copy commands
COPY vs-agent-core/core ./core
COPY clinical_trial_agent/agent/middleware ./agent/middleware

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8080/ping || exit 1

CMD ["python", "agent.py"]
