# VS AgentCore Platform — Clinical Trial Research Agent

> **Vidya Sankalp · Applied GenAI Engineering**
> A production multi-agent AI platform built on AWS Bedrock AgentCore for clinical trial research.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment — Step by Step](#deployment--step-by-step)
- [Verify Deployment](#verify-deployment)
- [Test the Agent](#test-the-agent)
- [Daily Development Workflow](#daily-development-workflow)
- [Observability](#observability)
- [Troubleshooting](#troubleshooting)
- [Tear Down](#tear-down)

---

## Overview

This platform lets researchers, clinicians, and pharma professionals search 5,772 clinical trial documents and a live biomedical knowledge graph using natural language. The agent autonomously selects tools, asks clarifying questions when needed (HITL), synthesises evidence across multiple sources, and cites every claim.

**Key capabilities:**
- Semantic search over clinical trial chunks (Pinecone)
- Structured graph queries — trial → drug, disease, sponsor relationships (Neo4j)
- Human-in-the-Loop clarification with real trial names as options
- Per-query observability — tokens, cost, latency, guardrail scores (DynamoDB)
- Episodic memory across sessions (Pinecone namespaces)
- Semantic cache to avoid re-running identical queries

---

## Architecture

![VS AgentCore Platform Architecture](docs/architecture_final.png)

> Place `architecture_final.png` from the repo root into `docs/architecture.png`.

---

### Request Flow

```
User
 └─► ALB (port 80)
      ├─► FastAPI UI Service    (ECS Fargate · port 8501)
      └─► Platform API Service  (ECS Fargate · port 8000)
               └─► AgentCore Runtime  (arm64 · LangGraph · ECR)
                        └─► AgentCore MCP Gateway
                                 ├─► Lambda: search_tool    ──► Pinecone (vector search)
                                 ├─► Lambda: graph_tool     ──► Neo4j AuraDB (Cypher)
                                 ├─► Lambda: clarify (HITL) ──► RDS PostgreSQL (HITL state)
                                 └─► Lambda: summariser_tool
                        ├─► OpenAI GPT-4o        (LLM reasoning + tool calling)
                        ├─► OpenAI Embeddings    (semantic cache + episodic memory)
                        ├─► RDS PostgreSQL       (LangGraph checkpointer)
                        ├─► DynamoDB             (observability traces)
                        ├─► Secrets Manager      (API keys at cold start)
                        ├─► SSM Parameter Store  (prompt ID/version, gateway URL)
                        └─► Bedrock Prompt Mgmt  (versioned system prompt)
```

---

### Component Deep Dive

#### 1. FastAPI UI Service (`ui/`)
A single-page web application served by FastAPI + uvicorn running on ECS Fargate (amd64, port 8501). It has zero JavaScript framework dependencies — the entire frontend is vanilla HTML + CSS + JS served inline from `app.py`.

Key features:
- **Real-time token streaming** — SSE proxy to Platform API, tokens appear character by character with a blinking cursor
- **Tool step indicators** — animated spinner while each tool runs, green ✓ when done
- **HITL card** — renders interrupt events as a numbered option card; clicking an option resumes the agent
- **Inline markdown renderer** — parses `**bold**`, `## headers`, `- lists`, code blocks without any CDN dependency (works in air-gapped ECS network)
- **Apple Gothic font stack** — `'AppleGothic', 'Apple Gothic', 'Gill Sans MT', 'Century Gothic'` — works natively on macOS, falls back gracefully on Linux
- **UUID fallback** — `crypto.randomUUID()` only works on HTTPS; the UI provides a `Math.random()` fallback so sessions work over HTTP (ALB without TLS)
- **Session management** — each `+ New` generates a fresh UUID, which creates a new LangGraph thread in Postgres and prevents stale HITL state from bleeding across conversations

#### 2. Platform API Service (`platform/`)
FastAPI application running on ECS Fargate (amd64, port 8000). Routes:
- `POST /api/v1/clinical-trial/chat` — new query, creates LangGraph thread
- `POST /api/v1/clinical-trial/resume` — resumes after HITL interrupt with `user_answer`
- `GET /health` — ALB health check

The Platform API calls `InvokeAgentRuntime` on the AgentCore Runtime and proxies the SSE stream back to the UI. It reads the runtime ARN from SSM at startup so redeploying the agent without touching the platform automatically picks up the new ARN.

#### 3. AgentCore Runtime (`agent/`)
The core of the platform. An arm64 Docker container (AgentCore only supports arm64) built with:
- **LangGraph** — stateful multi-step agent graph with `PostgresSaver` checkpointer
- **LangChain 1.0** — tool binding, message handling, LLM abstraction
- **OpenAI GPT-4o** — primary LLM for reasoning and tool calling
- **9-layer middleware stack** — wraps every request before and after the LangGraph graph

At cold start, the container reads from:
- `Secrets Manager` — OpenAI API key, Pinecone API key, Neo4j credentials, Postgres credentials
- `SSM Parameter Store` — MCP Gateway URL, Bedrock prompt ID and version, DynamoDB table name

The agent runs as a streaming SSE server. AgentCore proxies the SSE stream to callers. Events emitted: `tool_start`, `tool_end`, `interrupt`, `token`, `done`, `error`.

**LangGraph State Machine:**

```
START
  │
  ▼
call_model (GPT-4o decides: answer or call tool)
  │
  ├─► [tool call] ──► call_tools ──► [interrupt if HITL] ──► back to call_model
  │
  └─► [no tool call] ──► END
```

`recursion_limit: 50` prevents infinite loops. The graph uses `interrupt_on` to pause on `clarify___ask_user_input` tool calls and surface them to the Platform API as interrupt events.

#### 4. AgentCore MCP Gateway
The MCP (Model Context Protocol) Gateway acts as a managed proxy between the AgentCore Runtime and the Lambda tools. The LangGraph agent connects to the gateway using `streamable_http` transport and discovers tools via the MCP `list_tools` handshake at cold start.

Tool naming is critical — the gateway target name becomes a prefix in the tool name the LLM sees:
- Target `clarify` → LLM sees `clarify___ask_user_input` (natural English → GPT-4o calls reliably)
- Target `tool-hitl` → LLM sees `tool-hitl___ask_user_input` (too technical → GPT-4o ignored it)

#### 5. Lambda Tools (`mcp_tools/`)

**`search_tool`** (Pinecone)
Embeds the query using OpenAI embeddings and queries the Pinecone `clinical-agent` index. Returns top-k chunks with relevance scores. The index has 5,772 chunks from clinical trial documents across multiple namespaces:
- `clinical-trials-index` — document chunks
- `cache_<domain>` — semantic cache entries (TTL: 1 hour, threshold: 0.97 cosine similarity)
- `episodic__<session_id>` — per-session memory entries

**`graph_tool`** (Neo4j)
Executes read-only Cypher queries against Neo4j AuraDB. The knowledge graph schema:
```
(Trial)-[:TARGETS]->(Disease)
(Trial)-[:USES]->(Drug)
(Trial)-[:SPONSORED_BY]->(Sponsor)
(Trial)-[:CONDUCTED_IN]->(Country)
(Trial)-[:MEASURES]->(Outcome)
(Trial)-[:INCLUDES]->(PatientPopulation)
```
Best used for structured discovery queries: "what trials exist", "who sponsors trial X", "what drugs are used for disease Y".

**`clarify` / HITL tool** (Human-in-the-Loop)
When called, this tool signals an interrupt to LangGraph via `NodeInterrupt`. The Platform API detects the interrupt event in the SSE stream and converts it into an `interrupt` event for the UI. The UI renders a HITL card with numbered options. When the user selects an option or types a custom answer, the Platform API calls `/resume` which injects the answer into the LangGraph checkpoint and resumes execution from the interrupted node.

The HITL state is persisted in RDS PostgreSQL between the `/chat` call and the `/resume` call — the agent is literally paused mid-graph, and the Postgres checkpoint stores the entire graph state including pending tool calls.

**`summariser_tool`** (GPT-4o)
Takes 3+ retrieved chunks plus the original query and produces a single synthesised answer with citations. Called as the final step after evidence gathering. Avoids the LLM trying to answer directly from context window — forces explicit synthesis.

#### 6. Data Layer

**Pinecone** — three namespace patterns in one index:
- `clinical-trials-index` — 5,772 document chunks, queried by `search_tool`
- `cache_pharma` — semantic cache, keyed by query embedding, TTL 1 hour
- `episodic__<session_id>` — per-session episodic memories, stored when `EPISODIC: YES` tag detected

**Neo4j AuraDB** — biomedical knowledge graph with ~50K nodes and ~200K relationships. Enables structured queries that Pinecone can't answer: "list all trials", "who sponsors this trial", "what drugs target this disease".

**RDS PostgreSQL** — LangGraph `PostgresSaver` checkpointer. Stores the complete LangGraph state (messages, pending tool calls, intermediate results) keyed by `thread_id`. This is what enables HITL: the agent pauses, the checkpoint is saved, the user answers hours later, and the graph resumes from exactly where it left off.

**DynamoDB `vs-agentcore-traces`** — structured observability. Every completed run writes one record with 20+ fields. TTL set to 30 days. Used for monitoring token costs, guardrail pass rates, cache hit rates, and HITL question quality.

#### 7. Middleware Stack (9 Layers)

Every request passes through 9 middleware layers, executed in order. Each layer has `before_agent` and `after_agent` hooks.

| # | Middleware | When it fires | What it does |
|---|---|---|---|
| 1 | `ContentFilterMiddleware` | before_agent | Blocks off-topic/harmful queries before any LLM call |
| 2 | `SemanticCacheMiddleware` | before_agent | Embeds query, checks Pinecone cache. If hit (cosine ≥ 0.97) → returns cached answer immediately, skips LLM entirely |
| 3 | `EpisodicMemoryMiddleware` | before_agent | Searches `episodic__<session_id>` namespace, injects relevant past facts into system prompt |
| 4 | `HumanInTheLoopMiddleware` | during_agent | Intercepts `clarify___ask_user_input` tool calls, pauses LangGraph graph, surfaces interrupt event |
| 5 | `SummarizationMiddleware` | before_model | Compresses conversation history when token count exceeds 8,000 to prevent context overflow |
| 6 | `TracerMiddleware` | before + after | Records request start, collects tool details and LLM timings, writes trace to DynamoDB on completion |
| 7 | `OutputGuardrailMiddleware` | after_agent | Layer 1: faithfulness check — GPT-4o-mini verifies answer is grounded in tool results (score ≥ 0.7) |
| 8 | `OutputGuardrailMiddleware` | after_agent | Layer 2: consistency check — verifies answer is internally consistent |
| 9 | `OutputGuardrailMiddleware` | after_agent | Layer 3: blocks answer if guardrail failed, returns safe refusal message |

The middleware layers communicate through `TracerMiddleware.update_trace(run_id, {...})` — a class-level registry that allows any middleware to annotate the trace record without direct coupling.

#### 8. Bedrock Prompt Management
The system prompt is versioned in Amazon Bedrock Prompt Management (prompt ID: `YEVDY4MYU6`). The current version is read from SSM at cold start:
```
/clinical-trial-agent/prod/bedrock/prompt_id      → YEVDY4MYU6
/clinical-trial-agent/prod/bedrock/prompt_version → 14
```
Updating the prompt without redeploying the agent: run `./scripts/deploy.sh prompt` (creates new Bedrock version, updates SSM), then `./scripts/deploy.sh agent` (agent cold-starts and reads new version from SSM).

#### 9. Config & Secrets Strategy
At cold start, the agent reads from two services:
- **Secrets Manager** — sensitive values (API keys, database passwords). Loaded once at startup, cached in memory for the container lifetime.
- **SSM Parameter Store** — non-sensitive config (prompt ID/version, gateway URL, DynamoDB table name, Pinecone index names). Same pattern.

This means zero secrets in environment variables, zero secrets in Docker images, and zero secrets in Terraform state.

---

### Middleware Stack Summary

```
Request arrives
     │
     ▼
[1] ContentFilter     — block if harmful
     │
     ▼
[2] SemanticCache     — return if cache hit (skip LLM entirely)
     │
     ▼
[3] EpisodicMemory    — inject past session facts
     │
     ▼
[4] HITL              — monitor for clarify tool calls
     │
     ▼
[5] Summarization     — compress history if > 8,000 tokens
     │
     ▼
[6] Tracer            — start timing, collect metadata
     │
     ▼
   LangGraph Graph    — GPT-4o reasons, calls tools, generates answer
     │
     ▼
[7-9] OutputGuardrail — faithfulness + consistency check
     │
     ▼
   Tracer (after)     — write 20+ field trace to DynamoDB
     │
     ▼
Response streams to user (SSE token events)
```

---

### Observability Schema (DynamoDB `vs-agentcore-traces`)

Every query writes one record with 20+ fields:

```
run_id            — UUID, primary key
session_id        — same as run_id (one session per query)
user_id           — session identifier
domain            — "pharma"
question          — original user query
answer            — final agent answer (full text)
prompt_version    — Bedrock prompt version used
llm_turns         — number of LLM calls in this run
tool_count        — total tool invocations
tools[]           — list of tool names called
tool_details[]    — [{name, args_summary, result_summary, is_error}]
llm_timings[]     — [{turn, elapsed_ms, input_tokens, output_tokens}]
input_tokens      — total input tokens
output_tokens     — total output tokens
total_tokens      — combined
token_cost_usd    — cost in USD
elapsed_ms        — total wall-clock latency
faithfulness_score  — 0.0–1.0 (guardrail layer 2)
consistency_score   — 0.0–1.0 (guardrail layer 3)
hitl_fired        — bool
hitl_question     — clarification question shown to user
hitl_options[]    — options presented
hitl_user_answer  — what the user selected/typed
cache_hit         — bool (SemanticCacheMiddleware)
episodic_hits     — count of memories injected
episodic_stored   — bool (was this answer stored as memory)
is_resume         — bool (was this a HITL resume call)
guardrail_passed  — bool
guardrail_blocked — bool
errors[]          — any middleware errors
has_errors        — bool
expires_at        — TTL timestamp (30 days)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Runtime | AWS Bedrock AgentCore (arm64) |
| Agent Framework | LangGraph + LangChain 1.0 |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-ada-002 |
| System Prompt | AWS Bedrock Prompt Management |
| Tool Protocol | MCP (Model Context Protocol) |
| Vector Store | Pinecone (search + cache + episodic memory) |
| Knowledge Graph | Neo4j AuraDB |
| Conversation State | RDS PostgreSQL (LangGraph PostgresSaver) |
| Observability | DynamoDB + CloudWatch Logs |
| Platform API | FastAPI + uvicorn (ECS Fargate amd64) |
| UI | FastAPI + vanilla HTML (ECS Fargate amd64) |
| Infrastructure | Terraform + AWS (VPC, ALB, ECS, RDS, DynamoDB) |
| Container Registry | Amazon ECR |
| Secrets | AWS Secrets Manager + SSM Parameter Store |

---

## Project Structure

```
vs-agentcore-platform-aws/
├── agent/                        # AgentCore Runtime (arm64)
│   ├── app.py                    # LangGraph agent, middleware stack, SSE streaming
│   ├── agent/
│   │   ├── agent.py              # LangGraph graph builder
│   │   └── tools/mcp_client.py   # MCP Gateway client
│   ├── core/
│   │   └── middleware/
│   │       ├── tracer.py         # TracerMiddleware — 20+ field DynamoDB traces
│   │       ├── semantic_cache.py # SemanticCacheMiddleware
│   │       ├── episodic_memory.py# EpisodicMemoryMiddleware
│   │       └── output_guardrail.py# Faithfulness + consistency guardrails
│   ├── Dockerfile                # linux/arm64
│   └── requirements.txt
│
├── platform/                     # Platform API (amd64)
│   ├── app.py                    # FastAPI — /api/v1/clinical-trial/chat + resume
│   └── Dockerfile
│
├── ui/                           # FastAPI UI (amd64)
│   ├── app.py                    # Dark clinical UI, SSE proxy, HITL card
│   ├── Dockerfile
│   └── requirements.txt
│
├── mcp_tools/                    # Lambda MCP tools (amd64)
│   ├── search_lambda/            # Pinecone semantic search
│   ├── graph_lambda/             # Neo4j Cypher queries
│   ├── hitl_lambda/              # HITL interrupt handler
│   └── summariser_lambda/        # GPT-4o synthesis
│
├── infra/                        # Terraform
│   ├── main.tf                   # VPC, ECS, ALB, RDS, DynamoDB
│   └── variables.tf
│
├── prompts/
│   └── system_prompt.txt         # Bedrock system prompt — deploy with: ./scripts/deploy.sh prompt
│
├── scripts/
│   └── deploy.sh                 # One-click deployment script
│
├── docs/
│   └── architecture.png          # Architecture diagram
│
└── .env.prod                     # Environment variables (not committed)
```

---

## Prerequisites

| Tool | Install | Verify |
|---|---|---|
| AWS CLI v2 | https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html | `aws --version` |
| Docker Desktop | https://www.docker.com/products/docker-desktop | `docker --version` |
| Terraform ≥ 1.5 | https://developer.hashicorp.com/terraform/install | `terraform --version` |
| Python 3.9+ | https://www.python.org/downloads | `python3 --version` |
| boto3 | `pip install boto3` | `python3 -c "import boto3"` |

Configure AWS credentials:
```bash
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region:        us-east-1
# Default output format: json

# Verify
aws sts get-caller-identity
```

---

## Quick Start

```bash
# Clone
git clone <repo-url>
cd vs-agentcore-platform-aws

# Configure
cp .env.example .env.prod
# Edit .env.prod — fill all values (see below)
source .env.prod

# Phase 1 — infrastructure (Steps 0–5)
./scripts/deploy.sh all

# ⚠️ PAUSE: fill POSTGRES_URL in .env.prod with RDS endpoint from terraform output
# POSTGRES_URL=postgresql://postgres:<pwd>@<rds-endpoint>/clinical_agent
source .env.prod
./scripts/deploy.sh secrets

# Phase 2 — agent
./scripts/deploy.sh agent
```

---

## Deployment — Step by Step

### Configure `.env.prod`

```bash
# ── OpenAI ────────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...

# ── Pinecone ──────────────────────────────────────────────────────────────
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=clinical-agent
CLINICAL_TRIALS_INDEX=clinical-trials-index

# ── Neo4j AuraDB ──────────────────────────────────────────────────────────
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# ── Platform ──────────────────────────────────────────────────────────────
PLATFORM_API_KEY=vs-platform-key-change-me   # any strong random string

# ── RDS — leave blank, fill after Step 5 ─────────────────────────────────
POSTGRES_URL=

# ── RDS master password ───────────────────────────────────────────────────
RDS_PASSWORD=YourStrongPassword123!

# ── AWS ───────────────────────────────────────────────────────────────────
AWS_REGION=us-east-1
```

```bash
source .env.prod
```

---

### Why Two Phases?

> Step 5 (Terraform) creates the RDS PostgreSQL instance.
> The agent container reads the Postgres endpoint at cold start.
> RDS must exist before the agent can be deployed.

---

### Phase 1 — Infrastructure + Platform

> Each step is idempotent — safe to re-run.
> `./scripts/deploy.sh all` runs Steps 0–5 then pauses for the manual POSTGRES_URL step.

**Step 0 — Bedrock Prompt**
```bash
./scripts/deploy.sh prompt
```
Creates the system prompt in Amazon Bedrock Prompt Management. Writes `prompt_id` and `prompt_version` to SSM Parameter Store. Re-run whenever you update `prompts/system_prompt.txt`.

**Step 1 — Secrets**
```bash
./scripts/deploy.sh secrets
```
Writes all API keys from `.env.prod` to AWS Secrets Manager and SSM Parameter Store.
> Note: Postgres secret is skipped if `POSTGRES_URL` is empty — re-run after Step 5.

**Step 2 — IAM Roles**
```bash
./scripts/deploy.sh iam
```
Creates:
- `vs-agentcore-lambda-mcp` — Lambda execution role (Secrets Manager + SSM)
- `vs-agentcore-gateway-role` — MCP Gateway role (Lambda invoke)
- `vs-agentcore-agent-role` — AgentCore role (ECR pull, DynamoDB, Bedrock, CloudWatch)

**Step 3 — Lambda Tools**
```bash
./scripts/deploy.sh lambdas
```
Builds 4 Lambda container images (linux/amd64) and deploys:
- `vs-agentcore-search-tool` → Pinecone semantic search
- `vs-agentcore-graph-tool` → Neo4j Cypher queries
- `vs-agentcore-hitl-tool` → HITL interrupt handler
- `vs-agentcore-summariser-tool` → GPT-4o synthesis

Each Lambda is test-invoked after deployment. Takes ~5 minutes.

**Step 4 — MCP Gateway**
```bash
./scripts/deploy.sh gateway
```
Creates the Bedrock AgentCore MCP Gateway `vs-agentcore-mcp` and registers 4 tool targets. Skipped automatically if the gateway already exists.

Tool target names (what the LLM sees):
- `tool-search___search_tool`
- `tool-graph___graph_tool`
- `clarify___ask_user_input` ← natural English so GPT-4o calls it reliably
- `tool-summariser___summariser_tool`

**Step 5 — Platform + UI (Terraform)**
```bash
./scripts/deploy.sh platform
```
Runs `terraform apply` — creates:
- VPC (public + private subnets, 2 AZs)
- Application Load Balancer
- ECS Fargate cluster + 2 services (platform API + UI)
- RDS PostgreSQL db.t3.micro (free tier: 750 hrs/month)
- DynamoDB table `vs-agentcore-traces` (PAY_PER_REQUEST + TTL)
- CloudWatch log groups
- S3 bucket for Terraform state

Takes ~10 minutes. Output:
```
ALB DNS:      http://vs-agentcore-alb-xxxxxxxxxx.us-east-1.elb.amazonaws.com
RDS endpoint: vs-agentcore-postgres.xxxx.us-east-1.rds.amazonaws.com:5432
```

---

### Between Phases — Fill Postgres URL

```bash
# Edit .env.prod
POSTGRES_URL=postgresql://postgres:YourStrongPassword123!@vs-agentcore-postgres.xxxx.us-east-1.rds.amazonaws.com/clinical_agent

# Push the postgres secret
source .env.prod
./scripts/deploy.sh secrets
```

---

### Phase 2 — Agent

**Step 6 — AgentCore Runtime**
```bash
./scripts/deploy.sh agent
```
Builds the agent container (linux/arm64 — AgentCore requires arm64), pushes to ECR, and creates/updates the AgentCore Runtime. Waits for READY status (~2 minutes).

---

## Verify Deployment

```bash
# AgentCore Runtime status
aws bedrock-agentcore-control get-agent-runtime \
  --region us-east-1 \
  --agent-runtime-id $(aws ssm get-parameter \
    --name "/vs-agentcore/prod/agent_runtime_arn" \
    --query "Parameter.Value" --output text | sed 's/.*runtime\///') \
  --query "{status:status,version:agentRuntimeVersion}" \
  --output json
# Expected: {"status": "READY", "version": "1"}

# ECS services
aws ecs describe-services \
  --cluster vs-agentcore-cluster \
  --services vs-agentcore-platform vs-agentcore-ui \
  --region us-east-1 \
  --query "services[].{name:serviceName,running:runningCount,desired:desiredCount}" \
  --output table
# Expected: running=1 desired=1 for both services

# Open the UI
echo "http://$(cd infra && terraform output -raw alb_dns)"
```

---

## Test the Agent

```bash
RUNTIME_ARN=$(aws ssm get-parameter \
  --name "/vs-agentcore/prod/agent_runtime_arn" \
  --region us-east-1 --query "Parameter.Value" --output text)

SESSION_ID=$(python3 -c "import uuid; print(str(uuid.uuid4()))")

# Simple query — should search and answer directly
PAYLOAD=$(echo -n "{\"message\":\"What are the Phase 3 results for Pfizer BNT162b2?\",\"thread_id\":\"${SESSION_ID}\",\"domain\":\"pharma\",\"resume\":false}" | base64)

aws bedrock-agentcore invoke-agent-runtime \
  --region us-east-1 \
  --agent-runtime-arn "$RUNTIME_ARN" \
  --runtime-session-id "$SESSION_ID" \
  --payload "$PAYLOAD" \
  /tmp/test.json && cat /tmp/test.json

# Expected: tool_start → tool_end → tokens → done
```

HITL flow test:
```bash
SESSION_ID=$(python3 -c "import uuid; print(str(uuid.uuid4()))")
PAYLOAD=$(echo -n "{\"message\":\"search for cancer trials\",\"thread_id\":\"${SESSION_ID}\",\"domain\":\"pharma\",\"resume\":false}" | base64)

aws bedrock-agentcore invoke-agent-runtime \
  --region us-east-1 --agent-runtime-arn "$RUNTIME_ARN" \
  --runtime-session-id "$SESSION_ID" --payload "$PAYLOAD" \
  /tmp/t1.json && cat /tmp/t1.json
# Expected: tool_start → tool_end → interrupt (with real trial names as options)

# Resume with selected option
RESUME=$(echo -n "{\"message\":\"\",\"thread_id\":\"${SESSION_ID}\",\"domain\":\"pharma\",\"resume\":true,\"user_answer\":\"NCI-MATCH molecular profiling trial\"}" | base64)

aws bedrock-agentcore invoke-agent-runtime \
  --region us-east-1 --agent-runtime-arn "$RUNTIME_ARN" \
  --runtime-session-id "$SESSION_ID" --payload "$RESUME" \
  /tmp/t2.json && cat /tmp/t2.json
# Expected: tool_start → tool_end → tokens → done (with citations)
```

---

## Local Development & Testing

Everything can be run and tested locally before deploying to AWS. This is the fastest way to iterate on agent logic, prompts, Lambda tools, and the UI.

---

### Prerequisites for Local Dev

```bash
pip install -r agent/requirements.txt
pip install -r platform/requirements.txt
pip install -r ui/requirements.txt
```

Create a local `.env.local` file (copy from `.env.prod` but use local values):
```bash
cp .env.prod .env.local
source .env.local
```

---

### 1. Run the Agent Locally (no AgentCore, no Docker)

The LangGraph agent runs directly as a Python process — no AWS AgentCore needed:

```bash
cd agent
source ../.env.local

# Run a single query directly
python3 - << 'EOF'
import asyncio
from app import create_agent

async def main():
    agent = await create_agent()
    session_id = "local-test-001"
    async for event in agent.astream({
        "message": "What are the Phase 3 results for Pfizer BNT162b2?",
        "thread_id": session_id,
        "domain": "pharma",
        "resume": False
    }):
        print(event)

asyncio.run(main())
EOF
```

---

### 2. Run the Platform API Locally

```bash
cd platform
source ../.env.local

# Point to local agent instead of AgentCore Runtime
export AGENT_MODE=local   # if supported by your platform/app.py

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Test it:
```bash
SESSION_ID=$(python3 -c "import uuid; print(str(uuid.uuid4()))")

curl -X POST http://localhost:8000/api/v1/clinical-trial/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PLATFORM_API_KEY}" \
  -d "{\"message\": \"What trials exist for semaglutide?\", \"thread_id\": \"${SESSION_ID}\", \"domain\": \"pharma\"}"
```

---

### 3. Run the UI Locally

```bash
cd ui
source ../.env.local

export AGENT_API_URL=http://localhost:8000
export AGENT_DOMAIN=pharma
export AGENT_API_KEY=${PLATFORM_API_KEY}

uvicorn app:app --host 0.0.0.0 --port 8501 --reload
```

Open: http://localhost:8501

---

### 4. Test Lambda Tools Locally

Each Lambda tool is a standalone Python function — invoke the handler directly:

```bash
# Test search_tool
cd mcp_tools/search_lambda
source ../../.env.local

python3 - << 'EOF'
from handler import handler
result = handler({"query": "semaglutide Phase 3 results", "top_k": 3}, {})
import json; print(json.dumps(result, indent=2))
EOF
```

```bash
# Test graph_tool
cd mcp_tools/graph_lambda
source ../../.env.local

python3 - << 'EOF'
from handler import handler
result = handler({"cypher": "MATCH (t:Trial) RETURN t.nctId, t.briefTitle LIMIT 5"}, {})
import json; print(json.dumps(result, indent=2))
EOF
```

```bash
# Test summariser_tool
cd mcp_tools/summariser_lambda
source ../../.env.local

python3 - << 'EOF'
from handler import handler
result = handler({
    "chunks": [
        "The BNT162b2 vaccine showed 95% efficacy in preventing COVID-19.",
        "The trial enrolled 43,548 participants across multiple sites."
    ],
    "query": "What were the efficacy results?"
}, {})
import json; print(json.dumps(result, indent=2))
EOF
```

---

### 5. Run with Docker Compose (full local stack)

Run the entire platform locally with Docker Compose — no AWS needed except for Pinecone, Neo4j, and OpenAI:

```bash
# Build all images locally
docker compose build

# Start platform API + UI
docker compose up

# UI:      http://localhost:8501
# API:     http://localhost:8000
# Health:  http://localhost:8000/health
```

`docker-compose.yml` example:
```yaml
version: "3.9"
services:
  platform:
    build: ./platform
    ports: ["8000:8000"]
    env_file: .env.local
    environment:
      - AGENT_MODE=local

  ui:
    build: ./ui
    ports: ["8501:8501"]
    env_file: .env.local
    environment:
      - AGENT_API_URL=http://platform:8000
      - AGENT_DOMAIN=pharma
    depends_on: [platform]
```

---

### 6. Test Individual Middleware Layers

Test each middleware layer in isolation:

```bash
cd agent
source ../.env.local

# Test SemanticCacheMiddleware
python3 - << 'EOF'
import asyncio
from core.middleware.semantic_cache import SemanticCacheMiddleware

async def test():
    mw = SemanticCacheMiddleware()
    result = await mw.check_cache(
        query="What are the Phase 3 results for BNT162b2?",
        domain="pharma"
    )
    print(f"Cache hit: {result is not None}")
    print(result)

asyncio.run(test())
EOF
```

```bash
# Test TracerMiddleware — verify DynamoDB write
python3 - << 'EOF'
import asyncio, uuid
from core.middleware.tracer import TracerMiddleware

async def test():
    run_id = str(uuid.uuid4())
    tm = TracerMiddleware()
    await tm.init_trace(run_id, {
        "question": "test query",
        "domain": "pharma",
        "session_id": run_id
    })
    await tm.finalize_trace(run_id, {
        "answer": "test answer",
        "total_tokens": 100,
        "token_cost_usd": 0.001,
        "elapsed_ms": 1500
    })
    print(f"Trace written: {run_id}")
    # Check DynamoDB
    import boto3
    table = boto3.resource("dynamodb", region_name="us-east-1").Table("vs-agentcore-traces")
    item = table.get_item(Key={"run_id": run_id}).get("Item")
    print(f"DynamoDB record: {item is not None}")

asyncio.run(test())
EOF
```

---

### 7. Test Neo4j Connection

```bash
python3 - << 'EOF'
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
)
with driver.session() as session:
    result = session.run("MATCH (t:Trial) RETURN t.nctId, t.briefTitle LIMIT 5")
    for r in result:
        print(r["t.nctId"], "-", r["t.briefTitle"])
driver.close()
EOF
```

---

### 8. Test Pinecone Connection

```bash
python3 - << 'EOF'
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["CLINICAL_TRIALS_INDEX"])

stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")
print(f"Namespaces: {list(stats.namespaces.keys())}")

# Test a query
results = index.query(
    vector=[0.0] * 1536,   # dummy vector — replace with real embedding
    top_k=3,
    include_metadata=True
)
print(f"Query returned {len(results.matches)} results")
EOF
```

---

### Local Testing Checklist

Before deploying to AWS, verify locally:

| Test | Command | Expected |
|---|---|---|
| Neo4j connection | `python3` + graph query above | 5 trial rows returned |
| Pinecone connection | `python3` + stats above | total_vector_count > 0 |
| search_tool | handler test above | results[] with score and text |
| graph_tool | handler test above | results[] with nctId and briefTitle |
| summariser_tool | handler test above | summary string returned |
| Platform API health | `curl localhost:8000/health` | `{"status": "ok"}` |
| UI loads | open localhost:8501 | Dark clinical UI appears |
| Full query | curl /chat endpoint | SSE events stream |
| HITL flow | vague query + resume | interrupt event + final answer |

---

## Daily Development Workflow

| What changed | Command |
|---|---|
| Agent code (`agent/`) | `./scripts/deploy.sh agent` |
| System prompt (`prompts/system_prompt.txt`) | `./scripts/deploy.sh prompt` then `./scripts/deploy.sh agent` |
| UI code (`ui/`) | Build + push, then `./scripts/deploy.sh redeploy ui` |
| Platform API (`platform/`) | Build + push, then `./scripts/deploy.sh redeploy platform` |
| Lambda tools (`mcp_tools/`) | `./scripts/deploy.sh lambdas` |
| Secrets / API keys | `./scripts/deploy.sh secrets` |

**Quick UI rebuild and redeploy:**
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_TAG="${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/vs-agentcore/ui:latest"

aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com"

docker buildx build --platform linux/amd64 --output type=registry \
  --provenance=false --no-cache -t "$ECR_TAG" ./ui

./scripts/deploy.sh redeploy ui
```

**Available deploy commands:**
```bash
./scripts/deploy.sh prompt     # Step 0: create/version Bedrock prompt → SSM
./scripts/deploy.sh secrets    # Step 1: push secrets + SSM params
./scripts/deploy.sh iam        # Step 2: create IAM roles
./scripts/deploy.sh lambdas    # Step 3: build + deploy Lambda tools
./scripts/deploy.sh gateway    # Step 4: create MCP Gateway + targets
./scripts/deploy.sh platform   # Step 5: Terraform (ECS, ALB, RDS)
./scripts/deploy.sh agent      # Step 6: build + deploy AgentCore Runtime
./scripts/deploy.sh all        # Steps 0–5 then pauses for POSTGRES_URL
./scripts/deploy.sh redeploy [platform|ui|both]  # Quick ECS force-redeploy
./scripts/deploy.sh plan       # Terraform plan (no changes)
./scripts/deploy.sh destroy    # Tear down all Terraform resources
```

---

## Observability

**Latest traces summary:**
```bash
aws dynamodb scan \
  --table-name vs-agentcore-traces \
  --region us-east-1 \
  --max-items 10 \
  --output json | python3 -c "
import sys, json
data = json.load(sys.stdin)
items = sorted(data['Items'], key=lambda x: x.get('ts',{}).get('N','0'), reverse=True)
print(f\"{'run_id':<38} {'tokens':>8} {'cost':>10} {'ms':>8} {'hitl':>6} {'cache':>6}\")
print('-' * 85)
for item in items:
    d = {k: list(v.values())[0] for k, v in item.items()}
    print(f\"{str(d.get('run_id',''))[:36]:<38} {str(d.get('total_tokens','0')):>8} \${str(d.get('token_cost_usd','0')):>9} {str(d.get('elapsed_ms','0')):>8} {str(d.get('hitl_fired','?')):>6} {str(d.get('cache_hit','?')):>6}\")
"
```

**Total cost across all traces:**
```bash
aws dynamodb scan --table-name vs-agentcore-traces --region us-east-1 \
  --output json | python3 -c "
import sys, json
data = json.load(sys.stdin)
items = data['Items']
total_cost   = sum(float(i.get('token_cost_usd',{}).get('N',0)) for i in items)
total_tokens = sum(int(i.get('total_tokens',{}).get('N',0)) for i in items)
print(f'Traces:       {len(items)}')
print(f'Total tokens: {total_tokens:,}')
print(f'Total cost:   \${total_cost:.4f}')
print(f'Avg per query: \${total_cost/max(len(items),1):.4f}')
"
```

**Agent CloudWatch logs:**
```bash
aws logs get-log-events \
  --region us-east-1 \
  --log-group-name "/aws/bedrock-agentcore/runtimes/vs_agentcore_clinical_trial-<id>-DEFAULT" \
  --log-stream-name "runtime-logs" \
  --limit 50 \
  --query "events[-50:].message" \
  --output text | grep -E "TRACER|ERROR|HITL|CACHE|EPISODIC"
```

---

## Architecture Notes

| Component | Platform | Reason |
|---|---|---|
| AgentCore Runtime | linux/arm64 | AgentCore only supports arm64 |
| Lambda tools | linux/amd64 | Lambda x86_64 standard |
| ECS Platform + UI | linux/amd64 | Fargate standard |
| All Docker builds | `--output type=registry` | Pushes directly to ECR — avoids arm64/amd64 contamination on Apple Silicon Macs |
| RDS | db.t3.micro | AWS free tier (750 hrs/month, 20 GB) |
| DynamoDB | PAY_PER_REQUEST | No provisioned capacity needed for course usage |
| MCP target "clarify" | `clarify___ask_user_input` | Natural English name — GPT-4o calls it reliably vs `tool-hitl___ask_user_input` which was ignored |

---

## Troubleshooting

**Agent returns 502**
```bash
# Check runtime is READY
aws bedrock-agentcore-control get-agent-runtime \
  --region us-east-1 --agent-runtime-id <runtime-id> \
  --query "status" --output text

# Check agent logs
aws logs get-log-events \
  --region us-east-1 \
  --log-group-name "/aws/bedrock-agentcore/runtimes/vs_agentcore_clinical_trial-<id>-DEFAULT" \
  --log-stream-name "runtime-logs" --limit 30 \
  --query "events[-30:].message" --output text | tail -20

# Common cause: RDS not reachable — verify POSTGRES_URL was pushed to Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id "/vs-agentcore/prod/postgres" \
  --region us-east-1 --query SecretString --output text
```

**DynamoDB traces not writing**
```bash
# Verify table exists
aws dynamodb describe-table --table-name vs-agentcore-traces --region us-east-1

# Check agent role has DynamoDB permissions
aws iam get-role-policy \
  --role-name vs-agentcore-agent-role \
  --policy-name AgentCorePolicy \
  --query "PolicyDocument.Statement[?Sid=='DynamoDB']"
```

**Lambda tool errors**
```bash
aws lambda invoke \
  --function-name vs-agentcore-search-tool \
  --payload '{"query":"cancer trial","top_k":3}' \
  --cli-binary-format raw-in-base64-out \
  --region us-east-1 /tmp/out.json && cat /tmp/out.json
```

**ECS service not healthy**
```bash
# Check service events
aws ecs describe-services \
  --cluster vs-agentcore-cluster \
  --services vs-agentcore-ui \
  --region us-east-1 \
  --query "services[0].events[:5].message" --output text

# Check task logs
aws logs tail /ecs/vs-agentcore/ui --since 5m
```

---

## Tear Down

```bash
./scripts/deploy.sh destroy
```

Destroys all Terraform-managed resources: ECS cluster, ALB, RDS, DynamoDB table, VPC, subnets, security groups, CloudWatch log groups, S3 state bucket.

**Manual cleanup** (not managed by Terraform):
- Lambda functions → AWS Console → Lambda
- AgentCore Runtime → AWS Console → Bedrock → AgentCore
- MCP Gateway → AWS Console → Bedrock → AgentCore → Gateways
- ECR repositories → AWS Console → ECR
- IAM roles → AWS Console → IAM

---

*Built for Vidya Sankalp Applied GenAI Engineering Program — Module 7: Agentic Systems on AWS*
