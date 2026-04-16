# vs-agentcore-platform-aws

Clinical Trial Research Agent — Production on AWS Bedrock AgentCore.

## Architecture

```
Chainlit UI (ECS Fargate :8501)
        │  SSE stream (token-by-token)
       ALB (idle_timeout=300s)
        │
FastAPI Platform (ECS Fargate :8000)
        │  boto3 invoke_agent_runtime → SSE proxy
Bedrock AgentCore Runtime (microVM per session)
        │  @app.entrypoint async generator → auto SSE
        │  LangGraph + 9-layer middleware
        │
MCP Gateway (bedrock-agentcore-control, AWS_IAM auth)
        ├── tool-search     → search_lambda  (Pinecone)
        ├── tool-graph      → graph_lambda   (Neo4j AuraDB)
        ├── tool-hitl       → hitl_lambda    (HITL)
        └── tool-summariser → summariser_lambda (GPT-4o-mini)
              │
        RDS Postgres (LangGraph checkpointer)
        Pinecone     (semantic cache + episodic memory)
        DynamoDB     (trace logs)
```

## Project Structure

```
vs-agentcore-platform-aws/
  agent/                       ← AgentCore Runtime container
    agent.py                   ← @app.entrypoint async generator (SSE)
    graph.py                   ← LangGraph agent builder
    prompt.py                  ← Bedrock Prompt Management
    middleware/__init__.py     ← 9-layer middleware stack
    tools/mcp_client.py        ← MCP Gateway client (AWS_IAM SigV4)
    requirements.txt
    Dockerfile
  platform/                    ← FastAPI SSE proxy
    main.py                    ← StreamingResponse + AgentCore invoke
    gateway/
      auth.py / schemas.py / rate_limiter.py / logging_mw.py
    requirements.txt
    Dockerfile
  mcp_tools/                   ← Lambda MCP tools
    search_lambda/             → handler.py + Dockerfile + requirements.txt
    graph_lambda/              → handler.py + Dockerfile + requirements.txt
    hitl_lambda/               → handler.py + Dockerfile + requirements.txt
    summariser_lambda/         → handler.py + Dockerfile + requirements.txt
  ui/                          ← Chainlit SSE streaming UI
    app.py
    requirements.txt
    Dockerfile
  infra/                       ← Terraform (VPC, ALB, ECS, RDS)
    main.tf / variables.tf
  scripts/
    deploy.sh                  ← Full deploy in order
  .env.prod.example
```

## Deploy

```bash
# 1. Fill in secrets
cp ..env.prod.example ..env.prod
source ..env.prod

# 2. Create Terraform state bucket (once)
aws s3 mb s3://vs-agentcore-tfstate --region us-east-1

# 3. Full deploy
chmod +x scripts/deploy.sh
./scripts/deploy.sh apply
```

## Deploy Steps (what deploy.sh does)

1. **Secrets** — push all API keys to Secrets Manager + SSM
2. **IAM** — create Lambda, Gateway, AgentCore execution roles
3. **Lambda tools** — build Docker images, push to ECR, create functions
4. **MCP Gateway** — create gateway (AWS_IAM auth), register 4 tool targets
5. **AgentCore Runtime** — build agent image, deploy to AgentCore
6. **Platform + UI** — build images, deploy to ECS Fargate via Terraform

## Local Testing

```bash
# Run platform locally pointing at AgentCore
export AGENT_API_URL=http://localhost:8000
export AGENT_API_KEY=local-dev-key
chainlit run ui/app.py --port 8501

# Run platform locally
cd platform
PLATFORM_API_KEY=local-dev-key \
SSM_PREFIX=/vs-agentcore/prod \
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Key Differences from Local (vs-agentic-platform)

| | Local | Production |
|--|--|--|
| Agent runtime | uvicorn FastAPI | AgentCore microVM |
| Tools | in-process functions | Lambda via MCP Gateway |
| Streaming | sync response | SSE (token by token) |
| Checkpointer | local Postgres | RDS Postgres |
| Auth | env var | Secrets Manager |
| UI | Chainlit direct | Chainlit → ALB → Platform → AgentCore |
