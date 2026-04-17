#!/bin/bash
# deploy.sh — vs-agentcore-platform-aws
# ========================================
# One-click deployment for students.
# Run individual steps or everything at once:
#
#   ./scripts/deploy.sh prompt     # Step 0: create/version Bedrock prompt → SSM
#   ./scripts/deploy.sh secrets    # Step 1: push secrets + SSM params
#   ./scripts/deploy.sh iam        # Step 2: create IAM roles
#   ./scripts/deploy.sh lambdas    # Step 3: build + deploy Lambda tools
#   ./scripts/deploy.sh gateway    # Step 4: create MCP Gateway + targets
#   ./scripts/deploy.sh platform   # Step 5: Terraform (ECS, ALB, RDS) — creates RDS first
#   ./scripts/deploy.sh agent      # Step 6: build + deploy AgentCore Runtime
#   ./scripts/deploy.sh all        # All steps in order (STUDENTS USE THIS)
#   ./scripts/deploy.sh redeploy   # Quick ECS redeploy after code changes
#   ./scripts/deploy.sh plan       # Terraform plan only (no AWS changes)
#   ./scripts/deploy.sh destroy    # Destroy Terraform resources
#
# DEPLOYMENT ORDER MATTERS:
#   platform runs BEFORE agent because:
#   - platform creates RDS PostgreSQL (LangGraph checkpointer)
#   - agent container needs POSTGRES_URL at cold start
#   - deploy platform → fill POSTGRES_URL in .env.prod → run secrets → run agent
#
# ARCHITECTURE NOTES:
#   AgentCore Runtime  → linux/arm64  (AgentCore ONLY supports arm64)
#   Lambda MCP tools   → linux/amd64  (Lambda x86_64)
#   ECS Platform/UI    → linux/amd64  (Fargate amd64)
#   All builds use --output type=registry to push directly to ECR —
#   avoids arm64/amd64 contamination when building on Apple Silicon Macs.
#
# GATEWAY TARGET NAMING:
#   Target "clarify" → tool name "ask_user_input" → LLM sees "clarify___ask_user_input"
#   Using "clarify" as the prefix makes the tool name natural English so GPT-4o
#   calls it reliably without code-enforced trigger words.
#
# IAM POLICY NOTES (fixes applied):
#   Agent role:
#     - dynamodb:CreateTable added  — TracerMiddleware.init_trace_table() needs it
#     - dynamodb:DescribeTable added — init_trace_table() checks if table exists first
#     - ecr:BatchGetImage resource changed to repository/* (was role ARN — caused ValidationException)
#   UI health check:
#     - Fixed in main.tf to /health — FastAPI serves /health, not /healthz (Chainlit default)
#   UI AGENT_API_KEY SSM path:
#     - Fixed in main.tf to /vs-agentcore/prod/platform_api_key (was wrong path)

set -euo pipefail

# ── Pre-requisite checks ───────────────────────────────────────────────────
check_prereqs() {
  local missing=()
  command -v aws        &>/dev/null || missing+=("aws-cli")
  command -v docker     &>/dev/null || missing+=("docker")
  command -v terraform  &>/dev/null || missing+=("terraform")
  command -v python3    &>/dev/null || missing+=("python3")
  python3 -c "import boto3" &>/dev/null || missing+=("boto3 — run: pip install boto3")
  if [ ${#missing[@]} -gt 0 ]; then
    echo "❌ Missing prerequisites: ${missing[*]}"
    echo "   Install them and re-run."
    exit 1
  fi
  aws sts get-caller-identity &>/dev/null || {
    echo "❌ AWS credentials not configured. Run: aws configure"
    exit 1
  }
}
check_prereqs

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PREFIX="vs-agentcore"
SSM_PREFIX="/${PREFIX}/prod"
GATEWAY_NAME="${PREFIX}-mcp"
ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

ACTION="${1:-plan}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "================================================"
echo "VS AgentCore Platform — ${ACTION}"
echo "Account: ${ACCOUNT_ID}  Region: ${REGION}"
echo "Root:    ${ROOT}"
echo "================================================"


# ── Helpers ────────────────────────────────────────────────────────────────

ecr_login() {
  echo "► ECR login"
  aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_BASE}"
}

ensure_ecr_repo() {
  local name="$1"
  aws ecr create-repository \
    --repository-name "${PREFIX}/${name}" \
    --region "${REGION}" > /dev/null 2>/dev/null || true
  echo "${ECR_BASE}/${PREFIX}/${name}"
}

wait_for_lambda() {
  local name="$1"
  echo -n "  Waiting for ${name}..."
  for i in {1..30}; do
    STATE=$(aws lambda get-function --function-name "${name}" \
      --region "${REGION}" --query 'Configuration.State' --output text 2>/dev/null || echo "NotFound")
    [ "${STATE}" = "Active" ] && echo " ✅" && return
    echo -n "."
    sleep 3
  done
  echo " ⚠️  timeout — check Lambda console"
}

wait_for_runtime() {
  local runtime_id="$1"
  echo -n "  Waiting for AgentCore runtime READY..."
  for i in {1..60}; do
    STATUS=$(aws bedrock-agentcore-control get-agent-runtime \
      --region "${REGION}" \
      --agent-runtime-id "${runtime_id}" \
      --query "status" --output text 2>/dev/null || echo "UNKNOWN")
    [ "${STATUS}" = "READY" ] && echo " ✅" && return
    echo -n "."
    sleep 5
  done
  echo " ⚠️  timeout — check AWS Console"
}


# ── Step 0: Bedrock Prompt ─────────────────────────────────────────────────

step_prompt() {
  echo ""
  echo "► Step 0: Bedrock Prompt"

  PROMPT_FILE="${ROOT}/prompts/system_prompt.txt"

  if [ ! -f "${PROMPT_FILE}" ]; then
    echo "  ❌ Prompt file not found: ${PROMPT_FILE}"
    echo "     Create prompts/system_prompt.txt with the system prompt and re-run."
    exit 1
  fi

  python3 - << PYEOF
import boto3, json, sys

region     = "${REGION}"
ssm_prefix = "${SSM_PREFIX}"
client     = boto3.client("bedrock-agent", region_name=region)
ssm_client = boto3.client("ssm",           region_name=region)

prompt_text = open("${PROMPT_FILE}").read()

# ── Check if prompt already exists ────────────────────────────────────────
existing_id = None
try:
    existing_id_param = ssm_client.get_parameter(
        Name="/clinical-trial-agent/prod/bedrock/prompt_id"
    )["Parameter"]["Value"]
    if existing_id_param and existing_id_param != "CHANGE_ME":
        existing_id = existing_id_param
        print(f"  Found existing prompt ID: {existing_id}")
except ssm_client.exceptions.ParameterNotFound:
    pass

if existing_id:
    # ── Create a new version of existing prompt ────────────────────────────
    print(f"  Creating new version of prompt {existing_id}...")
    resp = client.create_prompt_version(
        promptIdentifier=existing_id,
        description="Deployed by deploy.sh step_prompt"
    )
    prompt_id      = existing_id
    prompt_version = str(resp["version"])
    print(f"  ✅ New version created: {prompt_version}")
else:
    # ── Create brand new prompt ────────────────────────────────────────────
    print("  Creating new Bedrock prompt...")
    resp = client.create_prompt(
        name="vs-agentcore-clinical-trial",
        description="Clinical Trial Research Agent system prompt",
        variants=[{
            "name":              "default",
            "modelId":           "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "templateType":      "TEXT",
            "templateConfiguration": {
                "text": {
                    "text":           prompt_text,
                    "inputVariables": []
                }
            },
            "inferenceConfiguration": {
                "text": {
                    "temperature": 0.0,
                    "maxTokens":   4096,
                }
            }
        }],
        defaultVariant="default"
    )
    prompt_id = resp["id"]
    print(f"  ✅ Prompt created: {prompt_id}")

    v_resp = client.create_prompt_version(
        promptIdentifier=prompt_id,
        description="Initial version — deployed by deploy.sh"
    )
    prompt_version = str(v_resp["version"])
    print(f"  ✅ Version published: {prompt_version}")

# ── Write IDs to SSM — both paths the agent reads ─────────────────────────
for name in [
    "/clinical-trial-agent/prod/bedrock/prompt_id",
    f"{ssm_prefix}/bedrock/prompt_id",
]:
    ssm_client.put_parameter(Name=name, Value=prompt_id, Type="String", Overwrite=True)
    print(f"  ✅ SSM: {name} = {prompt_id}")

for name in [
    "/clinical-trial-agent/prod/bedrock/prompt_version",
    f"{ssm_prefix}/bedrock/prompt_version",
]:
    ssm_client.put_parameter(Name=name, Value=prompt_version, Type="String", Overwrite=True)
    print(f"  ✅ SSM: {name} = {prompt_version}")

print("")
print(f"  Prompt done ✅  ID={prompt_id}  version={prompt_version}")
print(f"  Re-deploy agent to pick up new prompt:")
print(f"    ./scripts/deploy.sh agent")
PYEOF
}


# ── Step 1: Secrets + SSM ─────────────────────────────────────────────────

step_secrets() {
  echo ""
  echo "► Step 1: Secrets + SSM"

  python3 - << PYEOF
import boto3, json, os, sys
from urllib.parse import urlparse

region     = "${REGION}"
ssm_prefix = "${SSM_PREFIX}"
sm         = boto3.client("secretsmanager", region_name=region)
ssm_client = boto3.client("ssm",            region_name=region)

def put_secret(name, value):
    try:
        sm.create_secret(Name=name, SecretString=json.dumps(value))
        print(f"  ✅ Created secret: {name}")
    except sm.exceptions.ResourceExistsException:
        sm.update_secret(SecretId=name, SecretString=json.dumps(value))
        print(f"  ✅ Updated secret: {name}")

def put_param(name, value, secure=False):
    ssm_client.put_parameter(
        Name=name, Value=value,
        Type="SecureString" if secure else "String",
        Overwrite=True,
    )
    print(f"  ✅ SSM param: {name}")

# BEDROCK_PROMPT_ID and BEDROCK_PROMPT_VERSION are managed by step_prompt — not required here
required = [
    "OPENAI_API_KEY", "PINECONE_API_KEY",
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
    "PLATFORM_API_KEY",
]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    print(f"ERROR: Missing required env vars: {missing}")
    print("Run: source .env.prod")
    sys.exit(1)

# ── API keys ───────────────────────────────────────────────────────────────
put_secret(f"{ssm_prefix}/openai",   {"api_key": os.environ["OPENAI_API_KEY"]})
put_secret(f"{ssm_prefix}/pinecone", {"api_key": os.environ["PINECONE_API_KEY"]})

# ── Neo4j ──────────────────────────────────────────────────────────────────
put_secret(f"{ssm_prefix}/neo4j", {
    "uri":      os.environ["NEO4J_URI"],
    "user":     os.environ["NEO4J_USER"],
    "password": os.environ["NEO4J_PASSWORD"],
})

# ── Postgres — only push if RDS endpoint is known ─────────────────────────
# POSTGRES_URL is empty on first deploy — RDS is created by step_platform.
# After step_platform completes: fill POSTGRES_URL in .env.prod and re-run:
#   source .env.prod && ./scripts/deploy.sh secrets
postgres_url = os.environ.get("POSTGRES_URL", "")
if postgres_url and "<rds-endpoint>" not in postgres_url:
    pg = urlparse(postgres_url)
    put_secret(f"{ssm_prefix}/postgres", {
        "username": pg.username,
        "password": pg.password,
        "host":     pg.hostname,
        "port":     str(pg.port or 5432),
        "dbname":   pg.path.lstrip("/"),
    })
    put_secret("clinical-agent/prod/postgres", {
        "username": pg.username,
        "password": pg.password,
        "host":     pg.hostname,
        "port":     str(pg.port or 5432),
        "dbname":   pg.path.lstrip("/"),
    })
    print("  ✅ Postgres secret written")
else:
    print("  ⏭  Skipping postgres — POSTGRES_URL not set yet")
    print("     After step_platform: fill POSTGRES_URL in .env.prod and re-run secrets")

# ── Platform auth ──────────────────────────────────────────────────────────
# Written to /vs-agentcore/prod/platform_api_key
# UI container reads this via SSM secrets injection (see main.tf)
put_secret(f"{ssm_prefix}/platform_api_key", {"api_key": os.environ["PLATFORM_API_KEY"]})

# ── Pinecone SSM params (agent reads these directly) ──────────────────────
put_param("/clinical-agent/prod/pinecone/api_key",
          os.environ["PINECONE_API_KEY"], secure=True)
put_param("/clinical-agent/prod/pinecone/index_name",
          os.environ.get("PINECONE_INDEX_NAME", "clinical-agent"))

# ── SSM non-secret config ──────────────────────────────────────────────────
put_param(f"{ssm_prefix}/pinecone/clinical_trials_index",
          os.environ.get("CLINICAL_TRIALS_INDEX", "clinical-trials-index"))
put_param(f"{ssm_prefix}/pinecone/cache_index_name",
          os.environ.get("PINECONE_INDEX_NAME", "clinical-agent"))
put_param(f"{ssm_prefix}/dynamodb/trace_table_name", "${PREFIX}-traces")
put_param("/clinical-agent/prod/dynamodb/trace_table_name", "${PREFIX}-traces")

# NOTE: Bedrock prompt ID and version are written by step_prompt, not here.

print("")
print("  All secrets and params written ✅")
PYEOF
}


# ── Step 2: IAM Roles ──────────────────────────────────────────────────────

step_iam() {
  echo ""
  echo "► Step 2: IAM roles"

  # ── Lambda execution role ─────────────────────────────────────────────────
  cat > /tmp/lambda-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-lambda-mcp" \
    --assume-role-policy-document file:///tmp/lambda-trust.json 2>/dev/null || true
  aws iam attach-role-policy --role-name "${PREFIX}-lambda-mcp" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
  cat > /tmp/lambda-secrets.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue", "ssm:GetParameter", "kms:Decrypt"],
      "Resource": "*"
    }
  ]
}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-lambda-mcp" \
    --policy-name SecretsAccess \
    --policy-document file:///tmp/lambda-secrets.json
  echo "  ✅ ${PREFIX}-lambda-mcp"

  # ── MCP Gateway role ──────────────────────────────────────────────────────
  cat > /tmp/gateway-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Sid":"GatewayAssumeRolePolicy","Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-gateway-role" \
    --assume-role-policy-document file:///tmp/gateway-trust.json 2>/dev/null || true
  cat > /tmp/gateway-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": [
        "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-search-tool",
        "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-graph-tool",
        "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-hitl-tool",
        "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-summariser-tool"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/bedrock-agentcore/runtimes/*:*"
    }
  ]
}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-gateway-role" \
    --policy-name GatewayPolicy \
    --policy-document file:///tmp/gateway-policy.json
  echo "  ✅ ${PREFIX}-gateway-role"

  # ── AgentCore Runtime role ────────────────────────────────────────────────
  cat > /tmp/agentcore-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-agent-role" \
    --assume-role-policy-document file:///tmp/agentcore-trust.json 2>/dev/null || true
  cat > /tmp/agentcore-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SecretsAndConfig",
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue", "ssm:GetParameter", "kms:Decrypt"],
      "Resource": "*"
    },
    {
      "Sid": "BedrockGateway",
      "Effect": "Allow",
      "Action": ["bedrock-agentcore:InvokeGateway"],
      "Resource": "*"
    },
    {
      "Sid": "BedrockPrompt",
      "Effect": "Allow",
      "Action": ["bedrock:GetPrompt"],
      "Resource": "*"
    },
    {
      "Sid": "DynamoDB",
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem",
        "dynamodb:Scan", "dynamodb:DescribeTable", "dynamodb:CreateTable"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ECRAuth",
      "Effect": "Allow",
      "Action": ["ecr:GetAuthorizationToken"],
      "Resource": "*"
    },
    {
      "Sid": "ECRPull",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchCheckLayerAvailability"
      ],
      "Resource": "arn:aws:ecr:${REGION}:${ACCOUNT_ID}:repository/*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup"],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogStreams",
      "Effect": "Allow",
      "Action": ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/bedrock-agentcore/runtimes/*:*"
    }
  ]
}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-agent-role" \
    --policy-name AgentCorePolicy \
    --policy-document file:///tmp/agentcore-policy.json
  echo "  ✅ ${PREFIX}-agent-role"

  echo ""
  echo "  IAM done ✅"
}


# ── Step 3: Lambda tools ───────────────────────────────────────────────────

step_lambdas() {
  echo ""
  echo "► Step 3: Lambda MCP tools"
  ecr_login

  LAMBDA_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-lambda-mcp"

  for tool in search graph hitl summariser; do
    echo ""
    echo "  ── ${tool}_lambda"
    REPO=$(ensure_ecr_repo "${tool}-tool")
    TAG="${REPO}:latest"
    FUNC="${PREFIX}-${tool}-tool"

    docker buildx build \
      --platform linux/amd64 \
      --output type=registry \
      --provenance=false \
      --no-cache \
      -t "${TAG}" \
      "${ROOT}/mcp_tools/${tool}_lambda"

    ENV_VARS="Variables={SSM_PREFIX=${SSM_PREFIX},AWS_REGION=${REGION}}"

    if aws lambda get-function --function-name "${FUNC}" --region "${REGION}" &>/dev/null; then
      echo "  Updating ${FUNC}..."
      aws lambda update-function-code \
        --function-name "${FUNC}" --image-uri "${TAG}" \
        --region "${REGION}" > /dev/null
    else
      echo "  Creating ${FUNC}..."
      aws lambda create-function \
        --function-name "${FUNC}" \
        --package-type Image \
        --code ImageUri="${TAG}" \
        --role "${LAMBDA_ROLE}" \
        --architectures x86_64 \
        --timeout 30 \
        --memory-size 512 \
        --image-config '{"Command":["handler.handler"]}' \
        --environment "${ENV_VARS}" \
        --region "${REGION}" > /dev/null
    fi

    wait_for_lambda "${FUNC}"

    echo "  ── Test invoke ${FUNC}:"
    case "${tool}" in
      search)     PAYLOAD='{"query":"cancer trial phase 3","top_k":2}' ;;
      graph)      PAYLOAD='{"cypher":"MATCH (t:Trial) RETURN t.nctId LIMIT 2"}' ;;
      hitl)       PAYLOAD='{"user_answer":"Pfizer BNT162b2"}' ;;
      summariser) PAYLOAD='{"chunks":["trial A results","trial B results"],"query":"test"}' ;;
    esac
    RESULT=$(aws lambda invoke --function-name "${FUNC}" \
      --payload "${PAYLOAD}" \
      --cli-binary-format raw-in-base64-out \
      --region "${REGION}" /tmp/lambda_out.json 2>/dev/null && cat /tmp/lambda_out.json)
    echo "  Response: ${RESULT:0:120}"
    if echo "${RESULT}" | grep -q '"error"'; then
      echo "  ⚠️  Lambda returned an error — check CloudWatch before proceeding"
    else
      echo "  ✅ ${FUNC} OK"
    fi
  done

  echo ""
  echo "  Lambdas done ✅"
}


# ── Step 4: MCP Gateway ────────────────────────────────────────────────────

step_gateway() {
  echo ""
  echo "► Step 4: MCP Gateway"

  GATEWAY_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-gateway-role"

  EXISTING_GW=$(aws bedrock-agentcore-control list-gateways \
    --region "${REGION}" \
    --query "items[?name=='${GATEWAY_NAME}'].gatewayId | [0]" \
    --output text 2>/dev/null || echo "")

  if [ -n "${EXISTING_GW}" ] && [ "${EXISTING_GW}" != "None" ]; then
    echo "  Gateway already exists: ${EXISTING_GW}"
    GATEWAY_ID="${EXISTING_GW}"
    GATEWAY_URL=$(aws bedrock-agentcore-control get-gateway \
      --region "${REGION}" --gateway-identifier "${GATEWAY_ID}" \
      --query "gatewayUrl" --output text 2>/dev/null || echo "")
    echo "  Gateway URL: ${GATEWAY_URL}"
  else
    echo "  Creating gateway ${GATEWAY_NAME}..."
    RESPONSE=$(aws bedrock-agentcore-control create-gateway \
      --region "${REGION}" \
      --name "${GATEWAY_NAME}" \
      --authorizer-type AWS_IAM \
      --protocol-type MCP \
      --role-arn "${GATEWAY_ROLE}" \
      --protocol-configuration "{\"mcp\":{\"supportedVersions\":[\"2025-03-26\"],\"instructions\":\"Clinical Trial Research MCP Gateway\"}}")

    GATEWAY_ID=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['gatewayId'])")
    GATEWAY_URL=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['gatewayUrl'])")
    echo "  Gateway ID:  ${GATEWAY_ID}"
    echo "  Gateway URL: ${GATEWAY_URL}"

    echo -n "  Waiting for ACTIVE..."
    for i in {1..24}; do
      STATUS=$(aws bedrock-agentcore-control get-gateway \
        --region "${REGION}" --gateway-identifier "${GATEWAY_ID}" \
        --query 'status' --output text 2>/dev/null || echo "UNKNOWN")
      [ "${STATUS}" = "ACTIVE" ] && echo " ✅" && break
      echo -n "."
      sleep 5
    done

    register_target() {
      local tgt_name="$1" tool_name="$2" tool_desc="$3" lambda_func="$4" schema="$5"
      echo "  Registering ${tgt_name} → ${lambda_func}..."
      aws bedrock-agentcore-control create-gateway-target \
        --region "${REGION}" \
        --gateway-identifier "${GATEWAY_ID}" \
        --name "${tgt_name}" \
        --description "${tool_desc}" \
        --target-configuration "{\"mcp\":{\"lambda\":{\"lambdaArn\":\"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${lambda_func}\",\"toolSchema\":{\"inlinePayload\":[{\"name\":\"${tool_name}\",\"description\":\"${tool_desc}\",\"inputSchema\":${schema}}]}}}}" \
        --credential-provider-configurations "[{\"credentialProviderType\":\"GATEWAY_IAM_ROLE\"}]" > /dev/null
      echo "  ✅ ${tgt_name}"
    }

    register_target "tool-search" "search_tool" \
      "Semantic search over 5,772 clinical trial document chunks in Pinecone. Best for: efficacy, safety, dosage, endpoints, adverse events, trial results, document-level evidence." \
      "${PREFIX}-search-tool" \
      '{"type":"object","properties":{"query":{"type":"string","description":"Natural language search query"},"top_k":{"type":"integer","description":"Number of results (default 5)"}},"required":["query"]}'

    register_target "tool-graph" "graph_tool" \
      "Cypher query on Neo4j biomedical knowledge graph. Best for: what trials exist, trial names and NCT IDs, sponsors, drug-disease relationships, patient eligibility. Schema: (Trial)-[:USES]->(Drug),[:TARGETS]->(Disease),[:SPONSORED_BY]->(Sponsor). Read-only." \
      "${PREFIX}-graph-tool" \
      '{"type":"object","properties":{"cypher":{"type":"string","description":"Read-only Cypher query. No CREATE, MERGE, SET, DELETE, DROP."}},"required":["cypher"]}'

    # Target name "clarify" → LLM sees "clarify___ask_user_input" — natural English
    register_target "clarify" "ask_user_input" \
      "Ask user to clarify an ambiguous query. Options MUST be exact trial names or NCT IDs from search/graph results — never generic categories." \
      "${PREFIX}-hitl-tool" \
      '{"type":"object","properties":{"question":{"type":"string"},"options":{"type":"array","items":{"type":"string"}},"allow_freetext":{"type":"boolean"},"user_answer":{"type":"string"}},"required":[]}'

    register_target "tool-summariser" "summariser_tool" \
      "FINAL step only. Synthesise chunks from search_tool/graph_tool into one answer with citations. Never call first." \
      "${PREFIX}-summariser-tool" \
      '{"type":"object","properties":{"chunks":{"type":"array","items":{"type":"string"}},"query":{"type":"string"}},"required":["chunks"]}'
  fi

  aws ssm put-parameter \
    --name "${SSM_PREFIX}/mcp/gateway_url" \
    --value "${GATEWAY_URL}" \
    --type String --overwrite \
    --region "${REGION}"

  echo ""
  echo "  Gateway done ✅  URL: ${GATEWAY_URL}"
  echo "  Targets:"
  aws bedrock-agentcore-control list-gateway-targets \
    --gateway-identifier "${GATEWAY_ID}" \
    --region "${REGION}" \
    --query 'items[].{name:name,status:status}' \
    --output table 2>/dev/null || true
}


# ── Step 5: Platform + UI (ECS Fargate via Terraform) ─────────────────────
# NOTE: This runs BEFORE step_agent so RDS exists when the agent cold-starts

step_platform() {
  echo ""
  echo "► Step 5: Platform + UI (ECS Fargate)"
  ecr_login

  PLATFORM_REPO=$(ensure_ecr_repo "platform")
  PLATFORM_TAG="${PLATFORM_REPO}:latest"
  docker buildx build \
    --platform linux/amd64 \
    --output type=registry \
    --provenance=false \
    --no-cache \
    -t "${PLATFORM_TAG}" \
    "${ROOT}/platform"
  echo "  ✅ Platform: ${PLATFORM_TAG}"

  UI_REPO=$(ensure_ecr_repo "ui")
  UI_TAG="${UI_REPO}:latest"
  docker buildx build \
    --platform linux/amd64 \
    --output type=registry \
    --provenance=false \
    --no-cache \
    -t "${UI_TAG}" \
    "${ROOT}/ui"
  echo "  ✅ UI: ${UI_TAG}"

  cd "${ROOT}/infra"

  aws s3 mb s3://${PREFIX}-tfstate --region "${REGION}" 2>/dev/null || true
  aws s3api put-bucket-versioning \
    --bucket ${PREFIX}-tfstate \
    --versioning-configuration Status=Enabled 2>/dev/null || true

  if [ -z "${RDS_PASSWORD:-}" ]; then
    echo "  ❌ RDS_PASSWORD not set — set it in .env.prod"
    exit 1
  fi
  export TF_VAR_postgres_password="${RDS_PASSWORD}"

  terraform init -upgrade -input=false

  TF_VARS="-var=platform_image_uri=${PLATFORM_TAG} -var=ui_image_uri=${UI_TAG} -var=aws_region=${REGION} -var=ssm_prefix=${SSM_PREFIX}"

  if [ "${ACTION}" = "plan" ]; then
    terraform plan ${TF_VARS}
  else
    terraform apply -auto-approve -input=false ${TF_VARS}
    ALB_DNS=$(terraform output -raw alb_dns 2>/dev/null || echo "check-terraform-output")
    RDS_EP=$(terraform output -raw rds_endpoint 2>/dev/null || echo "check-terraform-output")

    echo ""
    echo "  Platform done ✅"
    echo "  ALB DNS:      http://${ALB_DNS}"
    echo "  RDS endpoint: ${RDS_EP}"
    echo ""
    echo "  ════════════════════════════════════════════════════"
    echo "  ⚠️  REQUIRED before running agent step:"
    echo ""
    echo "  1. Fill POSTGRES_URL in .env.prod:"
    echo "     POSTGRES_URL=postgresql://postgres:<password>@${RDS_EP}/clinical_agent"
    echo ""
    echo "  2. Push postgres credentials:"
    echo "     source .env.prod && ./scripts/deploy.sh secrets"
    echo ""
    echo "  3. Then deploy the agent:"
    echo "     ./scripts/deploy.sh agent"
    echo "  ════════════════════════════════════════════════════"
  fi
}


# ── Step 6: AgentCore Runtime ──────────────────────────────────────────────
# NOTE: Runs AFTER step_platform so RDS exists for the agent's Postgres connection

step_agent() {
  echo ""
  echo "► Step 6: AgentCore Runtime"
  ecr_login

  AGENT_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-agent-role"
  AGENT_DIR="${ROOT}/agent"
  AGENT_REPO=$(ensure_ecr_repo "agent")
  AGENT_TAG="${AGENT_REPO}:latest"

  echo "  Building agent container (linux/arm64 — AgentCore requirement)..."
  docker buildx build \
    --platform linux/arm64 \
    --output type=registry \
    --provenance=false \
    --no-cache \
    -t "${AGENT_TAG}" \
    "${AGENT_DIR}"
  echo "  ✅ Agent image pushed: ${AGENT_TAG}"

  ENV_JSON="{\"SSM_PREFIX\":\"${SSM_PREFIX}\",\"AWS_REGION\":\"${REGION}\",\"AWS_DEFAULT_REGION\":\"${REGION}\",\"AGENT_ENV\":\"prod\"}"

  EXISTING_ARN=$(aws bedrock-agentcore-control list-agent-runtimes \
    --region "${REGION}" \
    --query "agentRuntimes[?agentRuntimeName=='${PREFIX//-/_}_clinical_trial'].agentRuntimeArn | [0]" \
    --output text 2>/dev/null || echo "")

  if [ -n "${EXISTING_ARN}" ] && [ "${EXISTING_ARN}" != "None" ]; then
    echo "  Runtime exists — updating with new image..."
    RUNTIME_ID=$(echo "${EXISTING_ARN}" | sed 's/.*runtime\///')
    aws bedrock-agentcore-control update-agent-runtime \
      --region "${REGION}" \
      --agent-runtime-id "${RUNTIME_ID}" \
      --role-arn "${AGENT_ROLE}" \
      --network-configuration "{\"networkMode\":\"PUBLIC\"}" \
      --agent-runtime-artifact "{\"containerConfiguration\":{\"containerUri\":\"${AGENT_TAG}\"}}" \
      --environment-variables "${ENV_JSON}" > /dev/null
    RUNTIME_ARN="${EXISTING_ARN}"
    echo "  Waiting for runtime to be READY..."
    wait_for_runtime "${RUNTIME_ID}"
  else
    echo "  Creating AgentCore Runtime..."
    RUNTIME_RESPONSE=$(aws bedrock-agentcore-control create-agent-runtime \
      --region "${REGION}" \
      --agent-runtime-name "${PREFIX//-/_}_clinical_trial" \
      --description "Clinical Trial Research Agent" \
      --role-arn "${AGENT_ROLE}" \
      --agent-runtime-artifact "{\"containerConfiguration\":{\"containerUri\":\"${AGENT_TAG}\"}}" \
      --network-configuration "{\"networkMode\":\"PUBLIC\"}" \
      --environment-variables "${ENV_JSON}" 2>/dev/null || echo "{}")

    RUNTIME_ARN=$(echo "${RUNTIME_RESPONSE}" | python3 -c \
      "import sys,json; d=json.load(sys.stdin); print(d.get('agentRuntimeArn',''))" 2>/dev/null || echo "")

    if [ -z "${RUNTIME_ARN}" ] || [ "${RUNTIME_ARN}" = "None" ]; then
      echo "  ❌ Could not create Runtime — check AWS Console"
      exit 1
    fi

    RUNTIME_ID=$(echo "${RUNTIME_ARN}" | sed 's/.*runtime\///')
    echo "  Waiting for runtime to be READY..."
    wait_for_runtime "${RUNTIME_ID}"
  fi

  aws ssm put-parameter \
    --name "${SSM_PREFIX}/agent_runtime_arn" \
    --value "${RUNTIME_ARN}" \
    --type String --overwrite \
    --region "${REGION}"

  echo "  Runtime ARN: ${RUNTIME_ARN}"
  aws bedrock-agentcore-control get-agent-runtime \
    --region "${REGION}" \
    --agent-runtime-id "${RUNTIME_ID}" \
    --query "{Version:agentRuntimeVersion,Tier:agentRuntimeTier}" \
    --output json
  echo ""
  echo "  Agent done ✅"
}


# ── Quick ECS redeploy ─────────────────────────────────────────────────────

step_redeploy() {
  local target="${2:-both}"
  echo ""
  echo "► Quick ECS redeploy — force-new-deployment (no Terraform, no image rebuild)"

  if [ "${target}" = "platform" ] || [ "${target}" = "both" ]; then
    aws ecs update-service \
      --cluster "${PREFIX}-cluster" \
      --service "${PREFIX}-platform" \
      --force-new-deployment \
      --region "${REGION}" \
      --query "service.deployments[0].{id:id,state:rolloutState}" \
      --output table
    echo "  ✅ ${PREFIX}-platform redeploy triggered"
  fi

  if [ "${target}" = "ui" ] || [ "${target}" = "both" ]; then
    aws ecs update-service \
      --cluster "${PREFIX}-cluster" \
      --service "${PREFIX}-ui" \
      --force-new-deployment \
      --region "${REGION}" \
      --query "service.deployments[0].{id:id,state:rolloutState}" \
      --output table
    echo "  ✅ ${PREFIX}-ui redeploy triggered"
  fi
}


# ── Main dispatch ──────────────────────────────────────────────────────────

case "${ACTION}" in
  prompt)   step_prompt   ;;
  secrets)  step_secrets  ;;
  iam)      step_iam      ;;
  lambdas)  step_lambdas  ;;
  gateway)  step_gateway  ;;
  platform) step_platform ;;
  plan)     step_platform ;;
  agent)    step_agent    ;;
  redeploy) step_redeploy "$@" ;;

  all)
    # ORDER MATTERS:
    # 1. prompt   — create Bedrock prompt, write IDs to SSM
    # 2. secrets  — push API keys (postgres skipped — RDS not created yet)
    # 3. iam      — create IAM roles
    # 4. lambdas  — build + deploy Lambda tools
    # 5. gateway  — create MCP Gateway
    # 6. platform — Terraform: creates RDS, ECS, ALB  ← MUST be before agent
    # 7. (manual) — fill POSTGRES_URL in .env.prod, re-run secrets
    # 8. agent    — build + deploy AgentCore Runtime
    step_prompt
    step_secrets
    step_iam
    step_lambdas
    step_gateway
    step_platform
    echo ""
    echo "  ════════════════════════════════════════════════════"
    echo "  ⚠️  MANUAL STEP REQUIRED before deploying agent:"
    echo ""
    echo "  1. Get RDS endpoint from terraform output above"
    echo "  2. Edit .env.prod:"
    echo "     POSTGRES_URL=postgresql://postgres:<pwd>@<rds-endpoint>/clinical_agent"
    echo "  3. Push postgres secret:"
    echo "     source .env.prod && ./scripts/deploy.sh secrets"
    echo "  4. Deploy agent:"
    echo "     ./scripts/deploy.sh agent"
    echo "  ════════════════════════════════════════════════════"
    ;;

  destroy)
    echo "⚠️  Destroying Terraform resources..."
    cd "${ROOT}/infra"
    terraform destroy -auto-approve \
      -var="platform_image_uri=placeholder" \
      -var="ui_image_uri=placeholder" \
      -var="aws_region=${REGION}" \
      -var="ssm_prefix=${SSM_PREFIX}"
    ;;

  *)
    echo "Usage: $0 {prompt|secrets|iam|lambdas|gateway|platform|agent|redeploy|all|plan|destroy}"
    echo ""
    echo "  prompt   — Create/version Bedrock prompt → SSM"
    echo "  all      — Steps 0-6 (pause after platform to fill POSTGRES_URL)"
    echo "  agent    — Build + deploy AgentCore Runtime (run after platform + postgres secret)"
    echo "  redeploy — Quick ECS force-redeploy: redeploy [platform|ui|both]"
    echo "  plan     — Terraform plan only (preview changes)"
    echo "  destroy  — Tear down all AWS infrastructure"
    exit 1
    ;;
esac