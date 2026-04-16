#!/bin/bash
# deploy.sh — vs-agentcore-platform-aws
# ========================================
# Run individual steps or all at once:
#
#   ./scripts/deploy.sh secrets    # Step 1: push secrets + SSM params
#   ./scripts/deploy.sh iam        # Step 2: create IAM roles
#   ./scripts/deploy.sh lambdas    # Step 3: build + deploy Lambda tools
#   ./scripts/deploy.sh gateway    # Step 4: create MCP Gateway + targets
#   ./scripts/deploy.sh agent      # Step 5: build + deploy AgentCore Runtime
#   ./scripts/deploy.sh platform   # Step 6: Terraform (ECS, ALB, RDS)
#   ./scripts/deploy.sh all        # All 6 steps in order
#   ./scripts/deploy.sh plan       # Terraform plan only (no AWS changes)
#   ./scripts/deploy.sh destroy    # Destroy Terraform resources

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PREFIX="vs-agentcore"
SSM_PREFIX="/${PREFIX}/prod"
GATEWAY_NAME="${PREFIX}-mcp"
ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

ACTION="${1:-plan}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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


# ── Step 1: Secrets + SSM ─────────────────────────────────────────────────

step_secrets() {
  echo ""
  echo "► Step 1: Secrets + SSM"

  # Verify required env vars
  REQUIRED="OPENAI_API_KEY PINECONE_API_KEY NEO4J_URI NEO4J_USER NEO4J_PASSWORD PLATFORM_API_KEY BEDROCK_PROMPT_ID BEDROCK_PROMPT_VERSION"
  MISSING=""
  for v in $REQUIRED; do
    [ -z "${!v:-}" ] && MISSING="$MISSING $v"
  done
  if [ -n "$MISSING" ]; then
    echo "  ❌ Missing env vars:$MISSING"
    echo "     Run: source .env.prod"
    exit 1
  fi

  # ── Helper: create or update a Secrets Manager secret ──────────────────
  put_secret() {
    local name="$1" value="$2"
    if aws secretsmanager describe-secret --secret-id "$name" \
         --region "$REGION" &>/dev/null; then
      aws secretsmanager update-secret \
        --secret-id "$name" --secret-string "$value" \
        --region "$REGION" > /dev/null
    else
      aws secretsmanager create-secret \
        --name "$name" --secret-string "$value" \
        --region "$REGION" > /dev/null
    fi
    echo "  ✅ Secret: $name"
  }

  # ── Helper: put SSM parameter ───────────────────────────────────────────
  put_param() {
    local name="$1" value="$2" type="${3:-String}"
    aws ssm put-parameter \
      --name "$name" --value "$value" \
      --type "$type" --overwrite \
      --region "$REGION" > /dev/null
    echo "  ✅ SSM:    $name"
  }

  ENV_TAG="prod"
  VS="${SSM_PREFIX}"                     # /vs-agentcore/prod
  CA="/clinical-agent/${ENV_TAG}"        # /clinical-agent/prod

  # ── Pinecone ───────────────────────────────────────────────────────────
  # Agent container reads from SSM (aws.py uses get_ssm_parameter)
  put_param "${CA}/pinecone/api_key"   "${PINECONE_API_KEY}"  "SecureString"
  put_param "${CA}/pinecone/index_name" \
    "${PINECONE_INDEX_NAME:-clinical-agent}"
  # Lambdas + platform read from Secrets Manager
  put_secret "${VS}/pinecone" "{\"api_key\":\"${PINECONE_API_KEY}\"}"
  put_param  "${VS}/pinecone/clinical_trials_index" \
    "${CLINICAL_TRIALS_INDEX:-clinical-trials-index}"
  put_param  "${VS}/pinecone/cache_index_name" \
    "${PINECONE_INDEX_NAME:-clinical-agent}"

  # ── OpenAI (search_lambda + summariser_lambda) ────────────────────────
  put_secret "${VS}/openai" "{\"api_key\":\"${OPENAI_API_KEY}\"}"

  # ── Neo4j (graph_lambda) ──────────────────────────────────────────────
  put_secret "${VS}/neo4j" \
    "{\"uri\":\"${NEO4J_URI}\",\"user\":\"${NEO4J_USER}\",\"password\":\"${NEO4J_PASSWORD}\"}"

  # ── DynamoDB table name ───────────────────────────────────────────────
  put_param "${CA}/dynamodb/trace_table_name" "vs-agentcore-traces"
  put_param "${VS}/dynamodb/trace_table_name" "vs-agentcore-traces"

  # ── Platform API key ──────────────────────────────────────────────────
  put_param  "${CA}/platform/api_key"    "${PLATFORM_API_KEY}" "SecureString"
  put_secret "${VS}/platform_api_key"    "{\"api_key\":\"${PLATFORM_API_KEY}\"}"

  # ── Bedrock prompt ────────────────────────────────────────────────────
  put_param "/clinical-trial-agent/${ENV_TAG}/bedrock/prompt_id"      "${BEDROCK_PROMPT_ID}"
  put_param "/clinical-trial-agent/${ENV_TAG}/bedrock/prompt_version"  "${BEDROCK_PROMPT_VERSION}"
  put_param "${VS}/bedrock/prompt_id"      "${BEDROCK_PROMPT_ID}"
  put_param "${VS}/bedrock/prompt_version"  "${BEDROCK_PROMPT_VERSION}"

  # ── Postgres — only when RDS endpoint is known ────────────────────────
  # POSTGRES_URL is blank on first deploy (RDS doesn't exist yet).
  # After step_platform: fill POSTGRES_URL in .env.prod and re-run secrets.
  if [ -n "${POSTGRES_URL:-}" ] && [[ "${POSTGRES_URL}" != *"<rds-endpoint>"* ]]; then
    # Parse postgresql://username:password@host:port/dbname using bash
    # Remove scheme
    _rest="${POSTGRES_URL#postgresql://}"
    # Split user:pass from host
    _userpass="${_rest%@*}"
    _hostpath="${_rest#*@}"
    _user="${_userpass%:*}"
    _pass="${_userpass#*:}"
    _host="${_hostpath%%:*}"
    _portdb="${_hostpath#*:}"
    _port="${_portdb%/*}"
    _db="${_portdb#*/}"

    put_secret "clinical-agent/${ENV_TAG}/postgres" \
      "{\"username\":\"${_user}\",\"password\":\"${_pass}\",\"host\":\"${_host}\",\"port\":\"${_port}\",\"dbname\":\"${_db}\"}"
  else
    echo "  ⏭  Postgres: POSTGRES_URL not set — skipping (fill after Terraform creates RDS)"
  fi

  echo ""
  echo "  All secrets and params written ✅"
}


# ── Step 2: IAM Roles ──────────────────────────────────────────────────────

step_iam() {
  echo ""
  echo "► Step 2: IAM roles"

  # Lambda execution role — reads Secrets Manager + SSM
  cat > /tmp/lambda-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-lambda-mcp" \
    --assume-role-policy-document file:///tmp/lambda-trust.json 2>/dev/null || true
  aws iam attach-role-policy --role-name "${PREFIX}-lambda-mcp" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
  cat > /tmp/lambda-secrets.json << POLICY
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["secretsmanager:GetSecretValue","ssm:GetParameter","kms:Decrypt"],"Resource":"*"}
]}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-lambda-mcp" \
    --policy-name SecretsAccess --policy-document file:///tmp/lambda-secrets.json
  echo "  ✅ ${PREFIX}-lambda-mcp"

  # MCP Gateway role — invokes Lambda tools
  cat > /tmp/gateway-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Sid":"GatewayAssumeRolePolicy","Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-gateway-role" \
    --assume-role-policy-document file:///tmp/gateway-trust.json 2>/dev/null || true
  cat > /tmp/gateway-lambda.json << POLICY
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["lambda:InvokeFunction"],"Resource":[
    "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-search-tool",
    "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-graph-tool",
    "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-hitl-tool",
    "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-summariser-tool"
  ]},
  {"Effect":"Allow","Action":["logs:CreateLogGroup"],"Resource":"*"},
  {"Effect":"Allow","Action":["logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"],"Resource":"arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/bedrock-agentcore/runtimes/*:*"}
]}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-gateway-role" \
    --policy-name GatewayPolicy --policy-document file:///tmp/gateway-lambda.json
  echo "  ✅ ${PREFIX}-gateway-role"

  # AgentCore Runtime role — calls gateway, Bedrock, SSM, Secrets, DynamoDB
  cat > /tmp/agentcore-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-agent-role" \
    --assume-role-policy-document file:///tmp/agentcore-trust.json 2>/dev/null || true
  cat > /tmp/agentcore-policy.json << POLICY
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["secretsmanager:GetSecretValue","ssm:GetParameter","kms:Decrypt"],"Resource":"*"},
  {"Effect":"Allow","Action":["bedrock-agentcore:InvokeGateway"],"Resource":"*"},
  {"Effect":"Allow","Action":["bedrock:GetPrompt"],"Resource":"*"},
  {"Effect":"Allow","Action":["dynamodb:PutItem","dynamodb:GetItem","dynamodb:UpdateItem","dynamodb:Scan"],"Resource":"*"},
  {"Effect":"Allow","Action":["ecr:GetAuthorizationToken"],"Resource":"*"},
  {"Effect":"Allow","Action":["ecr:BatchGetImage","ecr:GetDownloadUrlForLayer","ecr:BatchCheckLayerAvailability"],"Resource":"arn:aws:ecr:${REGION}:${ACCOUNT_ID}:repository/*"},
  {"Effect":"Allow","Action":["logs:CreateLogGroup"],"Resource":"*"},
  {"Effect":"Allow","Action":["logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"],"Resource":"arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/bedrock-agentcore/runtimes/*:*"}
]}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-agent-role" \
    --policy-name AgentCorePolicy --policy-document file:///tmp/agentcore-policy.json
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

    ENV_VARS="Variables={SSM_PREFIX=${SSM_PREFIX}}"

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
      search)    PAYLOAD='{"query":"cancer trial phase 3","top_k":2}' ;;
      graph)     PAYLOAD='{"cypher":"MATCH (t:Trial) RETURN t.nctId LIMIT 2"}' ;;
      hitl)      PAYLOAD='{"user_answer":"Pfizer BNT162b2"}' ;;
      summariser) PAYLOAD='{"chunks":["trial A results","trial B results"],"query":"test"}' ;;
    esac
    RESULT=$(aws lambda invoke --function-name "${FUNC}" \
      --payload "${PAYLOAD}" \
      --cli-binary-format raw-in-base64-out \
      --region "${REGION}" /tmp/lambda_out.json 2>/dev/null && cat /tmp/lambda_out.json)
    echo "  Response: ${RESULT:0:120}"
    if echo "${RESULT}" | grep -q '"error"'; then
      echo "  ⚠️  Lambda returned an error — check CloudWatch before proceeding to gateway step"
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

  # Register tool targets
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
    "ALWAYS call first. Semantic search over 5,772 clinical trial chunks. Use for: efficacy, safety, dosage, endpoints, adverse events, patient populations, trial results." \
    "${PREFIX}-search-tool" \
    '{"type":"object","properties":{"query":{"type":"string","description":"Natural language search query"},"top_k":{"type":"integer","description":"Number of results (default 5)"}},"required":["query"]}'

  register_target "tool-graph" "graph_tool" \
    "Cypher query on Neo4j trial graph. Use AFTER search_tool for relationships. Schema: (Trial)-[:USES]->(Drug),[:TARGETS]->(Disease),[:SPONSORED_BY]->(Sponsor). Read-only." \
    "${PREFIX}-graph-tool" \
    '{"type":"object","properties":{"cypher":{"type":"string","description":"Read-only Cypher query. No CREATE, MERGE, SET, DELETE, DROP."}},"required":["cypher"]}'

  register_target "tool-hitl" "ask_user_input" \
    "Ask user to clarify ambiguous queries. ONLY when search_tool cannot resolve ambiguity. Always call search_tool first. Never ask what you can retrieve yourself." \
    "${PREFIX}-hitl-tool" \
    '{"type":"object","properties":{"question":{"type":"string"},"options":{"type":"array","items":{"type":"string"}},"allow_freetext":{"type":"boolean"},"user_answer":{"type":"string"}},"required":[]}'

  register_target "tool-summariser" "summariser_tool" \
    "FINAL step only. Synthesise chunks from search_tool/graph_tool into one answer with trial ID citations. Never call first — retrieve evidence first." \
    "${PREFIX}-summariser-tool" \
    '{"type":"object","properties":{"chunks":{"type":"array","items":{"type":"string"}},"query":{"type":"string"}},"required":["chunks"]}'


  # Store gateway URL so agent container can find it
  aws ssm put-parameter \
    --name "${SSM_PREFIX}/mcp/gateway_url" \
    --value "${GATEWAY_URL}" \
    --type String --overwrite \
    --region "${REGION}"

  echo ""
  echo ""
  echo "  ── Testing tools via Gateway ─────────────────────────────────────"

  python3 - << INNEREOF
import boto3, json, sys

try:
    import httpx
except ImportError:
    print("  ⚠️  httpx not installed — skipping gateway test (pip3 install httpx)")
    sys.exit(0)

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

REGION = "${REGION}"
URL    = "${GATEWAY_URL}"

def call_tool(tool_name, arguments):
    body  = json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/call",
                        "params":{"name":tool_name,"arguments":arguments}}).encode()
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    req   = AWSRequest(method="POST", url=URL, data=body,
                       headers={"Content-Type":"application/json"})
    SigV4Auth(creds, "bedrock-agentcore", REGION).add_auth(req)
    resp  = httpx.post(URL, content=body, headers=dict(req.headers), timeout=30)
    return resp.json()

all_ok = True

# search_tool
try:
    r = call_tool("tool-search___search_tool", {"query": "COVID-19 vaccine phase 3 efficacy", "top_k": 2})
    if r.get("result", {}).get("isError") == False:
        hits = json.loads(r["result"]["content"][0]["text"]).get("count", 0)
        print(f"  ✅ search_tool    → {hits} results")
    else:
        print(f"  ❌ search_tool    → {r}"); all_ok = False
except Exception as e:
    print(f"  ❌ search_tool    → {e}"); all_ok = False

# graph_tool
try:
    r = call_tool("tool-graph___graph_tool",
                  {"cypher": "MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC LIMIT 3"})
    if r.get("result", {}).get("isError") == False:
        rows = json.loads(r["result"]["content"][0]["text"]).get("count", 0)
        print(f"  ✅ graph_tool     → {rows} node types found")
    else:
        print(f"  ❌ graph_tool     → {r}"); all_ok = False
except Exception as e:
    print(f"  ❌ graph_tool     → {e}"); all_ok = False

# summariser_tool
try:
    r = call_tool("tool-summariser___summariser_tool", {
        "chunks": ["Trial showed 95% efficacy", "44,000 participants enrolled"],
        "query":  "What is the efficacy?"
    })
    if r.get("result", {}).get("isError") == False:
        summary = json.loads(r["result"]["content"][0]["text"]).get("summary", "")
        print(f"  ✅ summariser_tool → {summary[:80]}")
    else:
        print(f"  ❌ summariser_tool → {r}"); all_ok = False
except Exception as e:
    print(f"  ❌ summariser_tool → {e}"); all_ok = False

print()
if all_ok:
    print("  All gateway tools OK ✅")
else:
    print("  ⚠️  Some tools failed — check Lambda CloudWatch logs before running step agent")
    sys.exit(1)
INNEREOF

  echo "  Gateway done ✅  URL: ${GATEWAY_URL}"
  echo "  ⚠️  Verify all targets are ACTIVE before running step agent:"
  aws bedrock-agentcore-control list-gateway-targets \
    --gateway-identifier "${GATEWAY_ID}" \
    --region "${REGION}" \
    --query 'items[].{name:name,status:status}' \
    --output table 2>/dev/null || true
}


# ── Step 5: AgentCore Runtime ──────────────────────────────────────────────

step_agent() {
  echo ""
  echo "► Step 5: AgentCore Runtime"
  ecr_login

  AGENT_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-agent-role"
  AGENT_DIR="${ROOT}/agent"
  AGENT_REPO=$(ensure_ecr_repo "agent")
  AGENT_TAG="${AGENT_REPO}:latest"

  echo "  Building agent container..."
  # Everything needed is already in the agent/ directory — no external copies needed
  # All middleware (pii.py, content_filter.py, hitl.py, output_guardrail.py) and
  # core utilities (MiddlewareAgent, SemanticCache, PineconeStore) live in this repo
  docker buildx build \
    --platform linux/amd64 \
    --output type=registry \
    --provenance=false \
    --no-cache \
    -t "${AGENT_TAG}" \
    "${AGENT_DIR}"
  echo "  ✅ Agent image pushed: ${AGENT_TAG}"

  echo "  Creating AgentCore Runtime..."
  ENV_JSON="{\"SSM_PREFIX\":\"${SSM_PREFIX}\",\"AWS_REGION\":\"${REGION}\",\"AWS_DEFAULT_REGION\":\"${REGION}\",\"AGENT_ENV\":\"prod\"}"

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

  # If create returned nothing, check if it already exists
  if [ -z "${RUNTIME_ARN}" ] || [ "${RUNTIME_ARN}" = "None" ]; then
    echo "  Checking existing runtimes..."
    RUNTIME_ARN=$(aws bedrock-agentcore-control list-agent-runtimes \
      --region "${REGION}" \
      --query "agentRuntimes[?agentRuntimeName=='${PREFIX//-/_}_clinical_trial'].agentRuntimeArn | [0]" \
      --output text 2>/dev/null || echo "")
  fi

  if [ -z "${RUNTIME_ARN}" ] || [ "${RUNTIME_ARN}" = "None" ]; then
    echo "  ❌ Could not get Runtime ARN — check AWS Console for errors"
    exit 1
  fi

  # Store ARN so Platform can find it
  aws ssm put-parameter \
    --name "${SSM_PREFIX}/agent_runtime_arn" \
    --value "${RUNTIME_ARN}" \
    --type String --overwrite \
    --region "${REGION}"

  echo "  Runtime ARN: ${RUNTIME_ARN}"
  echo ""
  echo "  Agent done ✅"
}


# ── Step 6: Platform + UI (ECS Fargate via Terraform) ─────────────────────

step_platform() {
  echo ""
  echo "► Step 6: Platform + UI (ECS Fargate)"
  ecr_login

  PLATFORM_REPO=$(ensure_ecr_repo "platform")
  PLATFORM_TAG="${PLATFORM_REPO}:latest"
  docker buildx build --platform linux/amd64 --output type=registry \
    --provenance=false --no-cache -t "${PLATFORM_TAG}" "${ROOT}/platform"
  echo "  ✅ Platform: ${PLATFORM_TAG}"

  UI_REPO=$(ensure_ecr_repo "ui")
  UI_TAG="${UI_REPO}:latest"
  docker buildx build --platform linux/amd64 --output type=registry \
    --provenance=false --no-cache -t "${UI_TAG}" "${ROOT}/ui"
  echo "  ✅ UI: ${UI_TAG}"

  cd "${ROOT}/infra"

  # Create S3 backend bucket if it doesn't exist (terraform init fails without it)
  aws s3 mb s3://vs-agentcore-tfstate --region "${REGION}" 2>/dev/null || true
  aws s3api put-bucket-versioning     --bucket vs-agentcore-tfstate     --versioning-configuration Status=Enabled 2>/dev/null || true

  # Pass RDS password to Terraform via env var
  # Set RDS_PASSWORD in .env.prod before running this step
  if [ -z "${RDS_PASSWORD:-}" ]; then
    echo "  ❌ RDS_PASSWORD not set — set it in .env.prod before running platform step"
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
    echo "  ⚠️  NEXT STEPS — required before the agent works:"
    echo ""
    echo "  1. Fill POSTGRES_URL in .env.prod:"
    echo "     POSTGRES_URL=postgresql://postgres:<password>@${RDS_EP}/clinical_agent"
    echo ""
    echo "  2. Push postgres credentials to Secrets Manager:"
    echo "     source .env.prod && ./scripts/deploy.sh secrets"
    echo ""
    echo "  3. Open the UI:"
    echo "     http://${ALB_DNS}"
    echo "  ════════════════════════════════════════════════════"
  fi
}


# ── Step 7: Quick ECS redeploy (no Terraform, no image rebuild) ───────────

step_redeploy() {
  local target="${2:-both}"   # usage: ./deploy.sh redeploy [platform|ui|both]
  echo ""
  echo "► Quick ECS redeploy (force-new-deployment, no Terraform)"

  if [ "${target}" = "platform" ] || [ "${target}" = "both" ]; then
    aws ecs update-service \
      --cluster "${PREFIX}-cluster" \
      --service "${PREFIX}-platform" \
      --force-new-deployment \
      --region "${REGION}" \
      --query "service.deployments[0].{id:id, state:rolloutState}" \
      --output table
    echo "  ✅ vs-agentcore-platform redeployment triggered"
  fi

  if [ "${target}" = "ui" ] || [ "${target}" = "both" ]; then
    aws ecs update-service \
      --cluster "${PREFIX}-cluster" \
      --service "${PREFIX}-ui" \
      --force-new-deployment \
      --region "${REGION}" \
      --query "service.deployments[0].{id:id, state:rolloutState}" \
      --output table
    echo "  ✅ vs-agentcore-ui redeployment triggered"
  fi

  echo ""
  echo "  Watch rollout (Ctrl+C to stop):"
  echo "  aws ecs describe-services --cluster ${PREFIX}-cluster \\"
  echo "    --services ${PREFIX}-platform ${PREFIX}-ui --region ${REGION} \\"
  echo "    --query \"services[*].{name:serviceName,running:runningCount,pending:pendingCount}\" \\"
  echo "    --output table"
}


# ── Main dispatch ──────────────────────────────────────────────────────────

case "${ACTION}" in
  secrets)  step_secrets  ;;
  iam)      step_iam      ;;
  lambdas)  step_lambdas  ;;
  gateway)  step_gateway  ;;
  agent)    step_agent    ;;
  platform) step_platform ;;
  plan)     step_platform ;;  # Terraform plan only
  redeploy) step_redeploy "$@" ;;

  all)
    step_secrets
    step_iam
    step_lambdas
    step_gateway
    step_agent
    step_platform
    echo ""
    echo "================================================"
    echo "✅ Full deployment complete!"
    GW=$(aws ssm get-parameter --name "${SSM_PREFIX}/mcp/gateway_url" \
      --region "${REGION}" --query Value --output text 2>/dev/null || echo "check SSM")
    AR=$(aws ssm get-parameter --name "${SSM_PREFIX}/agent_runtime_arn" \
      --region "${REGION}" --query Value --output text 2>/dev/null || echo "check SSM")
    echo "🔗 MCP Gateway:   ${GW}"
    echo "🤖 Agent ARN:     ${AR}"
    echo "================================================"
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
    echo "Usage: $0 {secrets|iam|lambdas|gateway|agent|platform|redeploy|all|plan|destroy}"
    echo "       redeploy [platform|ui|both]  — force ECS redeploy without rebuild"
    exit 1
    ;;
esac