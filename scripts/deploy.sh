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
    --region "${REGION}" 2>/dev/null || true
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

  python3 - << PYEOF
import boto3, json, os, sys
from urllib.parse import urlparse

region     = "${REGION}"
ssm_prefix = "${SSM_PREFIX}"
sm         = boto3.client("secretsmanager", region_name=region)
ssm_client = boto3.client("ssm",            region_name=region)

def put_secret(name, value):
    """Create or update a Secrets Manager secret."""
    try:
        sm.create_secret(Name=name, SecretString=json.dumps(value))
        print(f"  ✅ Created secret: {name}")
    except sm.exceptions.ResourceExistsException:
        sm.update_secret(SecretId=name, SecretString=json.dumps(value))
        print(f"  ✅ Updated secret: {name}")

def put_param(name, value, secure=False):
    """Create or overwrite an SSM parameter."""
    ssm_client.put_parameter(
        Name=name, Value=value,
        Type="SecureString" if secure else "String",
        Overwrite=True,
    )
    print(f"  ✅ SSM param: {name}")

# Verify all required env vars are set before doing anything
required = [
    "OPENAI_API_KEY", "PINECONE_API_KEY",
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
    "POSTGRES_URL", "PLATFORM_API_KEY",
    "BEDROCK_PROMPT_ID", "BEDROCK_PROMPT_VERSION",
]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    print(f"ERROR: Missing required env vars: {missing}")
    print("Run: source .env.prod")
    sys.exit(1)

# ── API keys ───────────────────────────────────────────────────────────
put_secret(f"{ssm_prefix}/openai",   {"api_key": os.environ["OPENAI_API_KEY"]})
put_secret(f"{ssm_prefix}/pinecone", {"api_key": os.environ["PINECONE_API_KEY"]})

# ── Neo4j ──────────────────────────────────────────────────────────────
put_secret(f"{ssm_prefix}/neo4j", {
    "uri":      os.environ["NEO4J_URI"],
    "user":     os.environ["NEO4J_USER"],
    "password": os.environ["NEO4J_PASSWORD"],
})

# ── Postgres — stored as INDIVIDUAL FIELDS (NOT a connection_string) ───
# graph.py builds the URL from these via init_postgres_url():
#   f"postgresql://{username}:{quote_plus(password)}@{host}:{port}/{dbname}"
# Storing as connection_string causes a KeyError at runtime.
pg = urlparse(os.environ["POSTGRES_URL"])
put_secret(f"{ssm_prefix}/postgres", {
    "username": pg.username,
    "password": pg.password,
    "host":     pg.hostname,
    "port":     str(pg.port or 5432),
    "dbname":   pg.path.lstrip("/"),
})

# ── Platform auth ──────────────────────────────────────────────────────
put_secret(f"{ssm_prefix}/platform_api_key", {"api_key": os.environ["PLATFORM_API_KEY"]})

# ── SSM non-secret config ──────────────────────────────────────────────
put_param(f"{ssm_prefix}/pinecone/clinical_trials_index",
          os.environ.get("CLINICAL_TRIALS_INDEX", "clinical-trials-index"))
put_param(f"{ssm_prefix}/pinecone/cache_index_name",
          os.environ.get("PINECONE_INDEX_NAME", "clinical-agent"))
put_param(f"{ssm_prefix}/dynamodb/trace_table_name", "${PREFIX}-traces")
put_param(f"{ssm_prefix}/bedrock/prompt_id",      os.environ["BEDROCK_PROMPT_ID"])
put_param(f"{ssm_prefix}/bedrock/prompt_version", os.environ["BEDROCK_PROMPT_VERSION"])

print("")
print("  All secrets and params written ✅")
PYEOF
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
  {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}
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
  {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}
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
      --output type=docker \
      --provenance=false \
      --no-cache \
      -t "${TAG}" \
      "${ROOT}/mcp_tools/${tool}_lambda"
    docker push "${TAG}"

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
    "Semantic search over clinical trials. Use FIRST for any evidence query." \
    "${PREFIX}-search-tool" \
    '{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer"}},"required":["query"]}'

  register_target "tool-graph" "graph_tool" \
    "Cypher query against Neo4j clinical trials graph." \
    "${PREFIX}-graph-tool" \
    '{"type":"object","properties":{"cypher":{"type":"string"}},"required":["cypher"]}'

  register_target "tool-hitl" "ask_user_input" \
    "Ask user a clarifying question. Always call search_tool first." \
    "${PREFIX}-hitl-tool" \
    '{"type":"object","properties":{"question":{"type":"string"},"options":{"type":"array","items":{"type":"string"}},"allow_freetext":{"type":"boolean"},"user_answer":{"type":"string"}},"required":[]}'

  register_target "tool-summariser" "summariser_tool" \
    "Synthesise multiple retrieved chunks into a concise summary." \
    "${PREFIX}-summariser-tool" \
    '{"type":"object","properties":{"chunks":{"type":"array","items":{"type":"string"}},"query":{"type":"string"}},"required":["chunks"]}'

  # Store gateway URL so agent container can find it
  aws ssm put-parameter \
    --name "${SSM_PREFIX}/mcp/gateway_url" \
    --value "${GATEWAY_URL}" \
    --type String --overwrite \
    --region "${REGION}"

  echo ""
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
    --output type=docker \
    --provenance=false \
    --no-cache \
    -t "${AGENT_TAG}" \
    "${AGENT_DIR}"
  docker push "${AGENT_TAG}"
  echo "  ✅ Agent image pushed: ${AGENT_TAG}"

  echo "  Creating AgentCore Runtime..."
  ENV_JSON="{\"SSM_PREFIX\":\"${SSM_PREFIX}\",\"AWS_REGION\":\"${REGION}\"}"

  RUNTIME_RESPONSE=$(aws bedrock-agentcore-control create-agent-runtime \
    --region "${REGION}" \
    --agent-runtime-name "${PREFIX}-clinical-trial" \
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
      --query "agentRuntimes[?agentRuntimeName=='${PREFIX}-clinical-trial'].agentRuntimeArn | [0]" \
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
  docker buildx build --platform linux/amd64 --output type=docker \
    --provenance=false --no-cache -t "${PLATFORM_TAG}" "${ROOT}/platform"
  docker push "${PLATFORM_TAG}"
  echo "  ✅ Platform: ${PLATFORM_TAG}"

  UI_REPO=$(ensure_ecr_repo "ui")
  UI_TAG="${UI_REPO}:latest"
  docker buildx build --platform linux/amd64 --output type=docker \
    --provenance=false --no-cache -t "${UI_TAG}" "${ROOT}/ui"
  docker push "${UI_TAG}"
  echo "  ✅ UI: ${UI_TAG}"

  cd "${ROOT}/infra"
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
    echo "  ⚠️  Now fill POSTGRES_URL in .env.prod with:"
    echo "     postgresql://postgres:PASSWORD@${RDS_EP}/clinical_agent"
    echo "  Then re-run: ./scripts/deploy.sh secrets"
  fi
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
    echo "Usage: $0 {secrets|iam|lambdas|gateway|agent|platform|all|plan|destroy}"
    exit 1
    ;;
esac
