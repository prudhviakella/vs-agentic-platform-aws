#!/bin/bash
# deploy.sh — vs-agentcore-platform-aws
# Follows the exact pattern from README:
#   1. Create IAM roles (Gateway + Lambda)
#   2. Build + push Lambda container images
#   3. Create Lambda functions
#   4. Create MCP Gateway
#   5. Register Lambda tools as Gateway targets
#   6. Deploy agent to AgentCore Runtime
#   7. Deploy Platform + UI to ECS Fargate
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

# ── Helper functions ───────────────────────────────────────────────────────

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
  echo -n "  Waiting for ${name} to be active..."
  for i in {1..30}; do
    STATE=$(aws lambda get-function --function-name "${name}" \
      --region "${REGION}" --query 'Configuration.State' --output text 2>/dev/null || echo "NotFound")
    [ "${STATE}" = "Active" ] && echo " ✅" && return
    echo -n "."
    sleep 3
  done
  echo " timeout"
}

# ── Step 1: Secrets & SSM params ──────────────────────────────────────────

step_secrets() {
  echo ""
  echo "► Step 1: Secrets & SSM"

  # Push secrets to Secrets Manager
  python3 - << PYEOF
import boto3, json, os, sys

region     = "${REGION}"
ssm_prefix = "${SSM_PREFIX}"
sm         = boto3.client("secretsmanager", region_name=region)
ssm        = boto3.client("ssm", region_name=region)

def put_secret(name, value):
    try:
        sm.create_secret(Name=name, SecretString=json.dumps(value))
    except sm.exceptions.ResourceExistsException:
        sm.update_secret(SecretId=name, SecretString=json.dumps(value))
    print(f"  ✅ Secret: {name}")

def put_param(name, value, secure=False):
    ssm.put_parameter(
        Name=name, Value=value,
        Type="SecureString" if secure else "String",
        Overwrite=True
    )
    print(f"  ✅ SSM: {name}")

# Required env vars
required = ["OPENAI_API_KEY","PINECONE_API_KEY","NEO4J_URI","NEO4J_USER",
            "NEO4J_PASSWORD","POSTGRES_URL","PLATFORM_API_KEY",
            "BEDROCK_PROMPT_ID","BEDROCK_PROMPT_VERSION"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    print(f"ERROR: Missing env vars: {missing}")
    sys.exit(1)

put_secret(f"{ssm_prefix}/openai",    {"api_key": os.environ["OPENAI_API_KEY"]})
put_secret(f"{ssm_prefix}/pinecone",  {"api_key": os.environ["PINECONE_API_KEY"]})
put_secret(f"{ssm_prefix}/neo4j",     {"uri": os.environ["NEO4J_URI"], "user": os.environ["NEO4J_USER"], "password": os.environ["NEO4J_PASSWORD"]})
put_secret(f"{ssm_prefix}/postgres",  {"connection_string": os.environ["POSTGRES_URL"]})
put_secret(f"{ssm_prefix}/platform_api_key", {"api_key": os.environ["PLATFORM_API_KEY"]})

put_param(f"{ssm_prefix}/pinecone/clinical_trials_index", os.environ.get("CLINICAL_TRIALS_INDEX","clinical-trials-index"))
put_param(f"{ssm_prefix}/pinecone/cache_index_name",      os.environ.get("PINECONE_INDEX_NAME","clinical-agent"))
put_param(f"{ssm_prefix}/dynamodb/trace_table_name",      "${PREFIX}-traces")
put_param(f"{ssm_prefix}/bedrock/prompt_id",              os.environ["BEDROCK_PROMPT_ID"])
put_param(f"{ssm_prefix}/bedrock/prompt_version",         os.environ["BEDROCK_PROMPT_VERSION"])

print("  Secrets done ✅")
PYEOF
}

# ── Step 2: IAM Roles ──────────────────────────────────────────────────────

step_iam() {
  echo ""
  echo "► Step 2: IAM roles"

  # Lambda execution role
  cat > /tmp/lambda-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-lambda-mcp" \
    --assume-role-policy-document file:///tmp/lambda-trust.json 2>/dev/null || true
  aws iam attach-role-policy --role-name "${PREFIX}-lambda-mcp" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true

  cat > /tmp/lambda-secrets-policy.json << POLICY
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["secretsmanager:GetSecretValue","ssm:GetParameter","kms:Decrypt"],"Resource":"*"}
]}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-lambda-mcp" \
    --policy-name SecretsAccess --policy-document file:///tmp/lambda-secrets-policy.json

  # MCP Gateway role (from README pattern)
  cat > /tmp/gateway-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Sid":"GatewayAssumeRolePolicy","Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-gateway-role" \
    --assume-role-policy-document file:///tmp/gateway-trust.json 2>/dev/null || true

  cat > /tmp/invoke-lambda-policy.json << POLICY
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["lambda:InvokeFunction"],"Resource":[
    "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-search-tool",
    "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-graph-tool",
    "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-hitl-tool",
    "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-summariser-tool"
  ]}
]}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-gateway-role" \
    --policy-name AllowInvokeLambdaTools \
    --policy-document file:///tmp/invoke-lambda-policy.json

  cat > /tmp/cw-logs.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}]}
JSON
  aws iam put-role-policy --role-name "${PREFIX}-gateway-role" \
    --policy-name AllowCloudWatchLogs \
    --policy-document file:///tmp/cw-logs.json

  # AgentCore Runtime execution role
  cat > /tmp/agentcore-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-agent-role" \
    --assume-role-policy-document file:///tmp/agentcore-trust.json 2>/dev/null || true

  cat > /tmp/agentcore-policy.json << POLICY
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["secretsmanager:GetSecretValue","ssm:GetParameter","kms:Decrypt"],"Resource":"*"},
  {"Effect":"Allow","Action":["bedrock-agentcore:InvokeGateway"],"Resource":"*"},
  {"Effect":"Allow","Action":["bedrock:GetPrompt","bedrock:ListPrompts"],"Resource":"*"},
  {"Effect":"Allow","Action":["dynamodb:PutItem","dynamodb:GetItem","dynamodb:UpdateItem"],"Resource":"*"},
  {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}
]}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-agent-role" \
    --policy-name AgentCorePolicy --policy-document file:///tmp/agentcore-policy.json

  echo "  IAM done ✅"
}

# ── Step 3: Lambda tools ───────────────────────────────────────────────────

step_lambdas() {
  echo ""
  echo "► Step 3: Lambda MCP tools"
  ecr_login

  LAMBDA_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-lambda-mcp"

  for tool in search graph hitl summariser; do
    echo "  Building ${tool}_lambda..."
    REPO=$(ensure_ecr_repo "${tool}-tool")
    TAG="${REPO}:latest"

    docker buildx build \
      --platform linux/amd64 \
      --output type=docker \
      --provenance=false \
      --no-cache \
      -t "${TAG}" \
      "${ROOT}/mcp_tools/${tool}_lambda"
    docker push "${TAG}"

    FUNC="${PREFIX}-${tool}-tool"
    ENV_VARS="Variables={SSM_PREFIX=${SSM_PREFIX},AWS_REGION=${REGION}}"

    if aws lambda get-function --function-name "${FUNC}" --region "${REGION}" &>/dev/null; then
      echo "  Updating ${FUNC}..."
      aws lambda update-function-code \
        --function-name "${FUNC}" \
        --image-uri "${TAG}" \
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
  done

  echo "  Lambdas done ✅"
}

# ── Step 4: MCP Gateway ────────────────────────────────────────────────────

step_gateway() {
  echo ""
  echo "► Step 4: MCP Gateway (bedrock-agentcore-control)"

  GATEWAY_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-gateway-role"

  # Create Gateway (AWS_IAM auth — matches README)
  echo "  Creating gateway..."
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

  # Wait for ACTIVE
  echo -n "  Waiting for gateway ACTIVE..."
  for i in {1..24}; do
    STATUS=$(aws bedrock-agentcore-control get-gateway \
      --region "${REGION}" \
      --gateway-identifier "${GATEWAY_ID}" \
      --query 'status' --output text 2>/dev/null || echo "UNKNOWN")
    [ "${STATUS}" = "ACTIVE" ] && echo " ✅" && break
    echo -n "."
    sleep 5
  done

  # Register tool targets (matches README pattern exactly)
  register_target() {
    local target_name="$1"
    local tool_name="$2"
    local tool_desc="$3"
    local lambda_func="$4"
    local schema="$5"

    echo "  Registering ${target_name}..."
    aws bedrock-agentcore-control create-gateway-target \
      --region "${REGION}" \
      --gateway-identifier "${GATEWAY_ID}" \
      --name "${target_name}" \
      --description "${tool_desc}" \
      --target-configuration "{\"mcp\":{\"lambda\":{\"lambdaArn\":\"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${lambda_func}\",\"toolSchema\":{\"inlinePayload\":[{\"name\":\"${tool_name}\",\"description\":\"${tool_desc}\",\"inputSchema\":${schema}}]}}}}" \
      --credential-provider-configurations "[{\"credentialProviderType\":\"GATEWAY_IAM_ROLE\"}]" > /dev/null
    echo "  ✅ ${target_name}"
  }

  register_target "tool-search" "search_tool" \
    "Semantic search over clinical trials knowledge base. Use FIRST for any evidence query." \
    "${PREFIX}-search-tool" \
    '{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"top_k":{"type":"integer","description":"Number of results (default 5)"}},"required":["query"]}'

  register_target "tool-graph" "graph_tool" \
    "Execute Cypher query against Neo4j clinical trials graph." \
    "${PREFIX}-graph-tool" \
    '{"type":"object","properties":{"cypher":{"type":"string","description":"Read-only Cypher query"}},"required":["cypher"]}'

  register_target "tool-hitl" "ask_user_input" \
    "Ask user clarifying question. ALWAYS call search_tool first." \
    "${PREFIX}-hitl-tool" \
    '{"type":"object","properties":{"question":{"type":"string"},"options":{"type":"array","items":{"type":"string"}},"allow_freetext":{"type":"boolean"},"user_answer":{"type":"string"}},"required":[]}'

  register_target "tool-summariser" "summariser_tool" \
    "Synthesise multiple retrieved chunks into a concise summary." \
    "${PREFIX}-summariser-tool" \
    '{"type":"object","properties":{"chunks":{"type":"array","items":{"type":"string"}},"query":{"type":"string"}},"required":["chunks"]}'

  # Store gateway URL in SSM
  aws ssm put-parameter \
    --name "${SSM_PREFIX}/mcp/gateway_url" \
    --value "${GATEWAY_URL}" \
    --type String --overwrite \
    --region "${REGION}"

  echo "  Gateway done ✅  URL: ${GATEWAY_URL}"
}

# ── Step 5: AgentCore Runtime ──────────────────────────────────────────────

step_agent() {
  echo ""
  echo "► Step 5: AgentCore Runtime"
  ecr_login

  AGENT_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-agent-role"
  AGENT_REPO=$(ensure_ecr_repo "agent")
  AGENT_TAG="${AGENT_REPO}:latest"

  # Copy shared middleware packages before build
  echo "  Copying shared packages..."
  AGENT_DIR="${ROOT}/agent"
  cp -r "${ROOT}/../vs-agentic-platform/vs-agent-core/core" "${AGENT_DIR}/core" 2>/dev/null || \
    echo "  WARNING: vs-agent-core/core not found — agent middleware will fail at runtime"
  cp -r "${ROOT}/../vs-agentic-platform/clinical_trial_agent/agent/middleware" "${AGENT_DIR}/agent_middleware" 2>/dev/null || \
    echo "  WARNING: clinical_trial_agent/middleware not found"

  echo "  Building agent container..."
  docker buildx build \
    --platform linux/amd64 \
    --output type=docker \
    --provenance=false \
    --no-cache \
    -t "${AGENT_TAG}" \
    "${AGENT_DIR}"
  docker push "${AGENT_TAG}"

  # Deploy to AgentCore Runtime via AWS CLI
  echo "  Creating AgentCore Runtime..."
  ENV_VARS="SSM_PREFIX=${SSM_PREFIX},AWS_REGION=${REGION}"

  RUNTIME_RESPONSE=$(aws bedrock-agentcore-control create-agent-runtime \
    --region "${REGION}" \
    --agent-runtime-name "${PREFIX}-clinical-trial" \
    --description "Clinical Trial Research Agent" \
    --agent-runtime-artifact "{\"containerConfiguration\":{\"containerUri\":\"${AGENT_TAG}\"}}" \
    --execution-role-arn "${AGENT_ROLE}" \
    --network-configuration "{\"networkMode\":\"PUBLIC\"}" \
    --environment-variables "${ENV_VARS}" 2>/dev/null || echo "{}")

  RUNTIME_ARN=$(echo "${RUNTIME_RESPONSE}" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('agentRuntimeArn',''))" 2>/dev/null || echo "")

  if [ -z "${RUNTIME_ARN}" ]; then
    # Try to get existing
    RUNTIME_ARN=$(aws bedrock-agentcore-control list-agent-runtimes \
      --region "${REGION}" \
      --query "agentRuntimes[?agentRuntimeName=='${PREFIX}-clinical-trial'].agentRuntimeArn | [0]" \
      --output text 2>/dev/null || echo "")
  fi

  echo "  Runtime ARN: ${RUNTIME_ARN}"

  # Store ARN in SSM
  if [ -n "${RUNTIME_ARN}" ] && [ "${RUNTIME_ARN}" != "None" ]; then
    aws ssm put-parameter \
      --name "${SSM_PREFIX}/agent_runtime_arn" \
      --value "${RUNTIME_ARN}" \
      --type String --overwrite \
      --region "${REGION}"
    echo "  Agent done ✅"
  else
    echo "  WARNING: Could not get Runtime ARN. Check AWS Console."
  fi
}

# ── Step 6: Platform + UI (ECS Fargate via Terraform) ─────────────────────

step_platform() {
  echo ""
  echo "► Step 6: Platform + UI (ECS Fargate)"
  ecr_login

  # Build + push platform image
  PLATFORM_REPO=$(ensure_ecr_repo "platform")
  PLATFORM_TAG="${PLATFORM_REPO}:latest"
  docker buildx build --platform linux/amd64 --output type=docker \
    --provenance=false --no-cache -t "${PLATFORM_TAG}" "${ROOT}/platform"
  docker push "${PLATFORM_TAG}"
  echo "  ✅ Platform: ${PLATFORM_TAG}"

  # Build + push UI image
  UI_REPO=$(ensure_ecr_repo "ui")
  UI_TAG="${UI_REPO}:latest"
  docker buildx build --platform linux/amd64 --output type=docker \
    --provenance=false --no-cache -t "${UI_TAG}" "${ROOT}/ui"
  docker push "${UI_TAG}"
  echo "  ✅ UI: ${UI_TAG}"

  # Terraform
  cd "${ROOT}/infra"
  terraform init -upgrade -input=false

  if [ "${ACTION}" = "plan" ]; then
    terraform plan \
      -var="platform_image_uri=${PLATFORM_TAG}" \
      -var="ui_image_uri=${UI_TAG}" \
      -var="aws_region=${REGION}" \
      -var="ssm_prefix=${SSM_PREFIX}"
  else
    terraform apply -auto-approve -input=false \
      -var="platform_image_uri=${PLATFORM_TAG}" \
      -var="ui_image_uri=${UI_TAG}" \
      -var="aws_region=${REGION}" \
      -var="ssm_prefix=${SSM_PREFIX}"

    ALB_DNS=$(terraform output -raw alb_dns 2>/dev/null || echo "check-terraform-output")
    echo "  Platform done ✅  ALB: http://${ALB_DNS}"
  fi
}

# ── Main ───────────────────────────────────────────────────────────────────

if [ "${ACTION}" = "plan" ]; then
  step_platform

elif [ "${ACTION}" = "apply" ]; then
  step_secrets
  step_iam
  step_lambdas
  step_gateway
  step_agent
  step_platform

  echo ""
  echo "================================================"
  echo "✅ Deployment complete!"
  echo ""
  GATEWAY_URL=$(aws ssm get-parameter --name "${SSM_PREFIX}/mcp/gateway_url" \
    --region "${REGION}" --query Value --output text 2>/dev/null || echo "check SSM")
  AGENT_ARN=$(aws ssm get-parameter --name "${SSM_PREFIX}/agent_runtime_arn" \
    --region "${REGION}" --query Value --output text 2>/dev/null || echo "check SSM")
  echo "🔗 MCP Gateway:   ${GATEWAY_URL}"
  echo "🤖 Agent ARN:     ${AGENT_ARN}"
  echo "================================================"

elif [ "${ACTION}" = "destroy" ]; then
  echo "⚠️  Destroying all resources..."
  cd "${ROOT}/infra" && terraform destroy -auto-approve \
    -var="platform_image_uri=placeholder" \
    -var="ui_image_uri=placeholder" \
    -var="aws_region=${REGION}" \
    -var="ssm_prefix=${SSM_PREFIX}"
fi
