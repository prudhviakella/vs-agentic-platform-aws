#!/bin/bash
# delete-all-paramstore.sh
# Deletes ALL SSM Parameter Store entries in the current AWS region

set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-south-1}"
DRY_RUN="${DRY_RUN:-false}"
PATH_PREFIX="${PATH_PREFIX:-/}"   # Set to e.g. "/vs-agentcore/" to scope deletion

echo "🔍 Fetching all SSM parameters in region: $AWS_REGION (prefix: $PATH_PREFIX)"

# Collect all parameter names (paginated)
ALL_PARAMS=()
NEXT_TOKEN=""

while true; do
  if [[ -z "$NEXT_TOKEN" ]]; then
    RESPONSE=$(aws ssm describe-parameters \
      --region "$AWS_REGION" \
      --parameter-filters "Key=Path,Option=Recursive,Values=$PATH_PREFIX" \
      --max-items 50 \
      --output json)
  else
    RESPONSE=$(aws ssm describe-parameters \
      --region "$AWS_REGION" \
      --parameter-filters "Key=Path,Option=Recursive,Values=$PATH_PREFIX" \
      --max-items 50 \
      --starting-token "$NEXT_TOKEN" \
      --output json)
  fi

  # Extract names from this page
  PAGE_PARAMS=$(echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for p in data.get('Parameters', []):
    print(p['Name'])
")

  while IFS= read -r name; do
    [[ -n "$name" ]] && ALL_PARAMS+=("$name")
  done <<< "$PAGE_PARAMS"

  # Check for next page
  NEXT_TOKEN=$(echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('NextToken', ''))
")

  [[ -z "$NEXT_TOKEN" ]] && break
done

TOTAL=${#ALL_PARAMS[@]}

if [[ $TOTAL -eq 0 ]]; then
  echo "✅ No parameters found under prefix '$PATH_PREFIX'. Nothing to delete."
  exit 0
fi

echo "🗑️  Found $TOTAL parameter(s) to delete"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Would delete the following parameters:"
  for p in "${ALL_PARAMS[@]}"; do
    echo "  - $p"
  done
  echo ""
  echo "[DRY RUN] No changes made."
  exit 0
fi

# SSM delete-parameters allows max 10 per call — batch accordingly
DELETED=0
FAILED=0

for ((i = 0; i < TOTAL; i += 10)); do
  CHUNK=("${ALL_PARAMS[@]:i:10}")

  RESULT=$(aws ssm delete-parameters \
    --region "$AWS_REGION" \
    --names "${CHUNK[@]}" \
    --output json)

  DEL_COUNT=$(echo "$RESULT" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('DeletedParameters', [])))")
  INV_COUNT=$(echo "$RESULT" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('InvalidParameters', [])))")

  DELETED=$((DELETED + DEL_COUNT))
  FAILED=$((FAILED + INV_COUNT))

  if [[ "$INV_COUNT" -gt 0 ]]; then
    echo "  ⚠️  Invalid (not found): $(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('InvalidParameters', []))")"
  fi

  echo "  ✅ Deleted batch $((i/10 + 1)): $DEL_COUNT param(s)"
done

echo ""
echo "🎉 Done. Deleted: $DELETED | Failed/Invalid: $FAILED"