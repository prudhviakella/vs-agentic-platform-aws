#!/bin/bash
# delete-all-ecr-images.sh
# Deletes ALL images from ALL ECR repositories in the current AWS region

set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-south-1}"  # Default to Mumbai; override via env var
DRY_RUN="${DRY_RUN:-false}"

echo "🔍 Fetching all ECR repositories in region: $AWS_REGION"

REPOS=$(aws ecr describe-repositories \
  --region "$AWS_REGION" \
  --query "repositories[*].repositoryName" \
  --output text)

if [[ -z "$REPOS" ]]; then
  echo "✅ No repositories found. Nothing to delete."
  exit 0
fi

for REPO in $REPOS; do
  echo ""
  echo "📦 Repository: $REPO"

  # Get all image digests in the repo
  IMAGE_IDS=$(aws ecr list-images \
    --region "$AWS_REGION" \
    --repository-name "$REPO" \
    --query "imageIds[*]" \
    --output json)

  COUNT=$(echo "$IMAGE_IDS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")

  if [[ "$COUNT" -eq 0 ]]; then
    echo "  ⚠️  No images found, skipping."
    continue
  fi

  echo "  🗑️  Found $COUNT image(s)"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [DRY RUN] Would delete $COUNT image(s) from $REPO"
    continue
  fi

  # Batch delete (ECR allows max 100 per call; loop handles larger repos)
  echo "$IMAGE_IDS" | python3 -c "
import sys, json
ids = json.load(sys.stdin)
# Split into chunks of 100
for i in range(0, len(ids), 100):
    chunk = ids[i:i+100]
    print(json.dumps(chunk))
" | while read -r CHUNK; do
    aws ecr batch-delete-image \
      --region "$AWS_REGION" \
      --repository-name "$REPO" \
      --image-ids "$CHUNK" \
      --output table
  done

  echo "  ✅ Deleted all images from: $REPO"
done

echo ""
echo "🎉 Done. All ECR images deleted."