#!/usr/bin/env bash
# Push the demo page to S3 and invalidate CloudFront.
#
#   bash infra/site/deploy.sh
#
# The bucket and the distribution come from terraform rather than from a
# constant here, so this cannot be pointed at a stale stack.
#
# WHY THE FRAMES ARE COPIED AND NOT COMMITTED TWICE
#
# The field viewer needs the twelve rendered frames, which eval/render_session.py
# already writes to docs/figures/session/ and which are already in git. Keeping a
# second copy under infra/site/fields/ would be 5.9 MB of duplicate binary in the
# repository and, worse, two copies that can disagree -- the page would then show
# frames that no longer match the numbers the report quotes. They are copied at
# deploy time instead, from the one committed set.

set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo/infra/terraform"
BUCKET=$(terraform output -raw site_bucket)
URL=$(terraform output -raw site_url)
DIST=$(aws cloudfront list-distributions \
  --query "DistributionList.Items[?contains(DomainName, '${URL#https://}')].Id | [0]" \
  --output text)
cd "$repo"

FRAMES="docs/figures/session"
[ -d "$FRAMES" ] || { echo "no frames in $FRAMES -- run eval/render_session.py first"; exit 1; }
n=$(ls "$FRAMES"/field-*.png 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -gt 0 ] || { echo "no field-*.png in $FRAMES"; exit 1; }

echo "==> $BUCKET  ($n frames)"
aws s3 cp infra/site/index.html "s3://$BUCKET/index.html" \
  --content-type text/html --cache-control 'max-age=60'
aws s3 cp infra/site/sample-session.json "s3://$BUCKET/sample-session.json" \
  --content-type application/json --cache-control 'max-age=60'
aws s3 sync "$FRAMES" "s3://$BUCKET/fields" \
  --exclude '*' --include 'field-*.png' \
  --content-type image/png --cache-control 'max-age=86400' --delete

echo "==> invalidating $DIST"
aws cloudfront create-invalidation --distribution-id "$DIST" --paths '/*' \
  --query 'Invalidation.Status' --output text
echo "==> $URL"
