#!/usr/bin/env bash
# Put one more patient on the live demo.
#
#   bash infra/site/add_patient.sh TF41_UT5
#   bash infra/site/add_patient.sh TF41_UT5 10        # cap the fields uploaded
#
# The demo page takes any session id, so the only thing that limits it is which
# slides have actually been read. This uploads a patient's fields and starts a
# reading through the public API, which is the same path a judge would use.
#
# WHY THE SESSION ID IS THE S3 PREFIX
#
# `read_field.field_key` builds `<session_id>/field-%04d.jpg`, so the id is not
# a label: it is where the job looks. Upload under one name and start under
# another and every field 404s.
#
# The EventBridge auto-start rule only matches the `slides/` prefix, so these
# uploads start nothing by themselves. The POST below is the trigger, which
# means one execution per patient instead of one per file.

set -euo pipefail
PATIENT="${1:?usage: add_patient.sh <patient-dir-under-data/raw> [max-fields]}"
MAX="${2:-14}"
REGION="${AWS_REGION:-us-east-1}"
API="${API:-https://fhemydiqba.execute-api.us-east-1.amazonaws.com}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$repo/data/raw/$PATIENT"
BUCKET="$(cd "$repo/infra/terraform" && terraform output -raw uploads_bucket)"

[ -d "$SRC" ] || { echo "no such patient: $SRC"; exit 1; }

echo "==> $PATIENT -> s3://$BUCKET/$PATIENT/"

# Sorted, because the capture timestamps are the reading order and the agent's
# trace only makes sense if field 3 really is the fourth one the camera saw.
# A while-read loop and not `mapfile`: macOS ships bash 3.2, which does not
# have it. Same family of trap as the empty-array expansion in eval/chain.sh.
N=0
while IFS= read -r f; do
  aws s3 cp "$f" "s3://$BUCKET/$PATIENT/$(printf 'field-%04d.jpg' "$N")" \
    --region "$REGION" --only-show-errors
  N=$((N + 1))
done <<EOF
$(ls "$SRC"/*.jpg 2>/dev/null | sort | head -n "$MAX")
EOF
[ "$N" -gt 0 ] || { echo "no .jpg fields in $SRC"; exit 1; }
echo "    $N fields uploaded" 

echo "==> starting a reading through the API"
curl -sSf -X POST "$API/sessions" -H 'content-type: application/json' \
  -d "{\"key\": \"$PATIENT/\", \"session_id\": \"$PATIENT\"}"
echo
echo "==> watch it: $API/sessions/$PATIENT/trace"
echo "    on the page: https://d1k2o3z37c1maf.cloudfront.net/?api=$API&session=$PATIENT"
