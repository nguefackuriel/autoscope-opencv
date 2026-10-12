#!/usr/bin/env bash
# Build the Lambda packages. Run this before the first `terraform apply` and
# again whenever src/autoscope/agent/moves.py changes.
#
#   bash infra/lambda/build.sh
#
# Two things happen:
#   1. moves.py is copied into the agent package, so the deployed prompt and
#      the one we measure offline are the same file and cannot drift.
#   2. the anthropic SDK is installed into a layer for arm64. The Lambdas run
#      on Graviton like everything else here.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

echo "==> moves.py into the agent package"
cp "$repo/src/autoscope/agent/moves.py" "$here/agent/moves.py"

echo "==> the anthropic layer, for python3.12 on arm64"
layer="$here/layers"
rm -rf "$layer/python"
mkdir -p "$layer/python"

# --platform/--only-binary is what makes this an arm64 layer built from an x86
# laptop. Without it pip silently builds for the host and the Lambda dies on
# import with an ELF error that says nothing useful.
python3 -m pip install \
  --target "$layer/python" \
  --platform manylinux2014_aarch64 \
  --implementation cp \
  --python-version 3.12 \
  --only-binary=:all: \
  --upgrade \
  'anthropic[bedrock]'

# boto3 is already in the Lambda runtime; shipping it again costs 15 MB and
# risks a version that disagrees with the one underneath.
rm -rf "$layer/python"/boto3* "$layer/python"/botocore* "$layer/python"/s3transfer*

# No zip here: Terraform packs layers/ itself (agent.tf), so that the archive
# contains python/ as a directory rather than its contents.
echo "==> $(du -sh "$layer/python" | cut -f1) in $layer/python"

echo
echo "Done. Now: cd infra/terraform && terraform apply"
