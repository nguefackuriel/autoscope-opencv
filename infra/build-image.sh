#!/usr/bin/env bash
# Build the pipeline image natively on Graviton and push it to ECR.
#
#   bash infra/build-image.sh
#
# WHY ON AN INSTANCE AND NOT ON A LAPTOP
#
# AWS Batch runs this image on Graviton4, so it has to be linux/arm64. Both of
# our laptops are x86: `docker buildx --platform linux/arm64` works there but
# runs every layer under QEMU, which turns a two-minute build into twenty and
# has a habit of failing inside pip on native extensions. A c8g instance builds
# it natively in a couple of minutes for about four cents, and does not care
# whether Docker Desktop is running.
#
# The build context travels through S3 rather than a git clone: the repository
# is private, and a deploy token on a throwaway instance is a token too many.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="${ACCOUNT:-617847240757}"
REPO="${REPO:-autoscope-pipeline}"
TAG="${TAG:-latest}"
BUCKET="${BUCKET:-autoscope-bench-617847240757}"
TYPE="${TYPE:-c8g.4xlarge}"
PROFILE_NAME="autoscope-bench-ssm"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
ECR="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO"

echo "==> packing the build context"
tar -czf /tmp/build-context.tar.gz -C "$repo_root" \
  Dockerfile requirements.txt pyproject.toml src models tests bench \
  $(cd "$repo_root" && ls eval/*.py eval/chain.sh)
aws s3 cp /tmp/build-context.tar.gz "s3://$BUCKET/build-context.tar.gz" --only-show-errors
echo "    $(du -h /tmp/build-context.tar.gz | cut -f1) uploaded"

AMI=$(aws ssm get-parameters --region "$REGION" \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
  --query 'Parameters[0].Value' --output text)
SUBNET=$(aws ec2 describe-subnets --region "$REGION" \
  --filters Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text)

ID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --subnet-id "$SUBNET" \
  --iam-instance-profile "Name=$PROFILE_NAME" \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=autoscope-image-build},{Key=Project,Value=autoscope}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "==> building on $ID ($TYPE, native arm64)"

cleanup() {
  echo "==> terminating $ID"
  aws ec2 terminate-instances --instance-ids "$ID" --region "$REGION" \
    --query 'TerminatingInstances[0].CurrentState.Name' --output text || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  online=$(aws ssm describe-instance-information --region "$REGION" \
    --filters "Key=InstanceIds,Values=$ID" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || echo None)
  [ "$online" = "Online" ] && break
  sleep 10
done
[ "${online:-}" = "Online" ] || { echo "SSM never saw the instance"; exit 1; }

CMD=$(cat <<EOF
set -eux
dnf install -y docker >/dev/null
systemctl start docker
mkdir -p /opt/build && cd /opt/build
aws s3 cp s3://$BUCKET/build-context.tar.gz .
tar -xzf build-context.tar.gz
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker build -t $ECR:$TAG .
docker push $ECR:$TAG
echo '--- DIGEST ---'
docker inspect --format='{{index .RepoDigests 0}}' $ECR:$TAG
EOF
)

PARAMS="$(mktemp -t ssmparams).json"
python3 - "$CMD" > "$PARAMS" <<'PYEOF'
import json, sys
print(json.dumps({"commands": sys.argv[1].split("\n")}))
PYEOF

CMD_ID=$(aws ssm send-command --region "$REGION" --instance-ids "$ID" \
  --document-name AWS-RunShellScript --comment "autoscope image build" \
  --parameters "file://$PARAMS" --timeout-seconds 3600 \
  --query 'Command.CommandId' --output text)
rm -f "$PARAMS"

echo -n "    "
for _ in $(seq 1 160); do          # 160 x 15 s = 40 min
  STATUS=$(aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$ID" \
    --region "$REGION" --query 'Status' --output text 2>/dev/null || echo Pending)
  case "$STATUS" in Success|Failed|Cancelled|TimedOut) break ;; esac
  echo -n "."
  sleep 15
done
echo " $STATUS"

if [ "$STATUS" != "Success" ]; then
  aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text | tail -40
  exit 1
fi

aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text \
  | sed -n '/--- DIGEST ---/,$p' | tail -n +2
echo "==> pushed $ECR:$TAG"
