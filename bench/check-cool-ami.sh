#!/usr/bin/env bash
# Has the COOL Marketplace subscription come through yet?
#
#   bash bench/check-cool-ami.sh
#
# Visibility is not subscription. `describe-images --owners aws-marketplace`
# lists every public Marketplace AMI whether or not this account may launch it,
# so the only honest test is a dry-run launch: an unsubscribed product answers
# OptInRequired and hands back the exact product URL to accept terms on.

set -uo pipefail
REGION="${AWS_REGION:-us-east-1}"
TYPE="${TYPE:-c8g.8xlarge}"

echo "==> COOL AMIs visible in $REGION"
aws ec2 describe-images --region "$REGION" --owners aws-marketplace \
  --filters "Name=name,Values=*COOL*" "Name=architecture,Values=arm64" \
  --query 'reverse(sort_by(Images,&CreationDate))[:6].[ImageId,Name]' \
  --output text | sed 's/^/    /'

AMI=$(aws ec2 describe-images --region "$REGION" --owners aws-marketplace \
  --filters "Name=name,Values=*COOL*Graviton4*" "Name=architecture,Values=arm64" \
  --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)

if [ -z "$AMI" ] || [ "$AMI" = "None" ]; then
  echo "==> no Graviton4 COOL AMI listed at all"
  exit 1
fi

SUBNET=$(aws ec2 describe-subnets --region "$REGION" \
  --filters Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text)

echo "==> can this account launch $AMI?"
OUT=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" \
  --instance-type "$TYPE" --subnet-id "$SUBNET" --dry-run 2>&1)

case "$OUT" in
  *DryRunOperation*)
    echo "    SUBSCRIBED. Put this in infra/terraform/terraform.tfvars and re-apply:"
    echo
    echo "        cool_ami_id = \"$AMI\""
    echo
    echo "    then the fourth benchmark arm:"
    echo "        COOL_AMI=$AMI bash bench/run_on_ec2.sh c8g.8xlarge c8g-cool"
    exit 0 ;;
  *OptInRequired*)
    echo "    NOT SUBSCRIBED yet. Accept the terms here, then re-run this script:"
    echo "$OUT" | grep -o 'https://aws.amazon.com/marketplace[^ ]*' | sed 's/^/        /'
    exit 2 ;;
  *)
    echo "    something else went wrong:"
    echo "$OUT" | head -3 | sed 's/^/        /'
    exit 3 ;;
esac
