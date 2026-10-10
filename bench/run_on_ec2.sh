#!/usr/bin/env bash
# Run bench/cool_bench.py on a real EC2 instance and bring the JSON home.
#
#   bash bench/run_on_ec2.sh c8g.8xlarge c8g-stock
#   bash bench/run_on_ec2.sh c7i.8xlarge c7i-stock
#
# One instance, one arm, one result file in bench/results/, instance terminated
# at the end whatever happens. No SSH key: the instance carries an SSM profile
# and commands go through Systems Manager, so there is no key to leave lying
# around and nothing to clean up but the instance itself.
#
# WHY A SCRIPT AND NOT A SESSION OF TYPING
#
# A benchmark nobody can re-run is an assertion. The COOL claim is worth what
# its reproduction is worth, so the machine, the image, the package versions
# and the command all live here in one file that a judge can read and run.
#
# WHAT THIS DOES NOT MEASURE
#
# cool_bench.py times isolated OpenCV calls on a synthetic tile. The
# whole-pipeline number -- slides per hour -- needs the NLM images, which are
# 4 GB and not on the instance. That is bench/field_profile.py and a separate
# step; see docs/aws-setup.md.

set -euo pipefail

TYPE="${1:?usage: run_on_ec2.sh <instance-type> <label> [runs]}"
LABEL="${2:?usage: run_on_ec2.sh <instance-type> <label> [runs]}"
RUNS="${3:-100}"
# KLEIDICV=off builds OpenCV from source with the Arm acceleration compiled
# out, instead of installing the wheel. That build is the control arm: the pip
# wheel on aarch64 already ships KleidiCV, so "COOL against the wheel" compares
# two accelerated builds and measures almost nothing.
KLEIDICV="${KLEIDICV:-on}"
REGION="${AWS_REGION:-us-east-1}"
PROFILE_NAME="autoscope-bench-ssm"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/.." && pwd)"

case "$TYPE" in
  *g.*|*g[0-9]*) ARCH=arm64 ;;
  *)             ARCH=x86_64 ;;
esac
AMI_PARAM="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-${ARCH}"

echo "==> $LABEL on $TYPE ($ARCH) in $REGION"
AMI=$(aws ssm get-parameters --names "$AMI_PARAM" --region "$REGION" \
        --query 'Parameters[0].Value' --output text)
SUBNET=$(aws ec2 describe-subnets --region "$REGION" \
        --filters Name=default-for-az,Values=true \
        --query 'Subnets[0].SubnetId' --output text)
echo "    ami $AMI, subnet $SUBNET"

ID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$TYPE" --subnet-id "$SUBNET" \
  --iam-instance-profile "Name=$PROFILE_NAME" \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=autoscope-cool-$LABEL},{Key=Project,Value=autoscope}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "    instance $ID"

# Terminate on any exit, including a failure or a Ctrl-C. An instance left
# running after a benchmark is the most expensive kind of mistake here.
cleanup() {
  echo "==> terminating $ID"
  aws ec2 terminate-instances --instance-ids "$ID" --region "$REGION" \
    --query 'TerminatingInstances[0].CurrentState.Name' --output text || true
}
trap cleanup EXIT

echo "==> waiting for SSM to pick it up"
for _ in $(seq 1 60); do
  online=$(aws ssm describe-instance-information --region "$REGION" \
    --filters "Key=InstanceIds,Values=$ID" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || echo None)
  [ "$online" = "Online" ] && break
  sleep 10
done
[ "${online:-}" = "Online" ] || { echo "SSM never saw the instance"; exit 1; }
echo "    online"

# The benchmark script goes over in the command itself rather than through S3
# or a git clone: it is one file, the repository is private, and a token on a
# throwaway instance is a token too many.
SCRIPT_B64=$(base64 < "$here/cool_bench.py" | tr -d '\n')

if [ "$KLEIDICV" = "off" ]; then
  echo "    building OpenCV 5 from source with KleidiCV OFF (20-40 min)"
  INSTALL_OPENCV=$(cat <<'BUILD'
dnf install -y python3.12-devel gcc gcc-c++ cmake ninja-build git \
    libjpeg-turbo-devel libpng-devel >/dev/null
python3.12 -m pip install --quiet 'numpy==2.3.4'
git clone --depth 1 --branch 5.0.0 https://github.com/opencv/opencv.git /opt/opencv-src
cmake -G Ninja -S /opt/opencv-src -B /opt/opencv-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_KLEIDICV=OFF -DWITH_CAROTENE=OFF \
  -DBUILD_LIST=core,imgproc,imgcodecs,python3 \
  -DBUILD_opencv_python3=ON \
  -DPYTHON3_EXECUTABLE=/usr/bin/python3.12 \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCS=OFF -DWITH_FFMPEG=OFF -DWITH_GTK=OFF >/dev/null
cmake --build /opt/opencv-build --parallel >/dev/null
cmake --install /opt/opencv-build >/dev/null
CV2=$(find /usr/local /usr/lib64 -name 'cv2*.so' 2>/dev/null | head -1)
export PYTHONPATH=$(dirname "$CV2")
# A control that still has the acceleration linked would report a 1.0x speedup
# as a finding. Fail here instead of producing a plausible wrong number.
python3.12 -c "
import cv2, sys
info = cv2.getBuildInformation().lower()
bad = [w for w in ('kleidicv','carotene') if w in info]
sys.exit('CONTROL BUILD STILL HAS: %s' % bad) if bad else print('control build clean, opencv', cv2.__version__)
"
BUILD
)
else
  INSTALL_OPENCV="python3.12 -m pip install --quiet 'opencv-python==5.0.0.93' 'numpy==2.3.4'"
fi

CMD=$(cat <<EOF
set -eux
# opencv-python links libGL, glib and xcb even when no window is ever opened,
# and the minimal AL2023 AMI has none of them. Same two packages the Dockerfile
# installs on Debian, under their Amazon Linux names. The wheel stays
# opencv-python and not -headless on purpose: the point is to time the same
# binary the project ships.
dnf install -y python3.12 python3.12-pip mesa-libGL libxcb glib2 >/dev/null
$INSTALL_OPENCV
mkdir -p /opt/bench/results
echo '$SCRIPT_B64' | base64 -d > /opt/bench/cool_bench.py
cd /opt/bench && PYTHONPATH="${PYTHONPATH:-}" python3.12 cool_bench.py --runs $RUNS --label '$LABEL'
echo '--- RESULT JSON BELOW ---'
cat /opt/bench/results/$LABEL.json
EOF
)

echo "==> running the benchmark ($RUNS runs per function)"

# The parameters go over as a JSON file, one array entry per line. The
# shorthand `--parameters "commands=[...]"` collapses newlines and silently
# produces a mangled script: `set -eux` arrived as `set -d` and two lines were
# concatenated into `cool_bench.pyncd`. A file is the only shape that survives.
PARAMS="$(mktemp -t ssmparams).json"
python3 - "$CMD" > "$PARAMS" <<'PYEOF'
import json, sys
print(json.dumps({"commands": sys.argv[1].split("\n")}))
PYEOF

CMD_ID=$(aws ssm send-command --region "$REGION" \
  --instance-ids "$ID" --document-name AWS-RunShellScript \
  --comment "autoscope COOL $LABEL" \
  --parameters "file://$PARAMS" \
  --timeout-seconds 3600 \
  --query 'Command.CommandId' --output text)
rm -f "$PARAMS"

# NOT `aws ssm wait command-executed`: its waiter gives up after about 100
# seconds, and a from-source OpenCV build takes half an hour. The first
# KleidiCV-off run was killed mid-build by that waiter, which then read
# "InProgress" as a failure and terminated the instance. Poll instead, with a
# cap that matches the longest thing this script can legitimately be asked to
# do.
echo -n "    "
for _ in $(seq 1 240); do          # 240 x 15 s = 60 min
  STATUS=$(aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$ID" \
    --region "$REGION" --query 'Status' --output text 2>/dev/null || echo Pending)
  case "$STATUS" in
    Success|Failed|Cancelled|TimedOut) break ;;
  esac
  echo -n "."
  sleep 15
done
echo " $STATUS"
OUT=$(aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text)

if [ "$STATUS" != "Success" ]; then
  echo "command $STATUS"
  aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text | tail -30
  exit 1
fi

mkdir -p "$here/results"
printf '%s\n' "$OUT" | sed -n '/--- RESULT JSON BELOW ---/,$p' | tail -n +2 \
  > "$here/results/$LABEL.json"

echo "==> $LABEL"
python3 - "$here/results/$LABEL.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"    opencv {r['opencv_version']} on {r['machine']}, "
      f"kleidicv in build: {r['kleidicv_in_build']}")
for name, v in r["results"].items():
    print(f"    {name:<22}{v['median_ms']:>9.3f} ms   IQR {v['iqr_ms']:.3f}")
PY
echo "    written to bench/results/$LABEL.json"
