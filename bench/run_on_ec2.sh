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
# PROFILE=1 runs bench/field_profile.py on real NLM fields instead of the
# synthetic microbenchmark. The fields come from S3 as a fixed bundle, so
# every arm profiles the same twelve images and the comparison is not a
# comparison of samples.
PROFILE="${PROFILE:-0}"
# GROUP=pipeline swaps the 1024px synthetic cases for the candidate stage's real
# calls at 12 MP. See _pipeline_cases in cool_bench.py for why that exists.
GROUP="${GROUP:-tile}"
THREADS="${THREADS:-0}"
# TILED_MORPH=1 turns on the band-parallel morphology in parasites.py for a
# PROFILE=1 run. Off by default so the committed profiles stay what they were
# measured with. Use a label that says so, e.g. c8g-cool-tiled, and point
# BUNDLE at a bundle whose src/ carries the flag.
TILED_MORPH="${TILED_MORPH:-}"
BUNDLE="${BUNDLE:-s3://autoscope-bench-617847240757/bench-bundle.tar.gz}"
REGION="${AWS_REGION:-us-east-1}"
PROFILE_NAME="autoscope-bench-ssm"
# COOL_AMI swaps the stock Amazon Linux image for the Marketplace Cloud
# Optimized OpenCV AMI. That image already carries OpenCV built the way COOL
# ships it, so this arm must NOT pip install opencv over the top: doing so
# would time the wheel again and label the result COOL.
COOL_AMI="${COOL_AMI:-}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/.." && pwd)"

case "$TYPE" in
  *g.*|*g[0-9]*) ARCH=arm64 ;;
  *)             ARCH=x86_64 ;;
esac
AMI_PARAM="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-${ARCH}"

echo "==> $LABEL on $TYPE ($ARCH) in $REGION"
# The COOL AMI is not Amazon Linux -- `dnf` is not on it -- and it already
# carries OpenCV with every library OpenCV needs. Installing the GUI packages
# is both impossible and pointless there.
if [ -n "$COOL_AMI" ]; then
  BASE_PACKAGES="true   # the COOL image brings its own OpenCV and its deps"
else
  BASE_PACKAGES="dnf install -y python3.12 python3.12-pip mesa-libGL libxcb glib2 >/dev/null"
fi

if [ -n "$COOL_AMI" ]; then
  AMI="$COOL_AMI"
  echo "    using the COOL Marketplace AMI"
else
  AMI=$(aws ssm get-parameters --names "$AMI_PARAM" --region "$REGION" \
          --query 'Parameters[0].Value' --output text)
fi
# Every default subnet, one per availability zone. A c8g.8xlarge is not always
# available in the first zone on the list: on 16 October us-east-1c answered
# InsufficientInstanceCapacity while 1a, 1b, 1d and 1f had capacity, and the
# run died on the one zone it had asked for. Try them in turn.
SUBNETS=$(aws ec2 describe-subnets --region "$REGION" \
        --filters Name=default-for-az,Values=true \
        --query 'Subnets[].[SubnetId,AvailabilityZone]' --output text)
ID=""
while read -r SUBNET AZ; do
  [ -n "$SUBNET" ] || continue
  echo "    ami $AMI, trying $AZ ($SUBNET)"
  OUT=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$TYPE" --subnet-id "$SUBNET" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --instance-initiated-shutdown-behavior terminate \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=autoscope-cool-$LABEL},{Key=Project,Value=autoscope}]" \
    --query 'Instances[0].InstanceId' --output text 2>&1) && { ID="$OUT"; break; }
  case "$OUT" in
    *InsufficientInstanceCapacity*) echo "    no $TYPE capacity in $AZ, next zone" ;;
    *) echo "$OUT"; exit 1 ;;
  esac
done <<EOF
$SUBNETS
EOF
[ -n "$ID" ] || { echo "no zone in $REGION has $TYPE capacity right now"; exit 1; }
echo "    instance $ID in $AZ"

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

# The Amazon Linux arms reach the bundle with `aws s3 cp`, because the AL2023
# image carries the CLI. The COOL image is Ubuntu and does not, and installing
# a CLI onto a benchmark instance is both a few minutes of apt and a change to
# the thing being measured. A presigned GET is the smaller move: the URL is
# generated here with our own credentials, expires in an hour, and the instance
# needs nothing but curl.
if [ -n "$COOL_AMI" ]; then
  FETCH_BUNDLE="curl -sSfL '$(aws s3 presign "$BUNDLE" --region "$REGION" --expires-in 3600)' -o /opt/bench/bundle.tar.gz"
else
  FETCH_BUNDLE="aws s3 cp '$BUNDLE' /opt/bench/bundle.tar.gz"
fi
if [ "$PROFILE" = "1" ]; then
  RUN_COOL="true   # the microbenchmark is skipped: this run profiles the pipeline"
else
  # `. ./.py || true` is not a guard: in dash under `set -e`, sourcing a file
  # that does not exist exits the shell before the `||` is ever reached, which
  # is how the stock arm died on a file only the COOL arm writes.
  RUN_COOL="cd /opt/bench; if [ -f ./.py ]; then . ./.py; fi; \${COOL_PY:-python3.12} cool_bench.py --runs $RUNS --group $GROUP --threads '$THREADS' --label '$LABEL'"
fi

if [ "$PROFILE" = "1" ]; then
  RESULT="profile-$LABEL"
  RUN_PROFILE=$(cat <<PROF
PY=\${COOL_PY:-python3.12}
\$PY -m pip install --quiet 'scipy==1.16.3'
$FETCH_BUNDLE
tar -xzf /opt/bench/bundle.tar.gz -C /opt/bench
cd /opt/bench && AUTOSCOPE_TILED_MORPH='$TILED_MORPH' PYTHONPATH="/opt/bench/src:\${PYTHONPATH:-}" \$PY bench/field_profile.py \
    --fields 12 --label '$LABEL'
cp /opt/bench/bench/results/profile-$LABEL.json /opt/bench/results/
PROF
)
else
  if [ "$GROUP" = "tile" ]; then RESULT="$LABEL"; else RESULT="$LABEL-pipeline"; fi
  RUN_PROFILE=""
fi

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
  `# dnn is in the list because the pipeline profile loads the patch network` \
  `# through cv.dnn. Leaving it out built fine and then failed at runtime with` \
  `# "module 'cv2' has no attribute 'dnn'", half an hour in.` \
  -DBUILD_LIST=core,imgproc,imgcodecs,dnn,python3 \
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
elif [ -n "$COOL_AMI" ]; then
  # The COOL image is Ubuntu 24.04, not Amazon Linux, and it keeps its OpenCV
  # off every default path: the bindings sit in /opt/cool/python_<ver>/
  # site-packages and the shared objects in /opt/cool/cpp_sdk/lib, so a bare
  # `python3 -c "import cv2"` fails on this image even though OpenCV is right
  # there -- which is how the first attempt at this arm concluded the image had
  # no OpenCV at all. The matching venv under /opt/cool/venvs carries numpy and
  # nothing else, so nothing has to be installed over COOL's own build.
  INSTALL_OPENCV=$(cat <<'COOLSETUP'
export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:${LD_LIBRARY_PATH:-}
COOL_PY=""; COOL_PP=""
for v in 3.12 3.11 3.10; do
  cand=/opt/cool/venvs/python_$v/bin/python
  [ -x "$cand" ] || continue
  if PYTHONPATH=/opt/cool/python_$v/site-packages $cand -c "import cv2" 2>/dev/null; then
    COOL_PY=$cand; COOL_PP=/opt/cool/python_$v/site-packages; break
  fi
done
# A COOL arm that quietly fell back to the pip wheel is worse than no COOL arm
# at all, so stop here rather than install one.
if [ -z "$COOL_PY" ]; then
  echo "NO PYTHON ON THIS IMAGE CAN IMPORT THE COOL OPENCV"
  echo "searched /opt/cool/venvs/python_{3.12,3.11,3.10} with LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib"
  ls -la /opt/cool 2>&1 | head -20
  exit 1
fi
export PYTHONPATH="$COOL_PP"
$COOL_PY -c "
import cv2, sys
info = cv2.getBuildInformation()
want = ('Custom HAL', 'Parallel framework', 'Baseline', 'Version control')
for line in info.splitlines():
    if line.strip().startswith(want):
        print('   ', line.strip())
print('    opencv', cv2.__version__, 'at', cv2.__file__)
if 'kleidicv' not in info.lower():
    sys.exit('THE COOL IMAGE HAS NO KLEIDICV IN ITS BUILD -- not a COOL arm')
"
mkdir -p /opt/bench
printf 'export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib\nexport PYTHONPATH=%s\nCOOL_PY=%s\n' \
  "$COOL_PP" "$COOL_PY" > /opt/bench/.py
COOLSETUP
)
else
  INSTALL_OPENCV="python3.12 -m pip install --quiet 'opencv-python==5.0.0.93' 'numpy==2.3.4'"
fi

CMD=$(cat <<EOF
set -eux
# A dead-man switch. The trap above terminates the instance when this script
# exits, but only if this script gets to run its trap: killing the orchestrating
# shell -- a Ctrl-C that is not delivered, a backgrounded run that is stopped --
# leaves a c8g.8xlarge billing at \$1.28 an hour with nobody watching it. That
# happened once, for fourteen minutes. The instance now shuts itself down after
# 90 minutes whatever else goes wrong, and the launch sets
# instance-initiated-shutdown-behavior=terminate, so a shutdown is a terminate.
shutdown -h +90 &
# opencv-python links libGL, glib and xcb even when no window is ever opened,
# and the minimal AL2023 AMI has none of them. Same two packages the Dockerfile
# installs on Debian, under their Amazon Linux names. The wheel stays
# opencv-python and not -headless on purpose: the point is to time the same
# binary the project ships.
$BASE_PACKAGES
$INSTALL_OPENCV
mkdir -p /opt/bench/results
echo '$SCRIPT_B64' | base64 -d > /opt/bench/cool_bench.py
$RUN_COOL
$RUN_PROFILE
echo '--- RESULT JSON BELOW ---'
cat /opt/bench/results/$RESULT.json
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
  > "$here/results/$RESULT.json"

echo "==> $RESULT"
python3 - "$here/results/$RESULT.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"    opencv {r['opencv_version']} on {r['machine']}, "
      f"kleidicv in build: {r['kleidicv_in_build']}")
if r.get("by_threads"):                  # cool_bench, one block per thread count
    for n, block in r["by_threads"].items():
        print(f"    --- {n if n != '0' else 'default'} thread(s) ---")
        for name, v in block.items():
            print(f"    {name:<22}{v['median_ms']:>9.3f} ms   IQR {v['iqr_ms']:.3f}")
elif "results" in r:                     # cool_bench: per-function timings
    for name, v in r["results"].items():
        print(f"    {name:<22}{v['median_ms']:>9.3f} ms   IQR {v['iqr_ms']:.3f}")
else:                                    # field_profile: the whole pipeline
    for name, v in r["stages"].items():
        print(f"    {name:<26}{v:>8.3f} s")
    print(f"    {'per field':<26}{r['seconds_per_field_median']:>8.3f} s")
    print(f"    {'slides/hour':<26}{r['slides_per_hour']:>8.1f}")
PY
echo "    written to bench/results/$RESULT.json"
