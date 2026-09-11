# ClearML on AWS, for the detector runs

ClearML tracks experiments and can start GPU instances on demand. It does not
provide compute: everything below is your AWS account spending your money, so
the budget alarm is step 1 and not step 4.

## 1. Budget alarm, before anything else runs

Week-1 item 1, still outstanding. Do this before the autoscaler exists, because
an autoscaler with a misconfigured idle timeout is exactly the thing that bills
you for a week of idle A10G.

```bash
aws budgets create-budget --account-id <ACCOUNT_ID> \
  --budget file://budget.json \
  --notifications-with-subscribers file://budget-notify.json
```

Set the threshold somewhere you would actually notice. A `g5.xlarge` on Spot is
roughly $0.30-0.40/hour depending on region and demand; on demand it is about
$1.00. Confirm the current numbers rather than trusting this line.

## 2. ClearML account and credentials

Free tier at `app.clear.ml`. Metrics, scalars and artifacts are stored on their
servers. That is acceptable for this project because the NLM set is
de-identified and public and the Tanzania set is CC0 — but it is worth being
deliberate about, because it would not be acceptable for clinic data.

```bash
pip install clearml==2.1.12
clearml-init          # paste the credentials from Settings > Workspace
```

This writes `~/clearml.conf`. It holds API secrets, so it never goes in the
repo.

## 3. The autoscaler

Configure it in the ClearML web UI under **Applications > AWS Autoscaler**, or
from `autoscaler.yaml` here. Settings that matter:

| setting | value | why |
|---|---|---|
| queue | `gpu-spot` | what `clearml-task` enqueues to |
| instance type | `g5.xlarge` or `g6.xlarge` | 24 GB VRAM, enough for Faster R-CNN at batch 4 on 640 px tiles |
| spot | yes | 60-70% cheaper; checkpoints are written every epoch so an interruption costs one epoch |
| max instances | `1` | two people, one experiment at a time. Raise it deliberately, never by default |
| idle timeout | `10` minutes | the single setting most likely to cost real money if wrong |
| AMI | a current Deep Learning AMI (Ubuntu, GPU) | ships the NVIDIA driver. Look the ID up for your region; they are region-specific and rotate |
| root volume | 100 GB | the NLM set is 3.9 GB, the tiles are larger, and torch is ~3 GB |

## 4. Getting the data onto the instance

The dataset is not in the repo and must not be. Two options:

- **Download on the instance.** `make data` fetches the NLM set directly. This
  is simplest and costs nothing but time.
- **`clearml-data`.** Version the *derived* tiles once and pull them on each
  run. Worth it after the export settles, because it pins the exact tiles a
  result came from — which is the same discipline `wbc_sweep.json` already
  applies to the split.

```bash
clearml-data create --project AutoScope --name parasite-tiles
clearml-data add --files data/derived/detect
clearml-data close
```

## 5. Running

```bash
clearml-task create --project AutoScope/parasites --name yolo26 \
  --repo <this repo> --branch b/parasite-detectors \
  --script train/yolo26.py --queue gpu-spot \
  --requirements requirements-detect.txt
```

Faster R-CNN is the same with `--script train/frcnn.py`.

## What is deliberately not here

No Terraform. The autoscaler is a ClearML-managed application, so putting the
instances in Terraform would give two things authority over the same
lifecycle and they would fight. The AWS pieces Terraform should own — the
budget, the IAM role, the S3 bucket — belong in `infra/` proper once that
exists, and are listed above as manual steps only until then.
