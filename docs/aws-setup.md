# Bringing the AWS account up

16 September 2026. Forty days to the deadline. This is the ordered list of
what a person has to do by hand in a browser or a terminal before any of the
Terraform in `infra/terraform/` can run, and before the COOL benchmark has a
machine to run on.

Three of these have a waiting time that is not ours to control — a quota
increase, Bedrock model access, and the Marketplace subscription. **Do those
three first, today, even if nothing else gets done.** Everything after them is
minutes of work.

## 0. Which account, which region

The laptop already has a `[default]` profile in `~/.aws/credentials` from May,
pointing at `us-east-1`. We do not know it is the competition account. Do not
build on top of it: give the competition its own named profile so a mistake
cannot reach into the other one, and so `AWS_PROFILE=autoscope` in a shell is
the whole of the safety story.

**Region: `us-east-1`.** It is the only one where all four things we need are
in the same place: `c8g` (Graviton4), Bedrock with the Claude models, the
Marketplace AMI, and the cheapest S3. Nothing in the design is region-locked,
but mixing regions costs egress and a day of confusion.

```bash
brew install awscli          # not installed on this laptop as of 16 Sept
aws --version                # want v2.x

aws configure --profile autoscope    # paste the access key for the NEW account
export AWS_PROFILE=autoscope         # put this in ~/.zshrc too
aws sts get-caller-identity          # prints the account id: write it down
```

If `get-caller-identity` prints an account you recognise as the old one, stop:
the keys went into the wrong profile.

## 1. The budget alarm, before anything that can cost money

The proposal promises this and it is the cheapest insurance there is. A
Graviton Spot fleet left running over a weekend is the whole grant.

Console → **Billing and Cost Management → Budgets → Create budget**:

- Cost budget, monthly, **$150**.
- Alerts at 50%, 80% and 100% of *forecasted* spend, to the email on the account.
- A second budget at **$25/month for Bedrock only** (filter: Service = Bedrock),
  because a loop that retries in a tight circle is the one failure mode that
  can spend money faster than we notice.

Do the same in `infra/terraform/budget.tf` later if you like, but click it now.
Terraform is not going to run before the quota comes through, and the window
where an accident is possible starts the moment the keys work.

## 2. The three waits — start all of them today

### 2a. Service quota for Graviton (24–48 h, sometimes longer)

A new AWS account frequently starts at **0 vCPU** for On-Demand and Spot. If
that is us, there is no `c8g` to benchmark on and no Batch queue that can
start, and we find out the day we try.

Console → **Service Quotas → AWS services → Amazon EC2**, request:

| Quota | Ask for |
|---|---|
| Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances | **32 vCPU** |
| All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests | **64 vCPU** |

32 on-demand vCPU is a `c8g.8xlarge` for the benchmark, or two smaller boxes
side by side. 64 Spot is the Batch fleet with room for the fleet to be
interrupted and come back. Ask for both in the same session; they are approved
independently and the Spot one is usually slower.

Check with:

```bash
aws service-quotas list-requested-service-quota-change-history \
  --service-code ec2 --region us-east-1 \
  --query 'RequestedQuotas[].{Q:QuotaName,S:Status,V:DesiredValue}' --output table
```

### 2b. Bedrock model access (minutes to a day)

Console → **Amazon Bedrock → Model access** (in `us-east-1`) → enable the
Anthropic Claude models. Some accounts get it instantly, some sit in review,
and a brand-new account with no billing history is the kind that sits.

The agent is written against **`anthropic.claude-opus-5`**. Confirm the id the
account actually offers, because that string is what goes in the Terraform:

```bash
aws bedrock list-foundation-models --region us-east-1 \
  --by-provider anthropic \
  --query 'modelSummaries[].modelId' --output table
```

Then prove a call works end to end, which is also the first thing that will be
asked on the check-in call:

```bash
cd autoscope && .venv/bin/pip install 'anthropic[bedrock]'
AWS_PROFILE=autoscope .venv/bin/python -c "
from anthropic import AnthropicBedrockMantle
c = AnthropicBedrockMantle(aws_region='us-east-1')
m = c.messages.create(model='anthropic.claude-opus-5', max_tokens=64,
                      messages=[{'role':'user','content':'reply with the single word: ready'}])
print(m.content[0].text, '|', m.usage)
"
```

### 2c. The COOL AMI subscription (hours, and it is a human approval)

AWS Marketplace → search **"Cloud Optimized OpenCV"** (OpenCV's own listing for
AWS Graviton4; COOL is the KleidiCV-powered build). Subscribe, accept the
terms, and note the **AMI id for `us-east-1`** — it goes in
`infra/terraform/variables.tf` as `cool_ami_id`.

Two warnings, both of which change what the benchmark means:

1. A Marketplace subscription on a new account can sit behind a manual review.
   It is the single item that can make the COOL award impossible to claim, and
   it is worth 30% of that award's score. Start it today.
2. **The stock pip wheel on aarch64 already ships KleidiCV** (`Custom HAL:
   KleidiCV (ver 26.03)`, printed by our own arm64 CI). So "COOL versus the pip
   wheel" compares two KleidiCV builds and measures close to nothing. The
   honest third arm is a source build with `-DWITH_KLEIDICV=OFF`, which is what
   `bench/Dockerfile.kleidicv-off` builds. All three arms go in the report.
   See `docs/plan-2026-09-02.md` §7.

## 3. An IAM role for Terraform and one for CI

Do not run Terraform as the root user, and do not put a long-lived key in
GitHub Actions.

```bash
# A user for Terraform from the laptop. Scope it down after the first apply;
# for now PowerUserAccess plus IAM is what a first apply needs.
aws iam create-user --user-name autoscope-terraform
aws iam attach-user-policy --user-name autoscope-terraform \
  --policy-arn arn:aws:iam::aws:policy/PowerUserAccess
aws iam create-access-key --user-name autoscope-terraform
```

For GitHub Actions use OIDC, so there is no key to leak. `infra/terraform/oidc.tf`
creates the provider and the role; the trust policy is scoped to
`repo:nguefackuriel/autoscope-opencv:*`. After `terraform apply`, put the role
ARN in the repo as the `AWS_ROLE_ARN` secret — that is the only secret CI needs.

## 4. Then Terraform

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in cool_ami_id, alert_email
terraform init
terraform plan        # read it. It should create ~30 resources and no IAM user.
terraform apply
```

`terraform destroy` between test runs is the other half of the money story.
The S3 buckets have `prevent_destroy` on them so the data survives; everything
with an hourly price does not.

## 5. What the check-in call will ask

The 30-minute Zoom between **21 September and 2 October** releases the second
half of the grant. From the rubric, the answerable questions are: is there a
working endpoint, does COOL actually run on Graviton, and does the agent do
something a threshold could not. Sections 1 to 4 above are what turns all three
from "planned" into a URL and a number.

Book it now. It is the one deadline on this project that cannot be recovered by
working a weekend.

## Running cost, roughly

| Thing | When | Cost |
|---|---|---|
| `c8g.8xlarge` on demand, 3 benchmark runs × 2 h | benchmark week | ~$12 |
| `c7i.8xlarge` on demand, the x86 arm, 2 h | benchmark week | ~$8 |
| Batch on Spot `c8g`, a few hundred slides | through October | ~$15 |
| Bedrock, Claude Opus 5, ~50 decisions/slide | through October | ~$20 |
| S3, DynamoDB, CloudFront, API Gateway | continuous | < $5/month |

Call it **$80 of the ~$250–300 we have**, which leaves room for the demo to
stay up through judging. The COOL AMI itself carries no software charge beyond
the instance.
