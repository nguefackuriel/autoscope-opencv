# Running AutoScope on AWS, and turning it off

What is deployed, what watches it, what stops it spending, and how to tear it
down. The stack is one `terraform apply` in `infra/terraform/`. The counts here
come from `infra/terraform/deployed.json`, which `inventory.py` writes from the
state after an apply.

61 resources in `us-east-1`.

## What a reading session does

A slide lands in the uploads bucket. S3 notifies EventBridge, which starts one
Step Functions execution. The execution reads one field at a time on AWS Batch,
on a Graviton4 Spot fleet, and asks the agent what to do next after each field.
It ends in one of two Succeed states: `Done` when the agent stopped, or
`FlagForHuman` when the slide goes to a person. Neither is a failure state, and
that is deliberate: a slide the pipeline will not call on its own is the correct
outcome for a screening aid.

## What is watched

| | |
|---|---|
| API Gateway access log | `/aws/apigateway/autoscope`, 30 days |
| Step Functions execution log | `/aws/states/autoscope-session`, 30 days, level `ALL` with execution data |
| Batch job log | `/aws/batch/autoscope`, 30 days |
| DynamoDB `autoscope-trace` | one row per field read: the observation, the counts, the move and its reason, 90-day TTL |
| DynamoDB `autoscope-sessions` | one row per slide: the call and where it got to |

The state machine logs at `ALL` with `include_execution_data`, so the input and
output of every state is in CloudWatch, not only the final answer. The totals a
job works from are rebuilt from the trace rows each time, so they cannot drift
from what the audit trail says.

To watch a live reading:

    aws logs tail /aws/states/autoscope-session --follow
    aws stepfunctions list-executions --state-machine-arn <arn> --max-items 5

## What stops it spending

| guard | value | where |
|---|---|---|
| Spot fleet ceiling | 64 vCPU | `batch_max_vcpus`, `batch.tf` |
| Fleet at rest | 0 vCPU | `desired_vcpus`, so an idle stack costs nothing but storage |
| One field read | 15 min, 3 attempts | job definition timeout, `batch.tf` |
| One agent call | 2 min | `Decide` task, `agent.tf` |
| One slide | 1 hour, 400 fields | state machine `TimeoutSeconds` and the `Limits` choice |
| API | 10 requests/second, burst 20 | `api.tf` |
| Alarm | more than 50 sessions started in 5 minutes | `autoscope-too-many-sessions` |
| Budget | $100 a month, email at 80% forecast and 100% actual | `budget.tf` |
| Budget, Bedrock alone | $25 a month | a retry loop is the one failure that can spend faster than anybody notices |
| Uploads | deleted after 30 days; incomplete multipart after 1 | `storage.tf` |

The alarm and both budgets publish to one SNS topic with an email subscription.
**The subscription has to be confirmed from the email before any of them reach a
person.** An unconfirmed subscription is an alarm that fires into nothing.

Measured cost: about $2.33 per 1,000 slides on demand and $0.93 on Spot, on
`c8g.8xlarge`. `bench/cost_model.py --markdown` prints it from
`bench/pricing.json`.

## What is public, and what is not

Public: the demo page on CloudFront, and the `/sessions` API, which is open on
purpose so a judge can read a session without credentials. It is throttled, and
it can start a reading only on a slide already in the uploads bucket.

Not public: all three S3 buckets block public access and are encrypted at rest.
The demo page is served through CloudFront with an origin access control, not
from a public bucket. Nine IAM roles, one per component, each scoped by ARN to
the bucket, table, state machine or function it needs. Deployments use GitHub
OIDC, so there is no long-lived access key anywhere.

There is no patient-identifying data to protect. Every slide is public,
de-identified, retrospective research material.

## Turning it off

In this order. Anything else leaves a Spot fleet running.

    cd infra/terraform
    terraform destroy

    # if a reading is in flight, stop it first, or Batch will hold the fleet
    aws stepfunctions stop-execution --execution-arn <arn>

`terraform destroy` empties nothing: the three buckets have to be emptied by
hand first if they hold objects, which is the usual Terraform behaviour and is
left that way on purpose so a destroy cannot silently delete results.

To stop spending without tearing down, set `batch_max_vcpus = 0` and apply. The
page, the API and the trace stay readable; no new reading can start a job.

## After an apply

    python infra/terraform/inventory.py      # refresh deployed.json
    python docs/report/build_report.py       # the report reads deployed.json

The report quotes the resource count, the log retention, the alarm threshold and
the two budget amounts. They come from that file, so the report describes the
stack that is running rather than the stack that was written.
