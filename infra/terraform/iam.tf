# Every role in the stack. Read this file to know what can reach what.
#
# The one rule worth stating: the agent's role can call Bedrock and read the
# numbers, and it cannot write a result. A result is written by the pipeline
# job from what OpenCV measured. The model picks where to look and when to
# stop; it never picks the diagnosis, and this is where that is enforced
# rather than promised.

data "aws_iam_policy_document" "assume" {
  for_each = toset([
    "batch.amazonaws.com",
    "ec2.amazonaws.com",
    "ecs-tasks.amazonaws.com",
    "lambda.amazonaws.com",
    "states.amazonaws.com",
    "spotfleet.amazonaws.com",
    "events.amazonaws.com",
  ])

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = [each.key]
    }
  }
}

# --- Batch -----------------------------------------------------------------

resource "aws_iam_role" "batch_service" {
  name               = "${var.name}-batch-service"
  assume_role_policy = data.aws_iam_policy_document.assume["batch.amazonaws.com"].json
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role" "spot_fleet" {
  name               = "${var.name}-spot-fleet"
  assume_role_policy = data.aws_iam_policy_document.assume["spotfleet.amazonaws.com"].json
}

resource "aws_iam_role_policy_attachment" "spot_fleet" {
  role       = aws_iam_role.spot_fleet.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

resource "aws_iam_role" "batch_instance" {
  name               = "${var.name}-batch-instance"
  assume_role_policy = data.aws_iam_policy_document.assume["ec2.amazonaws.com"].json
}

resource "aws_iam_role_policy_attachment" "batch_instance" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_instance" {
  name = "${var.name}-batch-instance"
  role = aws_iam_role.batch_instance.name
}

# What the pipeline container itself can do: read a slide, write results and
# the per-field numbers. It cannot call Bedrock.
resource "aws_iam_role" "batch_job" {
  name               = "${var.name}-batch-job"
  assume_role_policy = data.aws_iam_policy_document.assume["ecs-tasks.amazonaws.com"].json
}

data "aws_iam_policy_document" "batch_job" {
  statement {
    sid     = "ReadSlides"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.uploads.arn,
      "${aws_s3_bucket.uploads.arn}/*",
    ]
  }

  statement {
    sid       = "WriteResults"
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.results.arn}/*"]
  }

  statement {
    sid    = "WriteTheNumbers"
    effect = "Allow"
    actions = [
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
      "dynamodb:GetItem",
      "dynamodb:Query",
    ]
    resources = [
      aws_dynamodb_table.trace.arn,
      aws_dynamodb_table.sessions.arn,
    ]
  }
}

resource "aws_iam_role_policy" "batch_job" {
  name   = "${var.name}-batch-job"
  role   = aws_iam_role.batch_job.id
  policy = data.aws_iam_policy_document.batch_job.json
}

# --- The agent Lambda -------------------------------------------------------

resource "aws_iam_role" "agent" {
  name               = "${var.name}-agent"
  assume_role_policy = data.aws_iam_policy_document.assume["lambda.amazonaws.com"].json
}

data "aws_iam_policy_document" "agent" {
  statement {
    sid    = "AskTheModel"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    # The foundation-model ARN is region-WILDCARD on purpose. A `us.` inference
    # profile is cross-region: it routes the call to whichever of us-east-1,
    # us-east-2 or us-west-2 has capacity, and the authorisation is checked
    # against the region it chose. Pinning this to var.region produced a 403
    # naming us-east-2 for a call made from us-east-1, which reads like a
    # configuration error somewhere else entirely.
    #
    # The inference-profile ARN stays pinned: that resource really does live in
    # this region and is account-scoped.
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:${var.region}:${local.account_id}:inference-profile/*",
    ]
  }

  # The Mantle endpoint is a DIFFERENT SERVICE NAMESPACE from classic Bedrock.
  # AnthropicBedrockMantle, which is the client the SDK recommends and the one
  # agent/bedrock.py uses, calls bedrock-mantle:CreateInference against a
  # project ARN -- not bedrock:InvokeModel against a model ARN. Granting only
  # the latter gets a 403 that names an action nothing in the stack mentions,
  # which is how the second end-to-end slide failed.
  statement {
    sid    = "AskTheModelThroughMantle"
    effect = "Allow"
    actions = [
      "bedrock-mantle:CreateInference",
      "bedrock-mantle:GetInference",
    ]
    resources = ["arn:aws:bedrock-mantle:${var.region}:${local.account_id}:project/*"]
  }

  statement {
    sid    = "ReadTheNumbers"
    effect = "Allow"
    # Read, and write the decision. Not the count: the count comes from the
    # pipeline job. The model does not get to write what it concluded into the
    # field the report reads.
    actions = [
      "dynamodb:GetItem",
      "dynamodb:Query",
      "dynamodb:PutItem",
    ]
    resources = [
      aws_dynamodb_table.trace.arn,
      aws_dynamodb_table.sessions.arn,
    ]
  }

  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${local.account_id}:*"]
  }
}

resource "aws_iam_role_policy" "agent" {
  name   = "${var.name}-agent"
  role   = aws_iam_role.agent.id
  policy = data.aws_iam_policy_document.agent.json
}

# --- The API Lambda ---------------------------------------------------------

resource "aws_iam_role" "api" {
  name               = "${var.name}-api"
  assume_role_policy = data.aws_iam_policy_document.assume["lambda.amazonaws.com"].json
}

data "aws_iam_policy_document" "api" {
  statement {
    sid       = "PresignUploads"
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.uploads.arn}/*"]
  }

  statement {
    sid       = "ReadResults"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.results.arn, "${aws_s3_bucket.results.arn}/*"]
  }

  statement {
    sid       = "ReadSessions"
    effect    = "Allow"
    actions   = ["dynamodb:GetItem", "dynamodb:Query", "dynamodb:PutItem", "dynamodb:UpdateItem"]
    resources = [aws_dynamodb_table.sessions.arn, aws_dynamodb_table.trace.arn]
  }

  statement {
    sid       = "StartASession"
    effect    = "Allow"
    actions   = ["states:StartExecution", "states:DescribeExecution"]
    resources = [aws_sfn_state_machine.session.arn]
  }

  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${local.account_id}:*"]
  }
}

resource "aws_iam_role_policy" "api" {
  name   = "${var.name}-api"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}

# --- Step Functions ---------------------------------------------------------

resource "aws_iam_role" "states" {
  name               = "${var.name}-states"
  assume_role_policy = data.aws_iam_policy_document.assume["states.amazonaws.com"].json
}

data "aws_iam_policy_document" "states" {
  statement {
    sid       = "RunTheAgent"
    effect    = "Allow"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.agent.arn, "${aws_lambda_function.agent.arn}:*"]
  }

  statement {
    sid    = "RunTheImageWork"
    effect = "Allow"
    actions = [
      "batch:SubmitJob",
      "batch:DescribeJobs",
      "batch:TerminateJob",
    ]
    resources = ["*"]
  }

  # Vended logs. Step Functions does not write to the log group directly: it
  # sets up a log DELIVERY, and that needs these actions on "*" rather than on
  # the group's ARN. Without them CreateStateMachine fails with "The state
  # machine IAM Role is not authorized to access the Log Destination", which
  # names the role and not the missing actions.
  statement {
    sid    = "VendedLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogDelivery",
      "logs:GetLogDelivery",
      "logs:UpdateLogDelivery",
      "logs:DeleteLogDelivery",
      "logs:ListLogDeliveries",
      "logs:PutResourcePolicy",
      "logs:DescribeResourcePolicies",
      "logs:DescribeLogGroups",
    ]
    resources = ["*"]
  }

  # Batch's .sync integration needs these to get told when a job ends.
  statement {
    sid    = "WaitForBatch"
    effect = "Allow"
    actions = [
      "events:PutTargets",
      "events:PutRule",
      "events:DescribeRule",
    ]
    resources = [
      "arn:aws:events:${var.region}:${local.account_id}:rule/StepFunctionsGetEventsForBatchJobsRule",
    ]
  }
}

resource "aws_iam_role_policy" "states" {
  name   = "${var.name}-states"
  role   = aws_iam_role.states.id
  policy = data.aws_iam_policy_document.states.json
}

# --- EventBridge ------------------------------------------------------------

resource "aws_iam_role" "events" {
  name               = "${var.name}-events"
  assume_role_policy = data.aws_iam_policy_document.assume["events.amazonaws.com"].json
}

data "aws_iam_policy_document" "events" {
  statement {
    effect    = "Allow"
    actions   = ["states:StartExecution"]
    resources = [aws_sfn_state_machine.session.arn]
  }
}

resource "aws_iam_role_policy" "events" {
  name   = "${var.name}-events"
  role   = aws_iam_role.events.id
  policy = data.aws_iam_policy_document.events.json
}
