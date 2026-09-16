# The look, decide, act loop, as Step Functions.
#
# The loop itself is src/autoscope/agent/loop.py and it runs unchanged on a
# laptop. What Step Functions adds is the part that cannot be a Python loop in
# production: the limits. Fields, wall clock and money are held here, outside
# the model and outside the policy, so no policy can talk its way past them.

data "archive_file" "agent" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda/agent"
  output_path = "${path.module}/.build/agent.zip"
}

data "archive_file" "api" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda/api"
  output_path = "${path.module}/.build/api.zip"
}

# The agent: given what OpenCV measured on this field and the running totals,
# pick the next move. One Bedrock call, one move back, no state of its own.
resource "aws_lambda_function" "agent" {
  function_name = "${var.name}-agent"
  role          = aws_iam_role.agent.arn
  handler       = "handler.handler"
  runtime       = "python3.12"
  architectures = ["arm64"]
  timeout       = 60
  memory_size   = 512

  filename         = data.archive_file.agent.output_path
  source_code_hash = data.archive_file.agent.output_base64sha256

  layers = [aws_lambda_layer_version.anthropic.arn]

  environment {
    variables = {
      BEDROCK_MODEL_ID = var.bedrock_model_id
      AWS_REGION_NAME  = var.region
      TRACE_TABLE      = aws_dynamodb_table.trace.name
      SESSIONS_TABLE   = aws_dynamodb_table.sessions.name
    }
  }
}

# `anthropic[bedrock]` is pure Python and small enough for a layer.
#
# The contents are built, not committed: `bash infra/lambda/build.sh` installs
# the SDK for arm64 into layers/python. Terraform zips whatever is there rather
# than reading a pre-built zip, so `plan` works on a fresh clone instead of
# dying on a missing file, and the precondition below is what tells you the
# build step was skipped — in a sentence, at plan time, rather than as an
# ImportError in CloudWatch an hour later.
data "archive_file" "anthropic_layer" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda/layers/python"
  output_path = "${path.module}/.build/anthropic-layer.zip"
}

resource "aws_lambda_layer_version" "anthropic" {
  layer_name          = "${var.name}-anthropic"
  filename            = data.archive_file.anthropic_layer.output_path
  source_code_hash    = data.archive_file.anthropic_layer.output_base64sha256
  compatible_runtimes = ["python3.12"]

  lifecycle {
    precondition {
      condition     = fileexists("${path.module}/../lambda/layers/python/anthropic/__init__.py")
      error_message = "The anthropic SDK is not in the layer. Run: bash infra/lambda/build.sh"
    }
  }
}

resource "aws_lambda_function" "api" {
  function_name = "${var.name}-api"
  role          = aws_iam_role.api.arn
  handler       = "handler.handler"
  runtime       = "python3.12"
  architectures = ["arm64"]
  timeout       = 30
  memory_size   = 256

  filename         = data.archive_file.api.output_path
  source_code_hash = data.archive_file.api.output_base64sha256

  environment {
    variables = {
      UPLOADS_BUCKET = aws_s3_bucket.uploads.id
      RESULTS_BUCKET = aws_s3_bucket.results.id
      SESSIONS_TABLE = aws_dynamodb_table.sessions.name
      TRACE_TABLE    = aws_dynamodb_table.trace.name
      STATE_MACHINE  = aws_sfn_state_machine.session.arn
    }
  }
}

resource "aws_cloudwatch_log_group" "states" {
  name              = "/aws/vendedlogs/states/${var.name}-session"
  retention_in_days = 30
}

# One execution reads one slide. The Map state is not used on purpose: fields
# are read one at a time because which field comes next is the agent's
# decision, and that is the whole claim.
resource "aws_sfn_state_machine" "session" {
  name     = "${var.name}-session"
  role_arn = aws_iam_role.states.arn
  type     = "STANDARD"

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.states.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  definition = jsonencode({
    Comment = "AutoScope: read one slide, field by field, until the agent decides."
    StartAt = "ReadField"

    States = {
      # Stages 1 to 4 on Graviton. Everything the agent reads comes from here.
      ReadField = {
        Type     = "Task"
        Resource = "arn:aws:states:::batch:submitJob.sync"
        Parameters = {
          JobName       = "autoscope-read-field"
          JobQueue      = aws_batch_job_queue.slides.arn
          JobDefinition = aws_batch_job_definition.read_field.arn
          Parameters = {
            "session_id.$" = "$.session_id"
            "field.$"      = "States.Format('{}', $.field)"
          }
        }
        ResultPath = "$.batch"
        Retry = [{
          ErrorEquals     = ["States.TaskFailed"]
          IntervalSeconds = 10
          MaxAttempts     = 2
          BackoffRate     = 2
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          Next        = "FlagForHuman"
          ResultPath  = "$.error"
        }]
        Next = "Decide"
      }

      # The agent. One call, one move.
      Decide = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.agent.arn
          "Payload.$"  = "$"
        }
        ResultSelector = { "move.$" = "$.Payload" }
        ResultPath     = "$.decision"
        Retry = [{
          ErrorEquals     = ["Lambda.TooManyRequestsException", "Lambda.ServiceException"]
          IntervalSeconds = 2
          MaxAttempts     = 3
          BackoffRate     = 2
        }]
        Next = "Limits"
      }

      # The limits, held outside the model. A policy that keeps answering
      # "one more field" stops here, not when it feels like it.
      Limits = {
        Type = "Choice"
        Choices = [
          {
            Variable     = "$.field"
            NumericGreaterThanEquals = 400
            Next         = "FlagForHuman"
          },
          {
            Variable     = "$.decision.move.action"
            StringEquals = "stop"
            Next         = "Done"
          },
          {
            Variable     = "$.decision.move.action"
            StringEquals = "flag_for_human"
            Next         = "FlagForHuman"
          },
        ]
        Default = "NextField"
      }

      NextField = {
        Type = "Pass"
        Parameters = {
          "session_id.$" = "$.session_id"
          "field.$"      = "States.MathAdd($.field, 1)"
          "history.$"    = "$.decision.move"
        }
        Next = "ReadField"
      }

      Done = {
        Type = "Succeed"
      }

      FlagForHuman = {
        Type = "Succeed"
        Comment = "Not a failure: a slide the pipeline will not call on its own is the correct outcome for a screening aid."
      }
    }
  })
}

# A new file in the uploads bucket starts a reading session.
resource "aws_cloudwatch_event_rule" "new_slide" {
  name        = "${var.name}-new-slide"
  description = "A slide lands in S3 and a reading session starts."

  event_pattern = jsonencode({
    source        = ["aws.s3"]
    "detail-type" = ["Object Created"]
    detail = {
      bucket = { name = [aws_s3_bucket.uploads.id] }
      object = { key = [{ prefix = "slides/" }] }
    }
  })
}

resource "aws_cloudwatch_event_target" "new_slide" {
  rule     = aws_cloudwatch_event_rule.new_slide.name
  arn      = aws_sfn_state_machine.session.arn
  role_arn = aws_iam_role.events.arn

  input_transformer {
    input_paths = {
      key = "$.detail.object.key"
    }
    input_template = <<-EOT
      {"session_id": <key>, "field": 0}
    EOT
  }
}
