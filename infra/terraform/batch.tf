# The image work: AWS Batch on Graviton4 Spot. This is the main workload and
# it runs on Arm. Stages 1 to 4 and 6 of the pipeline happen inside this
# container; the model runs in the same process through cv.dnn, so there is no
# second endpoint to set up, pay for or measure.

# The default VPC is enough: the jobs pull from ECR and read S3, and neither
# needs anything we would have to build. Building a VPC with NAT gateways would
# cost more per month than all the compute in this stack.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_ecr_repository" "pipeline" {
  name                 = "${var.name}-pipeline"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep the last ten images. Every chain run pushes one and they are ~1 GB.
resource "aws_ecr_lifecycle_policy" "pipeline" {
  repository = aws_ecr_repository.pipeline.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_security_group" "batch" {
  name        = "${var.name}-batch"
  description = "AutoScope Batch jobs: outbound only, to ECR and S3."
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Jobs pull the image and read and write S3. Nothing reaches in."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_batch_compute_environment" "graviton_spot" {
  compute_environment_name = "${var.name}-graviton-spot"
  type = "MANAGED"

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"

    # Spot on Graviton4. A slide that loses its instance is re-queued, and the
    # per-field results already written to DynamoDB are not lost, so an
    # interruption costs the fields in flight and nothing else.
    bid_percentage = 100

    min_vcpus     = 0
    desired_vcpus = 0
    max_vcpus     = var.batch_max_vcpus

    instance_type = var.batch_instance_types

    # Empty means the ECS-optimised Arm AMI, which runs our own stock-OpenCV
    # container: the honest control arm. Set cool_ami_id and the same container
    # runs on the COOL build instead, which is the arm we are claiming.
    dynamic "ec2_configuration" {
      for_each = var.cool_ami_id == "" ? [] : [1]
      content {
        image_id_override = var.cool_ami_id
        image_type        = "ECS_AL2023_ARM64"
      }
    }

    subnets            = data.aws_subnets.default.ids
    security_group_ids = [aws_security_group.batch.id]

    instance_role = aws_iam_instance_profile.batch_instance.arn
    spot_iam_fleet_role = aws_iam_role.spot_fleet.arn

    tags = {
      Name = "${var.name}-batch-worker"
    }
  }

  service_role = aws_iam_role.batch_service.arn

  # The compute environment has to be disabled before it can be replaced.
  lifecycle {
    create_before_destroy = true
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service]
}

resource "aws_batch_job_queue" "slides" {
  name     = "${var.name}-slides"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.graviton_spot.arn
  }
}

# One job reads one field: pull the tile from S3, run stages 1 to 4, write the
# numbers to DynamoDB. The agent decides whether there is a next one.
resource "aws_batch_job_definition" "read_field" {
  name = "${var.name}-read-field"
  type = "container"

  platform_capabilities = ["EC2"]

  # A lost Spot instance is the expected case, not an error. Three attempts on
  # a host failure; a job that fails on its own output is not retried, because
  # it will fail the same way again.
  retry_strategy {
    attempts = 3

    evaluate_on_exit {
      action           = "RETRY"
      on_status_reason = "Host EC2*"
    }

    evaluate_on_exit {
      action    = "EXIT"
      on_reason = "*"
    }
  }

  timeout {
    attempt_duration_seconds = 900
  }

  container_properties = jsonencode({
    image      = "${aws_ecr_repository.pipeline.repository_url}:latest"
    jobRoleArn = aws_iam_role.batch_job.arn

    # 4 vCPU and 8 GB is one field comfortably: the 2680 px crop in float32
    # plus the patch batch peaks under 3 GB.
    resourceRequirements = [
      { type = "VCPU", value = "4" },
      { type = "MEMORY", value = "8192" },
    ]

    command = [
      "python", "-m", "autoscope.cloud.read_field",
      "--session", "Ref::session_id",
      "--field", "Ref::field",
    ]

    environment = [
      { name = "AUTOSCOPE_UPLOADS_BUCKET", value = aws_s3_bucket.uploads.id },
      { name = "AUTOSCOPE_RESULTS_BUCKET", value = aws_s3_bucket.results.id },
      { name = "AUTOSCOPE_TRACE_TABLE", value = aws_dynamodb_table.trace.name },
      { name = "AUTOSCOPE_SESSIONS_TABLE", value = aws_dynamodb_table.sessions.name },
      { name = "OPENCV_LOG_LEVEL", value = "ERROR" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "read-field"
      }
    }
  })
}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/${var.name}"
  retention_in_days = 30
}
