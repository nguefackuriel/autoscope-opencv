# S3 and DynamoDB: where slides, results and the audit trail live.

data "aws_caller_identity" "current" {}

locals {
  account_id  = data.aws_caller_identity.current.account_id
  bucket_base = "${var.name}-${local.account_id}"
}

# --- Slides in, results out -------------------------------------------------

# The phone uploads straight here with a presigned POST. Nothing sits between
# the phone and S3, which is the only part of the design that has to work on a
# bad connection in a district hospital.
resource "aws_s3_bucket" "uploads" {
  bucket = "${local.bucket_base}-uploads"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket" "results" {
  bucket = "${local.bucket_base}-results"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_public_access_block" "uploads" {
  bucket                  = aws_s3_bucket.uploads.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "results" {
  bucket                  = aws_s3_bucket.results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# A slide video is 100 MB and we are not a storage company. Uploads age out;
# results, which are small JSON and marked-up crops, stay for the judging.
resource "aws_s3_bucket_lifecycle_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id

  rule {
    id     = "expire-raw-uploads"
    status = "Enabled"

    filter {}

    expiration {
      days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# The phone page is served from CloudFront on another origin, so the browser
# posting a file here is cross-origin.
resource "aws_s3_bucket_cors_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["PUT", "POST", "GET"]
    allowed_origins = ["https://${aws_cloudfront_distribution.site.domain_name}"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# EventBridge has to be told to emit for a bucket before a rule can match on it.
resource "aws_s3_bucket_notification" "uploads" {
  bucket      = aws_s3_bucket.uploads.id
  eventbridge = true
}

# --- The static page --------------------------------------------------------

resource "aws_s3_bucket" "site" {
  bucket = "${local.bucket_base}-site"
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket                  = aws_s3_bucket.site.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Session state and the audit trail --------------------------------------

# One row per session, holding the decision and the totals.
resource "aws_dynamodb_table" "sessions" {
  name         = "${var.name}-sessions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"

  attribute {
    name = "session_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }
}

# One row per field read, and one per agent decision. This is the audit trail:
# what was looked at, what OpenCV measured, what the agent did about it and
# why. A screening aid that cannot show its working is not one a lab can use.
resource "aws_dynamodb_table" "trace" {
  name         = "${var.name}-trace"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"
  range_key    = "step"

  attribute {
    name = "session_id"
    type = "S"
  }

  attribute {
    name = "step"
    type = "N"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}
