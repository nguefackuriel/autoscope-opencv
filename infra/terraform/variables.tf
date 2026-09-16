variable "region" {
  description = "Everything in one region: c8g, Bedrock and the Marketplace AMI are all in us-east-1."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Prefix for every resource name."
  type        = string
  default     = "autoscope"
}

variable "alert_email" {
  description = "Where the budget alarms go. Confirm the SNS subscription by email or they go nowhere."
  type        = string
}

variable "monthly_budget_usd" {
  description = "Total monthly cost budget. The grant plus free tier is about 250-300 USD for the whole build."
  type        = number
  default     = 150
}

variable "bedrock_budget_usd" {
  description = "Bedrock alone. A retry loop is the one failure that can spend faster than we notice."
  type        = number
  default     = 25
}

variable "bedrock_model_id" {
  description = <<-EOT
    The agent's model on Bedrock. Bedrock ids carry an `anthropic.` prefix.
    Confirm what the account actually offers before applying:
      aws bedrock list-foundation-models --by-provider anthropic \
        --query 'modelSummaries[].modelId' --output table
  EOT
  type        = string
  default     = "anthropic.claude-opus-5"
}

variable "cool_ami_id" {
  description = <<-EOT
    The Cloud Optimized OpenCV (COOL) AMI for Graviton4, from AWS Marketplace,
    in this region. Leave empty and the Batch compute environment falls back to
    the ECS-optimised Arm AMI running our own stock-OpenCV container, which is
    the honest control arm and still a working pipeline. Set it once the
    subscription comes through.
  EOT
  type        = string
  default     = ""
}

variable "batch_max_vcpus" {
  description = "Ceiling on the Spot fleet. Ask Service Quotas for at least this many Spot vCPU first."
  type        = number
  default     = 64
}

variable "batch_instance_types" {
  description = "Graviton4 only. This is the claim: the main workload runs on Arm."
  type        = list(string)
  default     = ["c8g.2xlarge", "c8g.4xlarge", "c8g.8xlarge"]
}

variable "github_repo" {
  description = "owner/name, for the GitHub Actions OIDC trust policy."
  type        = string
  default     = "nguefackuriel/autoscope-opencv"
}

variable "session_ttl_days" {
  description = "How long a reading session's per-field rows live in DynamoDB."
  type        = number
  default     = 90
}
