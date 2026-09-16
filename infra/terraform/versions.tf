# The stack of section 6 of the proposal, as code.
#
#   terraform init && terraform plan && terraform apply
#
# Read docs/aws-setup.md first: three things have to be done by hand in a
# browser (the quota, Bedrock model access, the Marketplace subscription)
# before any of this has a machine to run on.

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # State stays local for now. One person applies, the state file is in
  # .gitignore, and a remote backend is a day of work we do not have. If Bruno
  # starts applying too, move it to S3 with DynamoDB locking before the second
  # person runs anything.
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "autoscope"
      ManagedBy = "terraform"
      Entry     = "opencv-ai-competition-2026"
    }
  }
}
