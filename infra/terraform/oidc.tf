# GitHub Actions authenticates with OIDC, so there is no long-lived key in the
# repository to leak. After apply, put role_arn in the repo as the AWS_ROLE_ARN
# secret; that is the only secret CI needs.

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Only this repository, any branch. Narrow to a branch if CI ever gets
    # permission to do something that matters.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github" {
  name               = "${var.name}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
}

# CI builds the arm64 image and pushes it. It cannot read a slide, cannot call
# Bedrock, and cannot start a session.
data "aws_iam_policy_document" "github" {
  statement {
    sid       = "GetAnECRToken"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "PushTheImage"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:BatchGetImage",
    ]
    resources = [aws_ecr_repository.pipeline.arn]
  }
}

resource "aws_iam_role_policy" "github" {
  name   = "${var.name}-github-actions"
  role   = aws_iam_role.github.id
  policy = data.aws_iam_policy_document.github.json
}
