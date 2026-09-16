output "api_url" {
  description = "The demo endpoint. This is the URL the submission has to carry."
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "site_url" {
  description = "The page a judge opens."
  value       = "https://${aws_cloudfront_distribution.site.domain_name}"
}

output "uploads_bucket" {
  value = aws_s3_bucket.uploads.id
}

output "results_bucket" {
  value = aws_s3_bucket.results.id
}

output "site_bucket" {
  description = "Sync the built page here, then invalidate CloudFront."
  value       = aws_s3_bucket.site.id
}

output "ecr_repository_url" {
  description = "docker push here; the Batch job definition reads :latest."
  value       = aws_ecr_repository.pipeline.repository_url
}

output "job_queue" {
  value = aws_batch_job_queue.slides.name
}

output "state_machine_arn" {
  value = aws_sfn_state_machine.session.arn
}

output "github_actions_role_arn" {
  description = "Put this in the repo as the AWS_ROLE_ARN secret."
  value       = aws_iam_role.github.arn
}

output "using_cool_ami" {
  description = "False means the Batch fleet is running the stock control arm, not COOL."
  value       = var.cool_ami_id != ""
}
