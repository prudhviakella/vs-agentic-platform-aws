variable "aws_region" {
  description = "AWS region to deploy into"
  default     = "us-east-1"
}

variable "ssm_prefix" {
  description = "SSM Parameter Store prefix for all config"
  default     = "/vs-agentcore/prod"
}

variable "platform_image_uri" {
  description = "ECR URI for Platform FastAPI image"
}

variable "ui_image_uri" {
  # Fix: was "ECR URI for Chainlit UI image" — UI is now FastAPI + vanilla HTML
  description = "ECR URI for FastAPI UI image"
}

variable "postgres_password" {
  description = "RDS master password — set via TF_VAR_postgres_password or .env.prod RDS_PASSWORD"
  sensitive   = true
  # No default — must be set explicitly.
  # deploy.sh exports TF_VAR_postgres_password from RDS_PASSWORD in .env.prod
}
