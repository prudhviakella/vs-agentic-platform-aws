variable "aws_region"         { default = "us-east-1" }
variable "ssm_prefix"         { default = "/vs-agentcore/prod" }
variable "platform_image_uri" { description = "ECR URI for Platform FastAPI image" }
variable "ui_image_uri"       { description = "ECR URI for Chainlit UI image" }
variable "postgres_password"  {
  description = "RDS master password — set via TF_VAR_postgres_password or .env.prod RDS_PASSWORD"
  sensitive   = true
  # No default — must be set explicitly. deploy.sh exports TF_VAR_postgres_password from RDS_PASSWORD.
}
