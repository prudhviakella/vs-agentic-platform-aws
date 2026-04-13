variable "aws_region"          { default = "us-east-1" }
variable "ssm_prefix"          { default = "/vs-agentcore/prod" }
variable "platform_image_uri"  { description = "ECR URI for Platform FastAPI image" }
variable "ui_image_uri"        { description = "ECR URI for Chainlit UI image" }
variable "postgres_password"   { description = "RDS Postgres password"; sensitive = true; default = "changeme123!" }
