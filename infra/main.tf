terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket = "vs-agentcore-tfstate"
    key    = "vs-agentcore-platform-aws/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" { region = var.aws_region }

data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  prefix     = "vs-agentcore"
}

# ── VPC ───────────────────────────────────────────────────────────────────

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${local.prefix}-vpc" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "${local.prefix}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Name = "${local.prefix}-private-${count.index}" }
}

resource "aws_internet_gateway" "main" { vpc_id = aws_vpc.main.id }

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route { cidr_block = "0.0.0.0/0"; gateway_id = aws_internet_gateway.main.id }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── ECS Cluster ───────────────────────────────────────────────────────────

resource "aws_ecs_cluster" "main" {
  name = "${local.prefix}-cluster"
  setting { name = "containerInsights"; value = "enabled" }
}

# ── IAM ───────────────────────────────────────────────────────────────────

resource "aws_iam_role" "ecs_task" {
  name = "${local.prefix}-ecs-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_exec" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "ecs-task-policy"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ssm:GetParameter","secretsmanager:GetSecretValue","kms:Decrypt"], Resource = "*" },
      { Effect = "Allow", Action = ["bedrock-agentcore:InvokeAgentRuntime"], Resource = "*" },
      { Effect = "Allow", Action = ["logs:*"], Resource = "*" }
    ]
  })
}

# ── Security Groups ───────────────────────────────────────────────────────

resource "aws_security_group" "alb" {
  name   = "${local.prefix}-alb-sg"
  vpc_id = aws_vpc.main.id
  ingress { from_port = 80;  to_port = 80;  protocol = "tcp"; cidr_blocks = ["0.0.0.0/0"] }
  ingress { from_port = 443; to_port = 443; protocol = "tcp"; cidr_blocks = ["0.0.0.0/0"] }
  egress  { from_port = 0;  to_port = 0;   protocol = "-1";  cidr_blocks = ["0.0.0.0/0"] }
}

resource "aws_security_group" "ecs" {
  name   = "${local.prefix}-ecs-sg"
  vpc_id = aws_vpc.main.id
  ingress { from_port = 0; to_port = 65535; protocol = "tcp"; security_groups = [aws_security_group.alb.id] }
  egress  { from_port = 0; to_port = 0;     protocol = "-1";  cidr_blocks = ["0.0.0.0/0"] }
}

# ── ALB ───────────────────────────────────────────────────────────────────

resource "aws_lb" "main" {
  name               = "${local.prefix}-alb"
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
  security_groups    = [aws_security_group.alb.id]
  idle_timeout       = 300   # SSE streams need longer timeout
}

resource "aws_lb_target_group" "platform" {
  name        = "${local.prefix}-platform"
  port        = 8000; protocol = "HTTP"; vpc_id = aws_vpc.main.id; target_type = "ip"
  health_check { path = "/health"; timeout = 10; interval = 30; healthy_threshold = 2; unhealthy_threshold = 3 }
}

resource "aws_lb_target_group" "ui" {
  name        = "${local.prefix}-ui"
  port        = 8501; protocol = "HTTP"; vpc_id = aws_vpc.main.id; target_type = "ip"
  health_check { path = "/health"; timeout = 10; interval = 30 }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port = 80; protocol = "HTTP"
  default_action { type = "forward"; target_group_arn = aws_lb_target_group.ui.arn }
}

resource "aws_lb_listener_rule" "platform_api" {
  listener_arn = aws_lb_listener.http.arn; priority = 10
  action { type = "forward"; target_group_arn = aws_lb_target_group.platform.arn }
  condition { path_pattern { values = ["/api/*"] } }
}

# ── RDS Postgres (LangGraph checkpointer) ────────────────────────────────

resource "aws_db_subnet_group" "main" {
  name       = "${local.prefix}-db-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_security_group" "rds" {
  name   = "${local.prefix}-rds-sg"
  vpc_id = aws_vpc.main.id
  ingress { from_port = 5432; to_port = 5432; protocol = "tcp"; security_groups = [aws_security_group.ecs.id] }
}

resource "aws_db_instance" "postgres" {
  identifier             = "${local.prefix}-postgres"
  engine                 = "postgres"; engine_version = "15.4"
  instance_class         = "db.t3.micro"; allocated_storage = 20
  db_name                = "clinical_agent"; username = "postgres"
  password               = var.postgres_password
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  skip_final_snapshot    = true; publicly_accessible = false
  tags = { Name = "${local.prefix}-postgres" }
}

# ── ECS: Platform ─────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "platform" {
  name              = "/ecs/${local.prefix}/platform"
  retention_in_days = 7
}

resource "aws_ecs_task_definition" "platform" {
  family                   = "${local.prefix}-platform"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"; memory = "1024"
  execution_role_arn       = aws_iam_role.ecs_task.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "platform"
    image = var.platform_image_uri
    portMappings = [{ containerPort = 8000 }]
    environment = [
      { name = "AWS_REGION",  value = var.aws_region },
      { name = "SSM_PREFIX",  value = var.ssm_prefix },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options   = { "awslogs-group" = aws_cloudwatch_log_group.platform.name, "awslogs-region" = var.aws_region, "awslogs-stream-prefix" = "platform" }
    }
  }])
}

resource "aws_ecs_service" "platform" {
  name = "${local.prefix}-platform"; cluster = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.platform.arn; desired_count = 1; launch_type = "FARGATE"
  network_configuration { subnets = aws_subnet.public[*].id; security_groups = [aws_security_group.ecs.id]; assign_public_ip = true }
  load_balancer { target_group_arn = aws_lb_target_group.platform.arn; container_name = "platform"; container_port = 8000 }
}

# ── ECS: UI ───────────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "ui" {
  name              = "/ecs/${local.prefix}/ui"
  retention_in_days = 7
}

resource "aws_ecs_task_definition" "ui" {
  family                   = "${local.prefix}-ui"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"; memory = "512"
  execution_role_arn       = aws_iam_role.ecs_task.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "ui"
    image = var.ui_image_uri
    portMappings = [{ containerPort = 8501 }]
    environment = [
      { name = "AGENT_API_URL", value = "http://${aws_lb.main.dns_name}" },
      { name = "AGENT_DOMAIN",  value = "pharma" },
    ]
    secrets = [{ name = "AGENT_API_KEY", valueFrom = "${var.ssm_prefix}/platform_api_key" }]
    logConfiguration = {
      logDriver = "awslogs"
      options   = { "awslogs-group" = aws_cloudwatch_log_group.ui.name, "awslogs-region" = var.aws_region, "awslogs-stream-prefix" = "ui" }
    }
  }])
}

resource "aws_ecs_service" "ui" {
  name = "${local.prefix}-ui"; cluster = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.ui.arn; desired_count = 1; launch_type = "FARGATE"
  network_configuration { subnets = aws_subnet.public[*].id; security_groups = [aws_security_group.ecs.id]; assign_public_ip = true }
  load_balancer { target_group_arn = aws_lb_target_group.ui.arn; container_name = "ui"; container_port = 8501 }
}

# ── Outputs ───────────────────────────────────────────────────────────────

output "alb_dns" { value = aws_lb.main.dns_name }
output "rds_endpoint" { value = aws_db_instance.postgres.endpoint }
