# ============================================================
# database.tf
# Industrial AI - Amazon RDS PostgreSQL
# ============================================================

locals {
  db_identifier = "${var.project_name}-${var.environment}-postgres"
}

# ------------------------------------------------------------
# Database Subnet Group
# ------------------------------------------------------------

resource "aws_db_subnet_group" "industrial_ai" {
  name       = "${local.db_identifier}-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name        = "${local.db_identifier}-subnet-group"
    Project     = var.project_name
    Environment = var.environment
  }
}

# ------------------------------------------------------------
# Database Security Group
# ------------------------------------------------------------

resource "aws_security_group" "database" {
  name_prefix = "${local.db_identifier}-"
  description = "PostgreSQL access from Industrial AI EKS workloads"
  vpc_id      = var.vpc_id

  tags = {
    Name        = "${local.db_identifier}-sg"
    Project     = var.project_name
    Environment = var.environment
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Allow PostgreSQL access from EKS cluster security group
resource "aws_vpc_security_group_ingress_rule" "postgres_from_eks" {
  security_group_id            = aws_security_group.database.id
  referenced_security_group_id = aws_eks_cluster.industrial_ai.vpc_config[0].cluster_security_group_id

  from_port   = 5432
  to_port     = 5432
  ip_protocol = "tcp"

  description = "PostgreSQL access from EKS"
}

# ------------------------------------------------------------
# PostgreSQL Parameter Group
# ------------------------------------------------------------

resource "aws_db_parameter_group" "industrial_ai" {
  name_prefix = "${local.db_identifier}-"
  family      = var.db_parameter_group_family

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  parameter {
    name  = "log_connections"
    value = "1"
  }

  parameter {
    name  = "log_disconnections"
    value = "1"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ------------------------------------------------------------
# PostgreSQL RDS Instance
# ------------------------------------------------------------

resource "aws_db_instance" "industrial_ai" {
  identifier = local.db_identifier

  engine         = "postgres"
  engine_version = var.db_engine_version
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = var.db_name
  username = var.db_master_username

  # AWS RDS manages the password in Secrets Manager.
  manage_master_user_password = true

  port = 5432

  db_subnet_group_name   = aws_db_subnet_group.industrial_ai.name
  vpc_security_group_ids = [aws_security_group.database.id]

  publicly_accessible = false

  parameter_group_name = aws_db_parameter_group.industrial_ai.name

  multi_az = var.db_multi_az

  backup_retention_period = var.db_backup_retention_days
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  auto_minor_version_upgrade = true
  copy_tags_to_snapshot      = true

  enabled_cloudwatch_logs_exports = [
    "postgresql",
    "upgrade"
  ]

  deletion_protection = var.db_deletion_protection
  skip_final_snapshot = var.db_skip_final_snapshot

  final_snapshot_identifier = (
    var.db_skip_final_snapshot
    ? null
    : "${local.db_identifier}-final"
  )

  apply_immediately = false

  tags = {
    Name        = local.db_identifier
    Project     = var.project_name
    Environment = var.environment
    Component   = "database"
  }
}

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

output "database_endpoint" {
  description = "RDS PostgreSQL hostname and port"
  value       = aws_db_instance.industrial_ai.endpoint
}

output "database_host" {
  description = "RDS PostgreSQL hostname"
  value       = aws_db_instance.industrial_ai.address
}

output "database_port" {
  description = "RDS PostgreSQL port"
  value       = aws_db_instance.industrial_ai.port
}

output "database_name" {
  description = "Application database name"
  value       = aws_db_instance.industrial_ai.db_name
}

output "database_security_group_id" {
  value = aws_security_group.database.id
}

output "database_master_secret_arn" {
  description = "AWS Secrets Manager ARN for RDS master credentials"
  value       = aws_db_instance.industrial_ai.master_user_secret[0].secret_arn
  sensitive   = true
}