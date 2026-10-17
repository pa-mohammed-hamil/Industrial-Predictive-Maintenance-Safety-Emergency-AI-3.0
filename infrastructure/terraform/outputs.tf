
# ============================================================
# outputs.tf
# Industrial Predictive Maintenance, Safety &
# Emergency Intelligence AI
# AWS Infrastructure Outputs
# ============================================================

# ------------------------------------------------------------
# 1. Project Information
# ------------------------------------------------------------

output "project_environment" {
  description = "Deployment environment"
  value       = var.environment
}

output "project_region" {
  description = "AWS deployment region"
  value       = var.aws_region
}

# ------------------------------------------------------------
# 2. VPC Networking
# ------------------------------------------------------------

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = aws_vpc.industrial_ai.cidr_block
}

output "vpc_availability_zones" {
  description = "Availability zones used by private subnets"
  value       = aws_subnet.private[*].availability_zone
}

output "public_subnet_details" {
  description = "Public subnet information"

  value = [
    for subnet in aws_subnet.public : {
      id                = subnet.id
      cidr_block        = subnet.cidr_block
      availability_zone = subnet.availability_zone
    }
  ]
}

output "private_subnet_details" {
  description = "Private subnet information"

  value = [
    for subnet in aws_subnet.private : {
      id                = subnet.id
      cidr_block        = subnet.cidr_block
      availability_zone = subnet.availability_zone
    }
  ]
}

output "internet_gateway_id" {
  description = "VPC internet gateway"
  value       = aws_internet_gateway.industrial_ai.id
}

output "public_route_table_id" {
  description = "Public route table"
  value       = aws_route_table.public.id
}

output "private_route_table_id" {
  description = "Private route table"
  value       = aws_route_table.private.id
}

# ------------------------------------------------------------
# 3. Amazon EKS
# ------------------------------------------------------------

output "eks_cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.industrial_ai.arn
}

output "eks_cluster_version" {
  description = "Kubernetes version"
  value       = aws_eks_cluster.industrial_ai.version
}

output "eks_cluster_status" {
  description = "EKS cluster status at last Terraform refresh"
  value       = aws_eks_cluster.industrial_ai.status
}

output "eks_oidc_issuer_url" {
  description = "EKS OIDC issuer"
  value       = aws_eks_cluster.industrial_ai.identity[0].oidc[0].issuer
}

output "eks_general_node_group_arn" {
  description = "General workload node group ARN"
  value       = aws_eks_node_group.general.arn
}

output "eks_ml_node_group_arn" {
  description = "ML workload node group ARN"
  value       = aws_eks_node_group.ml.arn
}

output "eks_general_instance_types" {
  description = "General worker node instance types"
  value       = aws_eks_node_group.general.instance_types
}

output "eks_ml_instance_types" {
  description = "ML worker node instance types"
  value       = aws_eks_node_group.ml.instance_types
}

# ------------------------------------------------------------
# 4. PostgreSQL RDS
# ------------------------------------------------------------

output "rds_instance_identifier" {
  description = "PostgreSQL RDS instance identifier"
  value       = aws_db_instance.industrial_ai.identifier
}

output "rds_instance_arn" {
  description = "PostgreSQL RDS ARN"
  value       = aws_db_instance.industrial_ai.arn
}

output "rds_engine_version" {
  description = "Actual PostgreSQL engine version"
  value       = aws_db_instance.industrial_ai.engine_version_actual
}

output "rds_instance_class" {
  description = "Database instance class"
  value       = aws_db_instance.industrial_ai.instance_class
}

output "rds_storage_encrypted" {
  description = "RDS storage encryption status"
  value       = aws_db_instance.industrial_ai.storage_encrypted
}

output "rds_public_access" {
  description = "Whether database has public access"
  value       = aws_db_instance.industrial_ai.publicly_accessible
}

output "rds_backup_retention_days" {
  description = "Automated backup retention period"
  value       = aws_db_instance.industrial_ai.backup_retention_period
}

output "rds_multi_az_enabled" {
  description = "RDS Multi-AZ configuration"
  value       = aws_db_instance.industrial_ai.multi_az
}

# ------------------------------------------------------------
# 5. CloudWatch Monitoring
# ------------------------------------------------------------

output "application_log_group_name" {
  description = "Application CloudWatch log group"
  value       = aws_cloudwatch_log_group.application.name
}

output "worker_log_group_name" {
  description = "Celery worker CloudWatch log group"
  value       = aws_cloudwatch_log_group.workers.name
}

output "ml_log_group_name" {
  description = "ML monitoring log group"
  value       = aws_cloudwatch_log_group.ml.name
}

output "rag_log_group_name" {
  description = "RAG monitoring log group"
  value       = aws_cloudwatch_log_group.rag.name
}

output "safety_log_group_name" {
  description = "Safety monitoring log group"
  value       = aws_cloudwatch_log_group.safety.name
}

output "vpc_flow_log_id" {
  description = "VPC Flow Logs ID"
  value       = aws_flow_log.industrial_ai.id
}

# ------------------------------------------------------------
# 6. Alerting
# ------------------------------------------------------------

output "sns_alert_topic_name" {
  description = "SNS alert topic name"
  value       = aws_sns_topic.industrial_alerts.name
}

# ------------------------------------------------------------
# 7. Kubernetes Connection
# ------------------------------------------------------------

output "kubectl_configure_command" {
  description = "Command to configure kubectl"

  value = format(
    "aws eks update-kubeconfig --region %s --name %s",
    var.aws_region,
    aws_eks_cluster.industrial_ai.name
  )
}

# ------------------------------------------------------------
# 8. Infrastructure Summary
# ------------------------------------------------------------

output "industrial_ai_deployment_summary" {
  description = "Complete Industrial AI deployment summary"

  value = {
    project     = var.project_name
    environment = var.environment
    region      = var.aws_region

    networking = {
      vpc_id          = aws_vpc.industrial_ai.id
      public_subnets  = aws_subnet.public[*].id
      private_subnets = aws_subnet.private[*].id
    }

    eks = {
      cluster_name = aws_eks_cluster.industrial_ai.name
      endpoint     = aws_eks_cluster.industrial_ai.endpoint

      general_node_group = aws_eks_node_group.general.node_group_name
      ml_node_group      = aws_eks_node_group.ml.node_group_name
    }

    database = {
      host     = aws_db_instance.industrial_ai.address
      port     = aws_db_instance.industrial_ai.port
      name     = aws_db_instance.industrial_ai.db_name
      username = aws_db_instance.industrial_ai.username
    }

    monitoring = {
      dashboard = aws_cloudwatch_dashboard.industrial_ai.dashboard_name
      sns_topic = aws_sns_topic.industrial_alerts.arn
    }
  }
}
