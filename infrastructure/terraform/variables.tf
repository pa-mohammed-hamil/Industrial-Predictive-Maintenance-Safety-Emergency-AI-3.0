
# ============================================================
# variables.tf
# Industrial Predictive Maintenance, Safety &
# Emergency Intelligence AI
# ============================================================

# ------------------------------------------------------------
# 1. General Configuration
# ------------------------------------------------------------

variable "project_name" {
  description = "Project name used for AWS resource naming"
  type        = string
  default     = "industrial-ai"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"

  validation {
    condition = contains(
      ["dev", "staging", "production"],
      var.environment
    )

    error_message = "Environment must be dev, staging, or production."
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

# ------------------------------------------------------------
# 2. Networking
# ------------------------------------------------------------

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.20.0.0/16"
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnet internet access"
  type        = bool
  default     = true
}

# ------------------------------------------------------------
# 3. Amazon EKS
# ------------------------------------------------------------

variable "eks_version" {
  description = "Kubernetes version supported by Amazon EKS"
  type        = string
  default     = "1.33"
}

variable "eks_public_endpoint" {
  description = "Enable public access to the EKS API endpoint"
  type        = bool
  default     = false
}

variable "eks_public_access_cidrs" {
  description = "Allowed CIDR blocks for the public EKS API endpoint"
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for cidr in var.eks_public_access_cidrs :
      can(cidrhost(cidr, 0))
    ])

    error_message = "Every public endpoint entry must be a valid CIDR."
  }
}

# ------------------------------------------------------------
# 4. General EKS Node Group
# ------------------------------------------------------------

variable "general_instance_types" {
  description = "EC2 instance types for general workloads"
  type        = list(string)
  default     = ["t3.large"]
}

variable "general_desired_nodes" {
  description = "Desired number of general nodes"
  type        = number
  default     = 2
}

variable "general_min_nodes" {
  description = "Minimum general nodes"
  type        = number
  default     = 1
}

variable "general_max_nodes" {
  description = "Maximum general nodes"
  type        = number
  default     = 5
}

# ------------------------------------------------------------
# 5. Machine Learning Node Group
# ------------------------------------------------------------

variable "ml_instance_types" {
  description = "EC2 instance types for ML and RAG workloads"
  type        = list(string)
  default     = ["m6i.xlarge"]
}

variable "ml_desired_nodes" {
  description = "Desired ML worker nodes"
  type        = number
  default     = 1
}

variable "ml_min_nodes" {
  description = "Minimum ML nodes"
  type        = number
  default     = 1
}

variable "ml_max_nodes" {
  description = "Maximum ML nodes"
  type        = number
  default     = 3
}

# ------------------------------------------------------------
# 6. PostgreSQL Database
# ------------------------------------------------------------

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "industrial_ai"
}

variable "db_master_username" {
  description = "PostgreSQL master username"
  type        = string
  default     = "industrial_admin"
}

variable "db_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "16.6"
}

variable "db_parameter_group_family" {
  description = "PostgreSQL parameter group family"
  type        = string
  default     = "postgres16"
}

variable "db_instance_class" {
  description = "RDS instance type"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Initial PostgreSQL storage in GiB"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum autoscaled PostgreSQL storage in GiB"
  type        = number
  default     = 100
}

variable "db_multi_az" {
  description = "Enable RDS Multi-AZ deployment"
  type        = bool
  default     = false
}

variable "db_backup_retention_days" {
  description = "RDS automated backup retention"
  type        = number
  default     = 7
}

variable "db_deletion_protection" {
  description = "Protect RDS from accidental deletion"
  type        = bool
  default     = true
}

variable "db_skip_final_snapshot" {
  description = "Skip final database snapshot when deleting RDS"
  type        = bool
  default     = false
}

variable "db_connection_alarm_threshold" {
  description = "CloudWatch database connection alert threshold"
  type        = number
  default     = 100
}

# ------------------------------------------------------------
# 7. Monitoring and Alerts
# ------------------------------------------------------------

variable "alert_email" {
  description = "Email address for SNS monitoring alerts"
  type        = string
  default     = ""
}

# ------------------------------------------------------------
# 8. Application Configuration
# ------------------------------------------------------------

variable "app_name" {
  description = "Industrial AI application name"
  type        = string
  default     = "Industrial Predictive Maintenance AI"
}

variable "app_namespace" {
  description = "Kubernetes application namespace"
  type        = string
  default     = "industrial-ai"
}

variable "api_replicas" {
  description = "FastAPI replica count"
  type        = number
  default     = 2
}

variable "frontend_replicas" {
  description = "Frontend replica count"
  type        = number
  default     = 2
}

variable "worker_replicas" {
  description = "Celery worker replica count"
  type        = number
  default     = 2
}

# ------------------------------------------------------------
# 9. Feature Flags
# ------------------------------------------------------------

variable "enable_ml_monitoring" {
  description = "Enable ML monitoring integrations"
  type        = bool
  default     = true
}

variable "enable_rag_monitoring" {
  description = "Enable RAG monitoring integrations"
  type        = bool
  default     = true
}

variable "enable_safety_monitoring" {
  description = "Enable safety monitoring integrations"
  type        = bool
  default     = true
}

variable "enable_emergency_monitoring" {
  description = "Enable emergency monitoring integrations"
  type        = bool
  default     = true
}

# ------------------------------------------------------------
# 10. Tags
# ------------------------------------------------------------

variable "additional_tags" {
  description = "Additional AWS resource tags"
  type        = map(string)
  default     = {}
}
