# ============================================================
# main.tf
# Industrial Predictive Maintenance, Safety &
# Emergency Intelligence AI
#
# Infrastructure: AWS + EKS + RDS + S3
# ============================================================

terraform {

  required_version = ">= 1.6.0"

  required_providers {

    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }

    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }

    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }

    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }

    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

# ============================================================
# AWS Provider
# ============================================================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
      Application = "Industrial-AI"
    }
  }
}

# ============================================================
# Common Configuration
# ============================================================

locals {

  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Application = "Industrial-AI"
  }

  application = {
    name        = "Industrial Predictive Maintenance AI"
    environment = var.environment
    region      = var.aws_region
  }

  services = {
    api      = "industrial-ai-api"
    frontend = "industrial-ai-frontend"
    worker   = "industrial-ai-worker"
    redis    = "redis"
    kafka    = "kafka"
    qdrant   = "qdrant"
  }
}

# ============================================================
# EKS Authentication
# ============================================================

data "aws_eks_cluster_auth" "industrial_ai" {
  name = aws_eks_cluster.industrial_ai.name
}

# ============================================================
# Kubernetes Provider
# ============================================================

provider "kubernetes" {

  host = aws_eks_cluster.industrial_ai.endpoint

  cluster_ca_certificate = base64decode(
    aws_eks_cluster.industrial_ai
    .certificate_authority[0].data
  )

  token = data.aws_eks_cluster_auth.industrial_ai.token
}

# ============================================================
# Helm Provider
# ============================================================

provider "helm" {

  kubernetes {

    host = aws_eks_cluster.industrial_ai.endpoint

    cluster_ca_certificate = base64decode(
      aws_eks_cluster.industrial_ai
      .certificate_authority[0].data
    )

    token = data.aws_eks_cluster_auth.industrial_ai.token
  }
}

# ============================================================
# AWS Account Information
# ============================================================

data "aws_caller_identity" "current_account" {}

data "aws_partition" "current" {}

data "aws_region" "current" {}

# ============================================================
# AWS Availability Zones
# ============================================================

data "aws_availability_zones" "available" {
  state = "available"
}

# ============================================================
# Project Information Outputs
# ============================================================

output "project_information" {

  description = "Industrial AI infrastructure information"

  value = {
    project     = var.project_name
    environment = var.environment
    region      = var.aws_region

    aws_account_id = data.aws_caller_identity.current_account.account_id

    eks_cluster = aws_eks_cluster.industrial_ai.name

    database_host = aws_db_instance.industrial_ai.address
  }
}

output "aws_account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current_account.account_id
}

output "deployment_region" {
  description = "AWS deployment region"
  value       = var.aws_region
}

output "infrastructure_name" {
  description = "Infrastructure resource prefix"
  value       = local.name_prefix
}