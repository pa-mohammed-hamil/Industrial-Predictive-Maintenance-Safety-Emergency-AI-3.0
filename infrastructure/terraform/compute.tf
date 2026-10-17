# ============================================================
# compute.tf
# AWS EKS Cluster + Managed Node Groups
# ============================================================

data "aws_caller_identity" "current" {}

# ------------------------------------------------------------
# IAM Role - EKS Cluster
# ------------------------------------------------------------

resource "aws_iam_role" "eks_cluster" {
  name = "${var.project_name}-${var.environment}-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

# ------------------------------------------------------------
# IAM Role - Worker Nodes
# ------------------------------------------------------------

resource "aws_iam_role" "eks_nodes" {
  name = "${var.project_name}-${var.environment}-eks-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "worker_node_policy" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "worker_ecr_policy" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly"
}

resource "aws_iam_role_policy_attachment" "worker_cni_policy" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

# ------------------------------------------------------------
# EKS Cluster
# ------------------------------------------------------------

resource "aws_eks_cluster" "industrial_ai" {
  name     = "${var.project_name}-${var.environment}-eks"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.eks_version

  vpc_config {
    subnet_ids = var.private_subnet_ids

    endpoint_private_access = true
    endpoint_public_access  = var.eks_public_endpoint

    public_access_cidrs = var.eks_public_endpoint ? var.eks_public_access_cidrs : null
  }

  enabled_cluster_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]

  access_config {
    authentication_mode = "API_AND_CONFIG_MAP"
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-eks"
    Project     = var.project_name
    Environment = var.environment
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy
  ]
}

# ------------------------------------------------------------
# General-Purpose Node Group
# FastAPI, Frontend, Redis, Celery
# ------------------------------------------------------------

resource "aws_eks_node_group" "general" {
  cluster_name    = aws_eks_cluster.industrial_ai.name
  node_group_name = "${var.project_name}-${var.environment}-general"

  node_role_arn = aws_iam_role.eks_nodes.arn

  subnet_ids = var.private_subnet_ids

  instance_types = var.general_instance_types
  capacity_type  = "ON_DEMAND"

  scaling_config {
    desired_size = var.general_desired_nodes
    min_size     = var.general_min_nodes
    max_size     = var.general_max_nodes
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    workload = "general"
    project  = var.project_name
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-general"
    Project     = var.project_name
    Environment = var.environment
  }

  depends_on = [
    aws_iam_role_policy_attachment.worker_node_policy,
    aws_iam_role_policy_attachment.worker_ecr_policy,
    aws_iam_role_policy_attachment.worker_cni_policy
  ]
}

# ------------------------------------------------------------
# ML / AI Node Group
# CPU-based ML Inference and RAG Workloads
# ------------------------------------------------------------

resource "aws_eks_node_group" "ml" {
  cluster_name    = aws_eks_cluster.industrial_ai.name
  node_group_name = "${var.project_name}-${var.environment}-ml"

  node_role_arn = aws_iam_role.eks_nodes.arn

  subnet_ids = var.private_subnet_ids

  instance_types = var.ml_instance_types
  capacity_type  = "ON_DEMAND"

  scaling_config {
    desired_size = var.ml_desired_nodes
    min_size     = var.ml_min_nodes
    max_size     = var.ml_max_nodes
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    workload = "ml"
    project  = var.project_name
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-ml"
    Project     = var.project_name
    Environment = var.environment
  }

  depends_on = [
    aws_iam_role_policy_attachment.worker_node_policy,
    aws_iam_role_policy_attachment.worker_ecr_policy,
    aws_iam_role_policy_attachment.worker_cni_policy
  ]
}

# ------------------------------------------------------------
# EKS OIDC Provider
# Used for IAM Roles for Service Accounts (IRSA)
# ------------------------------------------------------------

data "tls_certificate" "eks_oidc" {
  url = aws_eks_cluster.industrial_ai.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  url = aws_eks_cluster.industrial_ai.identity[0].oidc[0].issuer

  client_id_list = [
    "sts.amazonaws.com"
  ]

  thumbprint_list = [
    data.tls_certificate.eks_oidc.certificates[0].sha1_fingerprint
  ]

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# ------------------------------------------------------------
# CloudWatch Log Group
# ------------------------------------------------------------

resource "aws_cloudwatch_log_group" "eks" {
  name              = "/aws/eks/${var.project_name}-${var.environment}-eks/cluster"
  retention_in_days = 30

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

output "eks_cluster_name" {
  value = aws_eks_cluster.industrial_ai.name
}

output "eks_cluster_endpoint" {
  value = aws_eks_cluster.industrial_ai.endpoint
}

output "eks_cluster_ca_certificate" {
  value     = aws_eks_cluster.industrial_ai.certificate_authority[0].data
  sensitive = true
}

output "eks_cluster_security_group_id" {
  value = aws_eks_cluster.industrial_ai.vpc_config[0].cluster_security_group_id
}

output "eks_oidc_provider_arn" {
  value = aws_iam_openid_connect_provider.eks.arn
}

output "general_node_group_name" {
  value = aws_eks_node_group.general.node_group_name
}

output "ml_node_group_name" {
  value = aws_eks_node_group.ml.node_group_name
}   