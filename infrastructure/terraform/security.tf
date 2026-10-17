
# ============================================================
# security.tf
# Industrial AI - AWS Security Infrastructure
# ============================================================

data "aws_caller_identity" "security" {}

data "aws_partition" "security" {}

# ------------------------------------------------------------
# 1. KMS Encryption Key
# ------------------------------------------------------------

resource "aws_kms_key" "industrial_ai" {
  description = "KMS key for Industrial AI application data"

  enable_key_rotation     = true
  deletion_window_in_days = 30

  tags = {
    Name        = "${var.project_name}-${var.environment}-kms"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_kms_alias" "industrial_ai" {
  name          = "alias/${var.project_name}-${var.environment}-ai"
  target_key_id = aws_kms_key.industrial_ai.key_id
}

# ------------------------------------------------------------
# 2. Application IAM Policy
# ------------------------------------------------------------

data "aws_iam_policy_document" "application" {

  statement {
    sid = "ReadApplicationSecrets"

    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret"
    ]

    resources = [
      aws_db_instance.industrial_ai.master_user_secret[0].secret_arn
    ]
  }

  statement {
    sid = "PublishMonitoringMetrics"

    actions = [
      "cloudwatch:PutMetricData"
    ]

    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"

      values = [
        "IndustrialAI/ML",
        "IndustrialAI/RAG",
        "IndustrialAI/Safety",
        "IndustrialAI/Emergency",
        "IndustrialAI/Application"
      ]
    }
  }

  statement {
    sid = "PublishNotifications"

    actions = [
      "sns:Publish"
    ]

    resources = [
      aws_sns_topic.industrial_alerts.arn
    ]
  }
}

resource "aws_iam_policy" "application" {
  name = "${var.project_name}-${var.environment}-application-policy"

  description = "Least privilege access for Industrial AI application"

  policy = data.aws_iam_policy_document.application.json
}

# ------------------------------------------------------------
# 3. EKS Service Account IAM Role
# ------------------------------------------------------------

data "aws_iam_policy_document" "eks_service_account_assume" {

  statement {
    effect = "Allow"

    actions = [
      "sts:AssumeRoleWithWebIdentity"
    ]

    principals {
      type = "Federated"

      identifiers = [
        aws_iam_openid_connect_provider.eks.arn
      ]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(
        aws_eks_cluster.industrial_ai.identity[0].oidc[0].issuer,
        "https://",
        ""
      )}:aud"

      values = [
        "sts.amazonaws.com"
      ]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(
        aws_eks_cluster.industrial_ai.identity[0].oidc[0].issuer,
        "https://",
        ""
      )}:sub"

      values = [
        "system:serviceaccount:industrial-ai:industrial-ai-app"
      ]
    }
  }
}

resource "aws_iam_role" "application" {
  name = "${var.project_name}-${var.environment}-app-role"

  assume_role_policy = data.aws_iam_policy_document.eks_service_account_assume.json

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "application" {
  role       = aws_iam_role.application.name
  policy_arn = aws_iam_policy.application.arn
}

# ------------------------------------------------------------
# 4. CloudTrail S3 Bucket
# ------------------------------------------------------------

resource "aws_s3_bucket" "cloudtrail" {
  bucket_prefix = "${var.project_name}-${var.environment}-audit-"

  force_destroy = false

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "audit"
  }
}

resource "aws_s3_bucket_public_access_block" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ------------------------------------------------------------
# 5. CloudTrail Bucket Policy
# ------------------------------------------------------------

data "aws_iam_policy_document" "cloudtrail_bucket" {

  statement {
    sid = "CloudTrailAclCheck"

    principals {
      type = "Service"
      identifiers = [
        "cloudtrail.amazonaws.com"
      ]
    }

    actions = [
      "s3:GetBucketAcl"
    ]

    resources = [
      aws_s3_bucket.cloudtrail.arn
    ]

    condition {
      test     = "StringEquals"
      variable = "aws:SourceArn"

      values = [
        aws_cloudtrail.industrial_ai.arn
      ]
    }
  }

  statement {
    sid = "CloudTrailWrite"

    principals {
      type = "Service"
      identifiers = [
        "cloudtrail.amazonaws.com"
      ]
    }

    actions = [
      "s3:PutObject"
    ]

    resources = [
      "${aws_s3_bucket.cloudtrail.arn}/AWSLogs/${data.aws_caller_identity.security.account_id}/*"
    ]

    condition {
      test     = "StringEquals"
      variable = "s3:x-amz-acl"

      values = [
        "bucket-owner-full-control"
      ]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceArn"

      values = [
        aws_cloudtrail.industrial_ai.arn
      ]
    }
  }
}

resource "aws_s3_bucket_policy" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  policy = data.aws_iam_policy_document.cloudtrail_bucket.json
}

# ------------------------------------------------------------
# 6. AWS CloudTrail
# ------------------------------------------------------------

resource "aws_cloudtrail" "industrial_ai" {
  name = "${var.project_name}-${var.environment}-cloudtrail"

  s3_bucket_name = aws_s3_bucket.cloudtrail.id

  include_global_service_events = true
  is_multi_region_trail         = true

  enable_logging                = true
  enable_log_file_validation    = true

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "security-audit"
  }

  depends_on = [
    aws_s3_bucket_policy.cloudtrail
  ]
}

# ------------------------------------------------------------
# 7. Security Outputs
# ------------------------------------------------------------

output "security_kms_key_arn" {
  description = "KMS key ARN"
  value       = aws_kms_key.industrial_ai.arn
}

output "security_application_role_arn" {
  description = "IAM role for Industrial AI Kubernetes workloads"
  value       = aws_iam_role.application.arn
}

output "security_application_policy_arn" {
  description = "IAM policy ARN"
  value       = aws_iam_policy.application.arn
}

output "security_cloudtrail_arn" {
  description = "CloudTrail ARN"
  value       = aws_cloudtrail.industrial_ai.arn
}

output "security_cloudtrail_bucket" {
  description = "CloudTrail audit bucket"
  value       = aws_s3_bucket.cloudtrail.id
}
