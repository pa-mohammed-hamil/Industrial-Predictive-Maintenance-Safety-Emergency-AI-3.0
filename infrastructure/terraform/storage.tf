
# ============================================================
# storage.tf
# Industrial AI - AWS S3 Storage Infrastructure
# ============================================================

locals {
  storage_buckets = {
    telemetry = {
      description = "Industrial IoT telemetry and sensor datasets"
      retention   = 90
    }

    datasets = {
      description = "ML, DL, CV training and evaluation datasets"
      retention   = 180
    }

    models = {
      description = "Trained models and ML artifacts"
      retention   = 365
    }

    inspections = {
      description = "Equipment inspection images and videos"
      retention   = 180
    }

    documents = {
      description = "Equipment manuals and RAG documents"
      retention   = 365
    }

    reports = {
      description = "Maintenance, safety and cost reports"
      retention   = 365
    }

    backups = {
      description = "Application backup objects"
      retention   = 365
    }
  }
}

# ------------------------------------------------------------
# 1. S3 Buckets
# ------------------------------------------------------------

resource "aws_s3_bucket" "industrial_storage" {
  for_each = local.storage_buckets

  bucket_prefix = "${var.project_name}-${var.environment}-${each.key}-"

  force_destroy = false

  tags = {
    Name        = "${var.project_name}-${var.environment}-${each.key}"
    Project     = var.project_name
    Environment = var.environment
    Component   = each.key
    Description = each.value.description
  }
}

# ------------------------------------------------------------
# 2. Block Public Access
# ------------------------------------------------------------

resource "aws_s3_bucket_public_access_block" "industrial_storage" {
  for_each = aws_s3_bucket.industrial_storage

  bucket = each.value.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ------------------------------------------------------------
# 3. Bucket Ownership
# ------------------------------------------------------------

resource "aws_s3_bucket_ownership_controls" "industrial_storage" {
  for_each = aws_s3_bucket.industrial_storage

  bucket = each.value.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# ------------------------------------------------------------
# 4. Server-Side Encryption
# ------------------------------------------------------------

resource "aws_s3_bucket_server_side_encryption_configuration" "industrial_storage" {
  for_each = aws_s3_bucket.industrial_storage

  bucket = each.value.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.industrial_ai.arn
    }

    bucket_key_enabled = true
  }
}

# ------------------------------------------------------------
# 5. Enable Versioning
# ------------------------------------------------------------

resource "aws_s3_bucket_versioning" "industrial_storage" {
  for_each = aws_s3_bucket.industrial_storage

  bucket = each.value.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ------------------------------------------------------------
# 6. Lifecycle Management
# ------------------------------------------------------------

resource "aws_s3_bucket_lifecycle_configuration" "industrial_storage" {
  for_each = aws_s3_bucket.industrial_storage

  bucket = each.value.id

  rule {
    id     = "transition-old-objects"
    status = "Enabled"

    filter {}

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }
  }

  rule {
    id     = "cleanup-incomplete-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = each.value.retention
    }
  }
}

# ------------------------------------------------------------
# 7. Enforce HTTPS for S3
# ------------------------------------------------------------

data "aws_iam_policy_document" "s3_https" {
  for_each = aws_s3_bucket.industrial_storage

  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = [
      "s3:*"
    ]

    resources = [
      each.value.arn,
      "${each.value.arn}/*"
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "industrial_storage" {
  for_each = aws_s3_bucket.industrial_storage

  bucket = each.value.id
  policy = data.aws_iam_policy_document.s3_https[each.key].json

  depends_on = [
    aws_s3_bucket_public_access_block.industrial_storage
  ]
}

# ------------------------------------------------------------
# 8. IAM Policy for Application Storage Access
# ------------------------------------------------------------

data "aws_iam_policy_document" "storage_access" {
  statement {
    sid = "ListIndustrialBuckets"

    effect = "Allow"

    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ]

    resources = [
      for bucket in aws_s3_bucket.industrial_storage :
      bucket.arn
    ]
  }

  statement {
    sid = "ReadWriteIndustrialObjects"

    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:AbortMultipartUpload"
    ]

    resources = [
      for bucket in aws_s3_bucket.industrial_storage :
      "${bucket.arn}/*"
    ]
  }

  statement {
    sid = "UseIndustrialKMSKey"

    effect = "Allow"

    actions = [
      "kms:Decrypt",
      "kms:Encrypt",
      "kms:GenerateDataKey",
      "kms:DescribeKey"
    ]

    resources = [
      aws_kms_key.industrial_ai.arn
    ]
  }
}

resource "aws_iam_policy" "storage_access" {
  name = "${var.project_name}-${var.environment}-storage-access"

  description = "S3 access for Industrial AI application"

  policy = data.aws_iam_policy_document.storage_access.json
}

resource "aws_iam_role_policy_attachment" "storage_access" {
  role       = aws_iam_role.application.name
  policy_arn = aws_iam_policy.storage_access.arn
}

# ------------------------------------------------------------
# 9. S3 Gateway VPC Endpoint
# ------------------------------------------------------------

resource "aws_vpc_endpoint" "s3" {
  vpc_id = aws_vpc.industrial_ai.id

  service_name = "com.amazonaws.${var.aws_region}.s3"

  vpc_endpoint_type = "Gateway"

  route_table_ids = [
    aws_route_table.private.id
  ]

  tags = {
    Name = "${var.project_name}-${var.environment}-s3-endpoint"
  }
}

# ------------------------------------------------------------
# 10. Outputs
# ------------------------------------------------------------

output "storage_bucket_names" {
  description = "All Industrial AI S3 bucket names"

  value = {
    for name, bucket in aws_s3_bucket.industrial_storage :
    name => bucket.id
  }
}

output "storage_bucket_arns" {
  description = "All Industrial AI S3 bucket ARNs"

  value = {
    for name, bucket in aws_s3_bucket.industrial_storage :
    name => bucket.arn
  }
}

output "storage_vpc_endpoint_id" {
  description = "S3 VPC endpoint ID"
  value       = aws_vpc_endpoint.s3.id
}

output "storage_iam_policy_arn" {
  description = "Application storage IAM policy ARN"
  value       = aws_iam_policy.storage_access.arn
}
