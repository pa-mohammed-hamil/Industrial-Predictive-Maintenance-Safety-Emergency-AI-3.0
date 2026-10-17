# ============================================================
# networking.tf
# Industrial AI - AWS VPC Networking
# ============================================================

# ------------------------------------------------------------
# VPC
# ------------------------------------------------------------

resource "aws_vpc" "industrial_ai" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_name}-${var.environment}-vpc"
  }
}

# ------------------------------------------------------------
# Internet Gateway
# ------------------------------------------------------------

resource "aws_internet_gateway" "industrial_ai" {
  vpc_id = aws_vpc.industrial_ai.id

  tags = {
    Name = "${var.project_name}-${var.environment}-igw"
  }
}

# ------------------------------------------------------------
# Availability Zones
# ------------------------------------------------------------

locals {
  networking_azs = slice(
    data.aws_availability_zones.available.names,
    0,
    2
  )
}

# ------------------------------------------------------------
# Public Subnets
# ------------------------------------------------------------

resource "aws_subnet" "public" {
  count = 2

  vpc_id = aws_vpc.industrial_ai.id

  cidr_block = cidrsubnet(
    var.vpc_cidr,
    8,
    count.index
  )

  availability_zone = local.networking_azs[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-${var.environment}-public-${count.index + 1}"

    "kubernetes.io/role/elb" = "1"

    "kubernetes.io/cluster/${var.project_name}-${var.environment}-eks" = "shared"
  }
}

# ------------------------------------------------------------
# Private Subnets
# ------------------------------------------------------------

resource "aws_subnet" "private" {
  count = 2

  vpc_id = aws_vpc.industrial_ai.id

  cidr_block = cidrsubnet(
    var.vpc_cidr,
    8,
    count.index + 10
  )

  availability_zone = local.networking_azs[count.index]

  map_public_ip_on_launch = false

  tags = {
    Name = "${var.project_name}-${var.environment}-private-${count.index + 1}"

    "kubernetes.io/role/internal-elb" = "1"

    "kubernetes.io/cluster/${var.project_name}-${var.environment}-eks" = "shared"
  }
}

# ------------------------------------------------------------
# Elastic IP for NAT Gateway
# ------------------------------------------------------------

resource "aws_eip" "nat" {
  count  = var.enable_nat_gateway ? 1 : 0
  domain = "vpc"

  tags = {
    Name = "${var.project_name}-${var.environment}-nat-eip"
  }

  depends_on = [
    aws_internet_gateway.industrial_ai
  ]
}

# ------------------------------------------------------------
# NAT Gateway
# ------------------------------------------------------------

resource "aws_nat_gateway" "industrial_ai" {
  count = var.enable_nat_gateway ? 1 : 0

  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name = "${var.project_name}-${var.environment}-nat"
  }

  depends_on = [
    aws_internet_gateway.industrial_ai
  ]
}

# ------------------------------------------------------------
# Public Route Table
# ------------------------------------------------------------

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.industrial_ai.id

  tags = {
    Name = "${var.project_name}-${var.environment}-public-rt"
  }
}

resource "aws_route" "public_internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.industrial_ai.id
}

resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ------------------------------------------------------------
# Private Route Table
# ------------------------------------------------------------

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.industrial_ai.id

  tags = {
    Name = "${var.project_name}-${var.environment}-private-rt"
  }
}

resource "aws_route" "private_nat" {
  count = var.enable_nat_gateway ? 1 : 0

  route_table_id         = aws_route_table.private.id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.industrial_ai[0].id
}

resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# ------------------------------------------------------------
# VPC Flow Logs
# ------------------------------------------------------------

resource "aws_cloudwatch_log_group" "vpc_flow_logs" {
  name              = "/aws/vpc/${var.project_name}-${var.environment}"
  retention_in_days = 30

  tags = {
    Component = "networking"
  }
}

resource "aws_iam_role" "vpc_flow_logs" {
  name = "${var.project_name}-${var.environment}-vpc-flow-logs-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "vpc-flow-logs.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "vpc_flow_logs" {
  name = "${var.project_name}-${var.environment}-vpc-flow-logs-policy"
  role = aws_iam_role.vpc_flow_logs.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]

        Resource = "${aws_cloudwatch_log_group.vpc_flow_logs.arn}:*"
      }
    ]
  })
}

resource "aws_flow_log" "industrial_ai" {
  iam_role_arn    = aws_iam_role.vpc_flow_logs.arn
  log_destination = aws_cloudwatch_log_group.vpc_flow_logs.arn

  traffic_type = "ALL"
  vpc_id       = aws_vpc.industrial_ai.id

  tags = {
    Name = "${var.project_name}-${var.environment}-flow-logs"
  }

  depends_on = [
    aws_iam_role_policy.vpc_flow_logs
  ]
}

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

output "network_vpc_id" {
  description = "Industrial AI VPC ID"
  value       = aws_vpc.industrial_ai.id
}

output "network_public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "network_private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "network_nat_gateway_id" {
  description = "NAT Gateway ID"
  value = var.enable_nat_gateway ? (
    aws_nat_gateway.industrial_ai[0].id
  ) : null
}