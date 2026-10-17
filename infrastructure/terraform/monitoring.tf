# ============================================================
# monitoring.tf
# Industrial AI - AWS CloudWatch Monitoring
# ============================================================

# ------------------------------------------------------------
# CloudWatch Log Groups
# ------------------------------------------------------------

resource "aws_cloudwatch_log_group" "application" {
  name              = "/industrial-ai/${var.environment}/application"
  retention_in_days = 30

  tags = {
    Component = "application"
  }
}

resource "aws_cloudwatch_log_group" "workers" {
  name              = "/industrial-ai/${var.environment}/workers"
  retention_in_days = 30

  tags = {
    Component = "celery-workers"
  }
}

resource "aws_cloudwatch_log_group" "ml" {
  name              = "/industrial-ai/${var.environment}/ml"
  retention_in_days = 30

  tags = {
    Component = "machine-learning"
  }
}

resource "aws_cloudwatch_log_group" "safety" {
  name              = "/industrial-ai/${var.environment}/safety"
  retention_in_days = 90

  tags = {
    Component = "industrial-safety"
  }
}

resource "aws_cloudwatch_log_group" "rag" {
  name              = "/industrial-ai/${var.environment}/rag"
  retention_in_days = 30

  tags = {
    Component = "rag"
  }
}

# ------------------------------------------------------------
# SNS Alert Notifications
# ------------------------------------------------------------

resource "aws_sns_topic" "industrial_alerts" {
  name = "${var.project_name}-${var.environment}-alerts"

  tags = {
    Component = "monitoring"
  }
}

resource "aws_sns_topic_subscription" "email" {
  count = var.alert_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.industrial_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ------------------------------------------------------------
# RDS CPU Utilization
# ------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "rds_high_cpu" {
  alarm_name          = "${var.project_name}-${var.environment}-rds-high-cpu"
  alarm_description   = "RDS CPU utilization exceeds 80%"
  comparison_operator = "GreaterThanThreshold"

  evaluation_periods = 3
  period             = 300

  metric_name = "CPUUtilization"
  namespace   = "AWS/RDS"
  statistic   = "Average"

  threshold = 80

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.industrial_ai.identifier
  }

  alarm_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  ok_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  treat_missing_data = "missing"

  tags = {
    Component = "database"
  }
}

# ------------------------------------------------------------
# RDS Low Storage
# ------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "rds_low_storage" {
  alarm_name          = "${var.project_name}-${var.environment}-rds-low-storage"
  alarm_description   = "RDS free storage is below 5 GiB"
  comparison_operator = "LessThanThreshold"

  evaluation_periods = 2
  period             = 300

  metric_name = "FreeStorageSpace"
  namespace   = "AWS/RDS"
  statistic   = "Average"

  threshold = 5368709120

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.industrial_ai.identifier
  }

  alarm_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  treat_missing_data = "missing"

  tags = {
    Component = "database"
  }
}

# ------------------------------------------------------------
# RDS High Database Connections
# ------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "rds_high_connections" {
  alarm_name          = "${var.project_name}-${var.environment}-rds-connections"
  alarm_description   = "RDS database connections exceed threshold"
  comparison_operator = "GreaterThanThreshold"

  evaluation_periods = 3
  period             = 300

  metric_name = "DatabaseConnections"
  namespace   = "AWS/RDS"
  statistic   = "Average"

  threshold = var.db_connection_alarm_threshold

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.industrial_ai.identifier
  }

  alarm_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  treat_missing_data = "missing"
}

# ------------------------------------------------------------
# EKS Cluster Failed Nodes
# Requires Container Insights metrics
# ------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "eks_failed_nodes" {
  alarm_name          = "${var.project_name}-${var.environment}-eks-failed-nodes"
  alarm_description   = "EKS cluster has failed nodes"
  comparison_operator = "GreaterThanThreshold"

  evaluation_periods = 2
  period             = 60

  metric_name = "cluster_failed_node_count"
  namespace   = "ContainerInsights"
  statistic   = "Maximum"

  threshold = 0

  dimensions = {
    ClusterName = aws_eks_cluster.industrial_ai.name
  }

  alarm_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  treat_missing_data = "missing"

  tags = {
    Component = "eks"
  }
}

# ------------------------------------------------------------
# Application Error Logs
# Requires application logs to be shipped to CloudWatch
# ------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "application_errors" {
  name           = "${var.project_name}-${var.environment}-errors"
  log_group_name = aws_cloudwatch_log_group.application.name

  pattern = "ERROR"

  metric_transformation {
    name      = "ApplicationErrorCount"
    namespace = "IndustrialAI/Application"
    value     = "1"
  }
}

resource "aws_cloudwatch_metric_alarm" "application_errors" {
  alarm_name          = "${var.project_name}-${var.environment}-application-errors"
  alarm_description   = "More than 10 application errors in 5 minutes"
  comparison_operator = "GreaterThanThreshold"

  evaluation_periods = 1
  period             = 300

  metric_name = "ApplicationErrorCount"
  namespace   = "IndustrialAI/Application"
  statistic   = "Sum"

  threshold = 10

  alarm_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  treat_missing_data = "notBreaching"
}

# ------------------------------------------------------------
# ML Inference Failures
# Custom CloudWatch Metric
# ------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "ml_inference_failures" {
  alarm_name          = "${var.project_name}-${var.environment}-ml-failures"
  alarm_description   = "Machine learning inference failures detected"
  comparison_operator = "GreaterThanThreshold"

  evaluation_periods = 1
  period             = 300

  metric_name = "InferenceFailures"
  namespace   = "IndustrialAI/ML"
  statistic   = "Sum"

  threshold = 5

  alarm_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  treat_missing_data = "notBreaching"
}

# ------------------------------------------------------------
# RAG Retrieval Failures
# Custom CloudWatch Metric
# ------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "rag_failures" {
  alarm_name          = "${var.project_name}-${var.environment}-rag-failures"
  alarm_description   = "RAG retrieval failures detected"
  comparison_operator = "GreaterThanThreshold"

  evaluation_periods = 1
  period             = 300

  metric_name = "RetrievalFailures"
  namespace   = "IndustrialAI/RAG"
  statistic   = "Sum"

  threshold = 5

  alarm_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  treat_missing_data = "notBreaching"
}

# ------------------------------------------------------------
# Safety Event Monitoring
# Custom CloudWatch Metric
# ------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "critical_safety_events" {
  alarm_name          = "${var.project_name}-${var.environment}-critical-safety"
  alarm_description   = "Critical industrial safety event detected"
  comparison_operator = "GreaterThanThreshold"

  evaluation_periods = 1
  period             = 60

  metric_name = "CriticalSafetyEvents"
  namespace   = "IndustrialAI/Safety"
  statistic   = "Sum"

  threshold = 0

  alarm_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  treat_missing_data = "notBreaching"
}

# ------------------------------------------------------------
# Emergency Event Monitoring
# Custom CloudWatch Metric
# ------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "emergency_events" {
  alarm_name          = "${var.project_name}-${var.environment}-emergency"
  alarm_description   = "Industrial emergency event detected"
  comparison_operator = "GreaterThanThreshold"

  evaluation_periods = 1
  period             = 60

  metric_name = "EmergencyEvents"
  namespace   = "IndustrialAI/Emergency"
  statistic   = "Sum"

  threshold = 0

  alarm_actions = [
    aws_sns_topic.industrial_alerts.arn
  ]

  treat_missing_data = "notBreaching"
}

# ------------------------------------------------------------
# CloudWatch Dashboard
# ------------------------------------------------------------

resource "aws_cloudwatch_dashboard" "industrial_ai" {
  dashboard_name = "${var.project_name}-${var.environment}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 2

        properties = {
          markdown = "# Industrial AI - Infrastructure Monitoring\nEKS, PostgreSQL, ML, RAG, Safety and Emergency"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 2
        width  = 12
        height = 6

        properties = {
          title  = "RDS CPU Utilization"
          region = var.aws_region

          metrics = [
            [
              "AWS/RDS",
              "CPUUtilization",
              "DBInstanceIdentifier",
              aws_db_instance.industrial_ai.identifier
            ]
          ]

          period = 300
          stat   = "Average"
          yAxis = {
            left = {
              min = 0
              max = 100
            }
          }
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 2
        width  = 12
        height = 6

        properties = {
          title  = "RDS Database Connections"
          region = var.aws_region

          metrics = [
            [
              "AWS/RDS",
              "DatabaseConnections",
              "DBInstanceIdentifier",
              aws_db_instance.industrial_ai.identifier
            ]
          ]

          period = 300
          stat   = "Average"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 8
        width  = 12
        height = 6

        properties = {
          title  = "ML Inference Failures"
          region = var.aws_region

          metrics = [
            [
              "IndustrialAI/ML",
              "InferenceFailures"
            ]
          ]

          period = 300
          stat   = "Sum"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 8
        width  = 12
        height = 6

        properties = {
          title  = "RAG Retrieval Failures"
          region = var.aws_region

          metrics = [
            [
              "IndustrialAI/RAG",
              "RetrievalFailures"
            ]
          ]

          period = 300
          stat   = "Sum"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 14
        width  = 12
        height = 6

        properties = {
          title  = "Critical Safety Events"
          region = var.aws_region

          metrics = [
            [
              "IndustrialAI/Safety",
              "CriticalSafetyEvents"
            ]
          ]

          period = 60
          stat   = "Sum"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 14
        width  = 12
        height = 6

        properties = {
          title  = "Emergency Events"
          region = var.aws_region

          metrics = [
            [
              "IndustrialAI/Emergency",
              "EmergencyEvents"
            ]
          ]

          period = 60
          stat   = "Sum"
        }
      }
    ]
  })
}

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

output "industrial_alerts_topic_arn" {
  description = "SNS topic ARN for Industrial AI alerts"
  value       = aws_sns_topic.industrial_alerts.arn
}

output "industrial_dashboard_name" {
  description = "CloudWatch dashboard name"
  value       = aws_cloudwatch_dashboard.industrial_ai.dashboard_name
}