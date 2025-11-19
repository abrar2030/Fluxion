# Fluxion Infrastructure - Enhanced for Financial Security Standards
# This configuration provisions secure cloud infrastructure for the Fluxion platform

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # Secure remote state configuration
  backend "s3" {
    bucket         = var.terraform_state_bucket
    key            = "fluxion/terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    kms_key_id     = var.terraform_state_kms_key
    dynamodb_table = var.terraform_state_lock_table

    # Enable versioning and MFA delete protection
    versioning = true
  }
}

# Configure AWS Provider with security best practices
provider "aws" {
  region = var.aws_region

  # Assume role for cross-account access if needed
  dynamic "assume_role" {
    for_each = var.assume_role_arn != null ? [1] : []
    content {
      role_arn     = var.assume_role_arn
      session_name = "terraform-fluxion-${random_id.session.hex}"
      external_id  = var.assume_role_external_id
    }
  }

  default_tags {
    tags = local.common_tags
  }
}

# Generate random session ID for assume role
resource "random_id" "session" {
  byte_length = 8
}

# Local values for consistent tagging and naming
locals {
  common_tags = merge(var.default_tags, {
    Project             = "Fluxion"
    Environment         = var.environment
    ManagedBy          = "Terraform"
    SecurityLevel      = "Financial"
    ComplianceLevel    = "Restricted"
    DataClassification = "Confidential"
    BackupRequired     = "true"
    MonitoringRequired = "true"
    CreatedDate        = formatdate("YYYY-MM-DD", timestamp())
  })

  name_prefix = "${var.app_name}-${var.environment}"
}

# Data sources for existing resources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

# KMS key for encryption
resource "aws_kms_key" "fluxion_key" {
  description             = "KMS key for Fluxion ${var.environment} environment encryption"
  deletion_window_in_days = var.kms_deletion_window
  enable_key_rotation     = true
  multi_region           = var.enable_multi_region

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow CloudTrail to encrypt logs"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      },
      {
        Sid    = "Allow CloudWatch Logs"
        Effect = "Allow"
        Principal = {
          Service = "logs.${data.aws_region.current.name}.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-kms-key"
    Type = "encryption"
  })
}

resource "aws_kms_alias" "fluxion_key" {
  name          = "alias/${local.name_prefix}-key"
  target_key_id = aws_kms_key.fluxion_key.key_id
}

# Network module
module "network" {
  source = "./modules/network"

  environment         = var.environment
  vpc_cidr            = var.vpc_cidr
  availability_zones  = var.availability_zones
  public_subnet_cidrs = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  enable_nat_gateway = var.enable_nat_gateway
  enable_vpn_gateway = var.enable_vpn_gateway
  enable_flow_logs   = var.enable_flow_logs
  kms_key_id         = aws_kms_key.fluxion_key.arn

  common_tags = local.common_tags
}

# Security module
module "security" {
  source = "./modules/security"

  environment  = var.environment
  vpc_id       = module.network.vpc_id
  app_name     = var.app_name
  vpc_cidr     = module.network.vpc_cidr
  kms_key_id   = aws_kms_key.fluxion_key.arn

  # Security configuration
  enable_guardduty     = var.enable_guardduty
  enable_config        = var.enable_config
  enable_cloudtrail    = var.enable_cloudtrail
  enable_security_hub  = var.enable_security_hub
  enable_inspector     = var.enable_inspector

  # Compliance settings
  cloudtrail_s3_bucket = var.cloudtrail_s3_bucket
  config_s3_bucket     = var.config_s3_bucket

  common_tags = local.common_tags
}

# Storage module
module "storage" {
  source = "./modules/storage"

  environment = var.environment
  app_name    = var.app_name
  kms_key_id   = aws_kms_key.fluxion_key.arn

  # Storage configuration
  enable_versioning     = var.enable_s3_versioning
  enable_mfa_delete     = var.enable_s3_mfa_delete
  lifecycle_rules       = var.s3_lifecycle_rules
  backup_retention_days = var.backup_retention_days

  common_tags = local.common_tags
}

# Database module
module "database" {
  source = "./modules/database"

  environment       = var.environment
  vpc_id            = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  db_instance_class = var.db_instance_class
  db_name           = var.db_name
  db_username       = var.db_username
  db_password       = var.db_password
  security_group_ids = [module.security.db_security_group_id]
  kms_key_id      = aws_kms_key.fluxion_key.arn

  # Database configuration
  allocated_storage       = var.db_allocated_storage
  max_allocated_storage   = var.db_max_allocated_storage
  backup_retention_period = var.db_backup_retention_period
  backup_window          = var.db_backup_window
  maintenance_window     = var.db_maintenance_window
  multi_az               = var.db_multi_az

  # Security settings
  deletion_protection     = var.db_deletion_protection
  skip_final_snapshot    = var.db_skip_final_snapshot
  copy_tags_to_snapshot  = true
  performance_insights   = var.db_performance_insights
  monitoring_interval    = var.db_monitoring_interval

  common_tags = local.common_tags
}

# Compute module
module "compute" {
  source = "./modules/compute"

  environment       = var.environment
  vpc_id            = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  public_subnet_ids = module.network.public_subnet_ids
  instance_type     = var.instance_type
  key_name          = var.key_name
  app_name          = var.app_name
  security_group_ids = [module.security.app_security_group_id]
  kms_key_id       = aws_kms_key.fluxion_key.arn

  # Compute configuration
  min_size             = var.asg_min_size
  max_size             = var.asg_max_size
  desired_capacity     = var.asg_desired_capacity
  enable_monitoring    = var.enable_detailed_monitoring
  enable_ebs_encryption = true

  # Load balancer configuration
  enable_alb           = var.enable_alb
  alb_certificate_arn  = var.alb_certificate_arn
  health_check_path    = var.health_check_path

  common_tags = local.common_tags
}
