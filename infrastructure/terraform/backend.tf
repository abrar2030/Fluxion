# Backend Configuration
# This file configures Terraform state backend
# For local development, comment out the S3 backend and use local state
# For production, configure S3 backend with proper values in terraform.tfvars

# Local backend (default for development)
terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}

# S3 backend (uncomment and configure for production)
# terraform {
#   backend "s3" {
#     # Configure these values in your backend configuration file
#     # or via -backend-config flag during terraform init
#     # 
#     # bucket         = "your-terraform-state-bucket"
#     # key            = "fluxion/terraform.tfstate"
#     # region         = "us-east-1"
#     # encrypt        = true
#     # kms_key_id     = "arn:aws:kms:region:account:key/key-id"
#     # dynamodb_table = "terraform-state-lock"
#   }
#}
