#!/bin/bash
# Infrastructure Validation Script
# Runs all validation checks for Terraform, Kubernetes, Ansible, and CI/CD

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "Fluxion Infrastructure Validation"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

VALIDATION_DIR="validation_logs"
mkdir -p "$VALIDATION_DIR"

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $2"
    else
        echo -e "${RED}✗ FAILED${NC}: $2"
    fi
}

# Terraform Validation
echo "============ Terraform ============"
cd terraform/

echo "1. Terraform Format Check..."
terraform fmt -check -recursive > "../$VALIDATION_DIR/terraform_fmt.log" 2>&1
FMT_STATUS=$?
print_status $FMT_STATUS "Terraform format check"

echo "2. Terraform Init (local backend)..."
terraform init -backend=false > "../$VALIDATION_DIR/terraform_init.log" 2>&1
INIT_STATUS=$?
print_status $INIT_STATUS "Terraform init"

echo "3. Terraform Validate..."
terraform validate > "../$VALIDATION_DIR/terraform_validate.log" 2>&1
VALIDATE_STATUS=$?
print_status $VALIDATE_STATUS "Terraform validate"

cd ..
echo ""

# Ansible Validation
echo "============ Ansible ============"
cd ansible/

if command -v ansible-lint &> /dev/null; then
    echo "1. Ansible Lint..."
    ansible-lint playbooks/main.yml > "../$VALIDATION_DIR/ansible_lint.log" 2>&1
    LINT_STATUS=$?
    print_status $LINT_STATUS "Ansible lint"
else
    echo -e "${YELLOW}⚠ SKIPPED${NC}: ansible-lint not installed"
fi

cd ..
echo ""

# Kubernetes Validation
echo "============ Kubernetes ============"
cd kubernetes/

if command -v yamllint &> /dev/null; then
    echo "1. YAML Lint..."
    yamllint base/ > "../$VALIDATION_DIR/yamllint.log" 2>&1
    YAML_STATUS=$?
    print_status $YAML_STATUS "YAML lint"
else
    echo -e "${YELLOW}⚠ SKIPPED${NC}: yamllint not installed"
fi

if command -v kubectl &> /dev/null; then
    echo "2. Kubernetes Manifest Validation (dry-run)..."
    kubectl apply --dry-run=client -f base/ > "../$VALIDATION_DIR/kubectl_validate.log" 2>&1
    KUBECTL_STATUS=$?
    print_status $KUBECTL_STATUS "Kubectl dry-run"
else
    echo -e "${YELLOW}⚠ SKIPPED${NC}: kubectl not installed"
fi

cd ..
echo ""

# CI/CD Validation
echo "============ CI/CD ============"
cd ci-cd/

if command -v yamllint &> /dev/null; then
    echo "1. CI/CD Workflow YAML Lint..."
    yamllint *.yml github-actions/*.yml > "../$VALIDATION_DIR/cicd_yamllint.log" 2>&1
    CICD_STATUS=$?
    print_status $CICD_STATUS "CI/CD YAML lint"
else
    echo -e "${YELLOW}⚠ SKIPPED${NC}: yamllint not installed"
fi

cd ..
echo ""

# Summary
echo "========================================="
echo "Validation Summary"
echo "========================================="
echo "Results saved to: $VALIDATION_DIR/"
echo ""
echo "Review logs for detailed output:"
ls -1 "$VALIDATION_DIR/"
echo ""
echo -e "${GREEN}Validation complete!${NC}"
