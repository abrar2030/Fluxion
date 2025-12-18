# Fluxion Infrastructure

Complete, audited, and hardened infrastructure configuration for the Fluxion platform.

## 🔒 Security & Compliance

This infrastructure has been audited and fixed to meet:

- DevOps best practices
- FinOps security standards
- SecOps hardening requirements
- Financial industry compliance

## 🚀 Quick Start

### Prerequisites

```bash
# Install required tools
terraform --version  # >= 1.6.6
ansible --version    # >= 2.9
kubectl version      # >= 1.25
python3 --version    # >= 3.10
```

### Local Development

1. **Copy example configurations:**

```bash
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
cp ansible/inventory.example ansible/inventory
cp ansible/group_vars/all.example ansible/group_vars/all.yml
cp kubernetes/base/app-secrets.example.yaml kubernetes/base/app-secrets.yaml
```

2. **Update with your values:**

```bash
# Edit Terraform variables
vim terraform/terraform.tfvars

# Set database password via environment variable (recommended)
export TF_VAR_db_password="your-secure-password"

# Encrypt Ansible secrets
ansible-vault create ansible/group_vars/all.yml
```

3. **Start local development environment:**

```bash
docker-compose up -d
```

## 🧪 Validation

Run the comprehensive validation script:

```bash
./validate_infrastructure.sh
```

Or validate components individually:

### Terraform

```bash
cd terraform/
terraform fmt -recursive
terraform init -backend=false
terraform validate
```

### Kubernetes

```bash
cd kubernetes/
yamllint base/
kubectl apply --dry-run=client -f base/
```

### Ansible

```bash
cd ansible/
ansible-galaxy install -r requirements.yml
ansible-lint playbooks/main.yml
```

## 📁 Directory Structure

```
infrastructure/
├── README.md                    # This file
├── CHANGES.md                   # Detailed change log
├── .gitignore                   # Ignore sensitive files
├── validate_infrastructure.sh   # Validation script
├── terraform/                   # Infrastructure as Code
│   ├── main.tf
│   ├── backend.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── terraform.tfvars.example  # COPY and update this
│   ├── .terraform-version
│   └── modules/
├── kubernetes/                  # Container orchestration
│   ├── base/                    # Base manifests
│   │   ├── *-deployment.yaml
│   │   ├── *-service.yaml
│   │   ├── app-secrets.example.yaml  # COPY and update this
│   │   └── pod-security-standards.yaml
│   └── environments/            # Environment overlays
├── ansible/                     # Configuration management
│   ├── playbooks/
│   ├── roles/
│   ├── inventory.example        # COPY and update this
│   ├── group_vars/all.example   # COPY and update this
│   └── requirements.yml
├── ci-cd/                       # CI/CD pipelines
│   ├── ci-cd.yml
│   ├── github-actions/
│   └── scripts/
├── docker-compose.yml           # Local development
└── validation_logs/             # Validation outputs
```

## 🔐 Secrets Management

### DO NOT commit these files:

- `terraform/terraform.tfvars`
- `terraform/*.tfstate*`
- `ansible/inventory`
- `ansible/group_vars/all.yml`
- `kubernetes/base/app-secrets.yaml`
- `.env` files

### Use `.example` files as templates:

- Copy `.example` files to remove the `.example` suffix
- Update with your actual values
- The `.gitignore` will prevent accidental commits

### Recommended secret management:

- **Terraform**: Use environment variables (`TF_VAR_*`) or Terraform Cloud
- **Kubernetes**: Use External Secrets Operator + HashiCorp Vault
- **Ansible**: Use `ansible-vault` for encrypted variables
- **CI/CD**: Use GitHub Secrets / GitLab CI Variables

## 🏗️ Deployment

### Terraform Infrastructure

```bash
cd terraform/

# Initialize
terraform init

# Plan
terraform plan -out=plan.out

# Apply
terraform apply plan.out
```

### Kubernetes Applications

```bash
cd kubernetes/

# Validate
kubectl apply --dry-run=client -f base/

# Deploy
kubectl apply -f base/

# Check status
kubectl get pods -A
```

### Ansible Configuration

```bash
cd ansible/

# Install collections
ansible-galaxy install -r requirements.yml

# Dry run
ansible-playbook -i inventory playbooks/main.yml --check

# Execute
ansible-playbook -i inventory playbooks/main.yml
```

## 📊 Monitoring & Logs

Access local monitoring stack:

- **Grafana**: http://localhost:3001
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601

## 🐛 Troubleshooting

### Terraform validation fails

```bash
# Check formatting
terraform fmt -check -recursive

# Reinitialize
rm -rf .terraform/
terraform init -backend=false
```

### Kubernetes manifests invalid

```bash
# Check specific file
kubectl apply --dry-run=client -f kubernetes/base/backend-deployment.yaml

# Validate all
find kubernetes/base/ -name "*.yaml" -exec kubectl apply --dry-run=client -f {} \;
```

### Ansible lint errors

```bash
# Install missing collections
ansible-galaxy collection install community.general

# Run with verbose output
ansible-lint -v playbooks/main.yml
```
