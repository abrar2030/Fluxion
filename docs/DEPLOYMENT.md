# Deployment Guide

## Prerequisites

### System Requirements

- CPU: 8+ cores
- RAM: 32GB+
- Storage: 500GB+ SSD
- Network: 1Gbps+
- GPU: NVIDIA A100 or similar (for AI models)

### Software Requirements

- Docker 24.0+
- Kubernetes 1.28+
- Helm 3.12+
- Node.js 18+
- Python 3.11+
- Foundry 0.2+

## Infrastructure Setup

### 1. Cloud Provider Configuration

```bash
# AWS Example
export AWS_REGION=us-east-1
export CLUSTER_NAME=fluxion-prod

# Create EKS cluster
eksctl create cluster \\
  --name $CLUSTER_NAME \\
  --region $AWS_REGION \\
  --nodes 5 \\
  --node-type m5.2xlarge
```

### 2. Database Setup

```bash
# TimescaleDB
helm repo add timescale https://charts.timescale.com
helm install timescaledb timescale/timescaledb-single \\
  --values infra/timescaledb-values.yaml

# Redis
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis \\
  --values infra/redis-values.yaml
```

### 3. Blockchain Node Setup

```bash
# Deploy Polygon zkEVM node
helm install zkevm polygon/zkevm-node \\
  --values infra/zkevm-values.yaml

# Deploy Chainlink node
helm install chainlink chainlink/chainlink-node \\
  --values infra/chainlink-values.yaml
```

## Smart Contract Deployment

### 1. Contract Compilation

```bash
# Compile contracts
cd blockchain
forge build

# Run tests
forge test

# Generate deployment artifacts
forge create --rpc-url $RPC_URL \\
  --private-key $DEPLOYER_KEY \\
  src/SyntheticAssetFactory.sol:SyntheticAssetFactory
```

### 2. Contract Verification

```bash
# Verify on Polygonscan
forge verify-contract \\
  --chain-id 1101 \\
  --compiler-version v0.8.19 \\
  $CONTRACT_ADDRESS \\
  src/SyntheticAssetFactory.sol:SyntheticAssetFactory
```

## Backend Deployment

### 1. API Service

```yaml
# kubernetes/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluxion-api
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: api
          image: fluxion/api:latest
          resources:
            requests:
              memory: "4Gi"
              cpu: "2"
            limits:
              memory: "8Gi"
              cpu: "4"
```

### 2. AI Service

```yaml
# kubernetes/ai-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluxion-ai
spec:
  template:
    spec:
      containers:
        - name: ai
          image: fluxion/ai:latest
          resources:
            limits:
              nvidia.com/gpu: 1
```

### 3. Worker Service

```yaml
# kubernetes/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluxion-worker
spec:
  replicas: 5
  template:
    spec:
      containers:
        - name: worker
          image: fluxion/worker:latest
```

## Frontend Deployment

### 1. Build Process

```bash
# Build frontend
cd frontend
npm install
npm run build

# Deploy to CDN
aws s3 sync build/ s3://fluxion-frontend
```

### 2. CDN Configuration

```bash
# Configure CloudFront
aws cloudfront create-distribution \\
  --origin-domain-name fluxion-frontend.s3.amazonaws.com \\
  --default-root-object index.html
```

## Monitoring Setup

### 1. Prometheus & Grafana

```bash
# Install monitoring stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \\
  --values infra/monitoring-values.yaml
```

### 2. Logging

```bash
# Install ELK stack
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch
helm install kibana elastic/kibana
helm install filebeat elastic/filebeat
```

## Security Configuration

### 1. Network Policies

```yaml
# kubernetes/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-policy
spec:
  podSelector:
    matchLabels:
      app: fluxion-api
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend
```

### 2. Secret Management

```bash
# Create secrets
kubectl create secret generic api-keys \\
  --from-literal=polygon-key=$POLYGON_KEY \\
  --from-literal=chainlink-key=$CHAINLINK_KEY

# Install vault
helm install vault hashicorp/vault \\
  --values infra/vault-values.yaml
```

## Backup and Recovery

### 1. Database Backups

```bash
# Configure automated backups
kubectl apply -f kubernetes/backup-cronjob.yaml

# Test restore procedure
kubectl exec -it timescaledb-0 -- \\
  pg_restore -d fluxion /backups/latest.dump
```

### 2. Blockchain Node Snapshots

```bash
# Configure node snapshots
kubectl apply -f kubernetes/snapshot-cronjob.yaml
```

## Performance Tuning

### 1. Resource Optimization

```bash
# Configure HPA
kubectl autoscale deployment fluxion-api \\
  --cpu-percent=80 \\
  --min=3 \\
  --max=10
```

### 2. Cache Configuration

```yaml
# kubernetes/redis-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
data:
  redis.conf: |
    maxmemory 8gb
    maxmemory-policy allkeys-lru
```

## Troubleshooting

### Common Issues

1. Node Connection Issues

```bash
# Check node status
kubectl exec -it zkevm-0 -- curl localhost:8545

# View node logs
kubectl logs -f zkevm-0
```

2. Database Performance

```bash
# Monitor queries
kubectl exec -it timescaledb-0 -- \\
  psql -U postgres -c "SELECT * FROM pg_stat_activity;"
```

3. API Errors

```bash
# Check API logs
kubectl logs -f -l app=fluxion-api

# Test API endpoint
curl -v https://api.fluxion.exchange/health
```

## Maintenance

### 1. Regular Updates

```bash
# Update dependencies
helm repo update
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack

# Update application
kubectl set image deployment/fluxion-api \\
  api=fluxion/api:latest
```

### 2. Health Checks

```bash
# Check system health
kubectl get pods
kubectl top nodes
kubectl describe nodes

# Check service health
for svc in $(kubectl get svc -o name); do
  kubectl describe $svc
done
```
