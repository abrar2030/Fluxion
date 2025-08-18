# Fluxion Backend Deployment Guide

This guide provides comprehensive instructions for deploying the Fluxion DeFi supply chain platform backend in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Database Configuration](#database-configuration)
4. [Application Configuration](#application-configuration)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Security Configuration](#security-configuration)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04+ or CentOS 8+
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Memory**: Minimum 8GB RAM, Recommended 16GB+ RAM
- **Storage**: Minimum 100GB SSD, Recommended 500GB+ SSD
- **Network**: Stable internet connection with sufficient bandwidth

### Software Dependencies

- **Python**: 3.11+
- **PostgreSQL**: 14+
- **Redis**: 6.2+
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (for multi-container setup)
- **Nginx**: 1.20+ (for reverse proxy)

### External Services

- **SMTP Server**: For email notifications
- **SSL Certificate**: For HTTPS (Let's Encrypt recommended)
- **Blockchain RPC Endpoints**: Ethereum, Polygon, etc.
- **External APIs**: KYC/AML providers, price feeds

## Environment Setup

### 1. Create Application User

```bash
# Create dedicated user for the application
sudo useradd -m -s /bin/bash fluxion
sudo usermod -aG sudo fluxion

# Switch to application user
sudo su - fluxion
```

### 2. Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    postgresql-client \
    redis-tools \
    nginx \
    supervisor \
    git \
    curl \
    wget \
    unzip

# Install Docker (optional, for containerized deployment)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker fluxion
```

### 3. Setup Application Directory

```bash
# Create application directory
sudo mkdir -p /opt/fluxion
sudo chown fluxion:fluxion /opt/fluxion

# Navigate to application directory
cd /opt/fluxion

# Extract application code
unzip fluxion_backend_production.zip
cd fluxion_backend_production
```

### 4. Create Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install application dependencies
pip install -r requirements.txt
```

## Database Configuration

### 1. PostgreSQL Setup

```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE fluxion_db;
CREATE USER fluxion_user WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE fluxion_db TO fluxion_user;
ALTER USER fluxion_user CREATEDB;
\q
EOF
```

### 2. Database Migration

```bash
# Set database URL
export DATABASE_URL="postgresql+asyncpg://fluxion_user:secure_password_here@localhost:5432/fluxion_db"

# Run database migrations
alembic upgrade head
```

### 3. Redis Setup

```bash
# Install Redis
sudo apt install -y redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf

# Update the following settings:
# bind 127.0.0.1
# requirepass your_redis_password_here
# maxmemory 2gb
# maxmemory-policy allkeys-lru

# Restart Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server
```

## Application Configuration

### 1. Environment Variables

Create a production environment file:

```bash
# Create .env file
cat > .env << EOF
# Application Settings
APP_NAME="Fluxion Backend"
VERSION="1.0.0"
ENVIRONMENT="production"
DEBUG=false
LOG_LEVEL="INFO"

# Security Settings
JWT_SECRET_KEY="your-super-secure-jwt-secret-key-here"
JWT_ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=30

# Database Configuration
DATABASE_URL="postgresql+asyncpg://fluxion_user:secure_password_here@localhost:5432/fluxion_db"
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL="redis://:your_redis_password_here@localhost:6379/0"

# CORS Settings
ALLOWED_ORIGINS=["https://yourdomain.com", "https://app.yourdomain.com"]
ALLOWED_HOSTS=["yourdomain.com", "app.yourdomain.com", "api.yourdomain.com"]

# Email Configuration
SMTP_HOST="smtp.yourmailprovider.com"
SMTP_PORT=587
SMTP_USERNAME="your-email@yourdomain.com"
SMTP_PASSWORD="your-email-password"
SMTP_USE_TLS=true

# Blockchain Configuration
ETHEREUM_RPC_URL="https://mainnet.infura.io/v3/your-project-id"
POLYGON_RPC_URL="https://polygon-mainnet.infura.io/v3/your-project-id"

# External API Keys
KYC_PROVIDER_API_KEY="your-kyc-provider-api-key"
AML_PROVIDER_API_KEY="your-aml-provider-api-key"
PRICE_FEED_API_KEY="your-price-feed-api-key"

# Monitoring
SENTRY_DSN="your-sentry-dsn-here"
EOF

# Secure the environment file
chmod 600 .env
```

### 2. Application Settings

Update `config/settings.py` for production:

```python
# Production-specific settings
ALLOWED_HOSTS = ["yourdomain.com", "api.yourdomain.com"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://app.yourdomain.com"
]

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
```

## Docker Deployment

### 1. Build Docker Image

```bash
# Build the application image
docker build -t fluxion-backend:latest .

# Verify the image
docker images | grep fluxion-backend
```

### 2. Docker Compose Deployment

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### 3. Docker Production Configuration

Update `docker-compose.yml` for production:

```yaml
version: '3.8'

services:
  app:
    image: fluxion-backend:latest
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgresql+asyncpg://fluxion_user:${DB_PASSWORD}@db:5432/fluxion_db
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
    depends_on:
      - db
      - redis
    networks:
      - fluxion-network

  db:
    image: postgres:14
    restart: unless-stopped
    environment:
      - POSTGRES_DB=fluxion_db
      - POSTGRES_USER=fluxion_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - fluxion-network

  redis:
    image: redis:6.2-alpine
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - fluxion-network

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    networks:
      - fluxion-network

volumes:
  postgres_data:
  redis_data:

networks:
  fluxion-network:
    driver: bridge
```

## Kubernetes Deployment

### 1. Kubernetes Manifests

Create Kubernetes deployment files:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fluxion-backend
  labels:
    app: fluxion-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fluxion-backend
  template:
    metadata:
      labels:
        app: fluxion-backend
    spec:
      containers:
      - name: fluxion-backend
        image: fluxion-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: fluxion-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: fluxion-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 2. Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods
kubectl get services
kubectl get ingress
```

## Security Configuration

### 1. SSL/TLS Setup

```bash
# Install Certbot for Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d yourdomain.com -d api.yourdomain.com

# Set up auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow necessary ports
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Check firewall status
sudo ufw status
```

### 3. Nginx Configuration

```nginx
# /etc/nginx/sites-available/fluxion
server {
    listen 80;
    server_name yourdomain.com api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring and Logging

### 1. Application Monitoring

```bash
# Install monitoring tools
pip install prometheus-client sentry-sdk

# Configure Prometheus metrics endpoint
# Add to main.py:
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 2. Log Management

```bash
# Configure log rotation
sudo nano /etc/logrotate.d/fluxion

# Add:
/opt/fluxion/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 fluxion fluxion
    postrotate
        systemctl reload fluxion-backend
    endscript
}
```

### 3. Health Checks

```bash
# Create health check script
cat > /opt/fluxion/scripts/health_check.sh << 'EOF'
#!/bin/bash
HEALTH_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "Service is healthy"
    exit 0
else
    echo "Service is unhealthy (HTTP $RESPONSE)"
    exit 1
fi
EOF

chmod +x /opt/fluxion/scripts/health_check.sh
```

## Backup and Recovery

### 1. Database Backup

```bash
# Create backup script
cat > /opt/fluxion/scripts/backup_db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/fluxion/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="fluxion_db_backup_$DATE.sql"

mkdir -p $BACKUP_DIR

pg_dump -h localhost -U fluxion_user -d fluxion_db > $BACKUP_DIR/$BACKUP_FILE

# Compress backup
gzip $BACKUP_DIR/$BACKUP_FILE

# Remove backups older than 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE.gz"
EOF

chmod +x /opt/fluxion/scripts/backup_db.sh

# Schedule daily backups
crontab -e
# Add: 0 2 * * * /opt/fluxion/scripts/backup_db.sh
```

### 2. Application Backup

```bash
# Create application backup script
cat > /opt/fluxion/scripts/backup_app.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/fluxion/backups"
DATE=$(date +%Y%m%d_%H%M%S)
APP_BACKUP="fluxion_app_backup_$DATE.tar.gz"

mkdir -p $BACKUP_DIR

tar -czf $BACKUP_DIR/$APP_BACKUP \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs' \
    /opt/fluxion/fluxion_backend_production

echo "Application backup completed: $APP_BACKUP"
EOF

chmod +x /opt/fluxion/scripts/backup_app.sh
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check database connectivity
psql -h localhost -U fluxion_user -d fluxion_db -c "SELECT version();"

# Check database logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log
```

#### 2. Redis Connection Issues

```bash
# Check Redis status
sudo systemctl status redis-server

# Test Redis connectivity
redis-cli -a your_redis_password ping

# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log
```

#### 3. Application Startup Issues

```bash
# Check application logs
tail -f /opt/fluxion/logs/app.log

# Check system resources
htop
df -h
free -m

# Check port availability
netstat -tlnp | grep :8000
```

#### 4. SSL Certificate Issues

```bash
# Check certificate status
sudo certbot certificates

# Test SSL configuration
openssl s_client -connect yourdomain.com:443

# Renew certificate manually
sudo certbot renew --dry-run
```

### Performance Optimization

#### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY idx_transactions_user_id ON transactions(user_id);
CREATE INDEX CONCURRENTLY idx_transactions_created_at ON transactions(created_at);

-- Analyze table statistics
ANALYZE users;
ANALYZE transactions;
ANALYZE portfolios;
```

#### 2. Application Optimization

```bash
# Increase worker processes
export WORKERS=4
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers $WORKERS

# Configure connection pooling
export DATABASE_POOL_SIZE=20
export DATABASE_MAX_OVERFLOW=30
```

### Maintenance Tasks

#### 1. Regular Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
pip install --upgrade -r requirements.txt

# Update Docker images
docker-compose pull
docker-compose up -d
```

#### 2. Log Cleanup

```bash
# Clean old logs
find /opt/fluxion/logs -name "*.log" -mtime +30 -delete

# Clean Docker logs
docker system prune -f
```

#### 3. Database Maintenance

```sql
-- Vacuum and analyze database
VACUUM ANALYZE;

-- Check database size
SELECT pg_size_pretty(pg_database_size('fluxion_db'));

-- Check table sizes
SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Support and Documentation

For additional support and documentation:

- **Application Logs**: `/opt/fluxion/logs/`
- **Configuration Files**: `/opt/fluxion/fluxion_backend_production/config/`
- **Database Schema**: Check `migrations/` directory
- **API Documentation**: `https://yourdomain.com/docs` (if enabled)

## Security Considerations

1. **Regular Security Updates**: Keep all system packages and dependencies updated
2. **Access Control**: Use strong passwords and consider implementing 2FA
3. **Network Security**: Use VPN for administrative access
4. **Data Encryption**: Ensure all sensitive data is encrypted at rest and in transit
5. **Audit Logging**: Monitor and log all administrative actions
6. **Backup Security**: Encrypt and secure all backup files
7. **Compliance**: Ensure deployment meets relevant regulatory requirements

This deployment guide provides a comprehensive foundation for deploying the Fluxion backend in production environments. Adjust configurations based on your specific infrastructure requirements and security policies.

