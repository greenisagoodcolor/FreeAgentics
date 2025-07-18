# Deployment Guide - FreeAgentics v1.0.0-alpha

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Installation Methods](#installation-methods)
5. [Configuration](#configuration)
6. [Database Setup](#database-setup)
7. [Security Configuration](#security-configuration)
8. [Deployment Options](#deployment-options)
9. [Monitoring Setup](#monitoring-setup)
10. [Troubleshooting](#troubleshooting)
11. [Quick Start Commands](#quick-start-commands)

## Overview

FreeAgentics v1.0.0-alpha is a multi-agent AI platform implementing Active Inference. This guide covers deployment options from development to production environments.

### Deployment Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Nginx     │────▶│   FastAPI   │────▶│ PostgreSQL  │
│  (Reverse   │     │  (Backend)  │     │ (Database)  │
│   Proxy)    │     └─────────────┘     └─────────────┘
└─────────────┘              │
       │                     │           ┌─────────────┐
       │                     └──────────▶│    Redis    │
       ▼                                 │   (Cache)   │
┌─────────────┐                         └─────────────┘
│   Next.js   │
│ (Frontend)  │
└─────────────┘
```

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / macOS 11+ / Windows WSL2
- **CPU**: 4 cores minimum (8 recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB free space
- **Network**: Stable internet connection

### Software Requirements
```bash
# Core Requirements
- Python 3.11+ (3.12 recommended)
- Node.js 18+ with npm 9+
- PostgreSQL 13+
- Redis 6+
- Git 2.25+

# Optional (for Docker deployment)
- Docker 20.10+
- Docker Compose 2.0+

# Optional (for Kubernetes)
- kubectl 1.25+
- Helm 3.10+
```

### Verify Prerequisites
```bash
# Check versions
python --version          # Should show 3.11+
node --version           # Should show v18+
psql --version          # Should show 13+
redis-server --version  # Should show 6+
docker --version        # Should show 20.10+
```

## Environment Setup

### 1. Clone Repository
```bash
# Clone the repository
git clone https://github.com/your-org/freeagentics.git
cd freeagentics

# Checkout specific version
git checkout v1.0.0-alpha
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies
```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd web
npm install
cd ..
```

## Installation Methods

### Method 1: Local Development
```bash
# Quick setup
make install
make dev

# Manual setup
# Terminal 1: Backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd web
npm run dev

# Terminal 3: Redis
redis-server

# Terminal 4: PostgreSQL (if not running)
postgres -D /usr/local/var/postgres
```

### Method 2: Docker Deployment
```bash
# Build and run all services
docker-compose up -d

# Or use make command
make docker

# Verify services
docker-compose ps

# View logs
docker-compose logs -f
```

### Method 3: Production Deployment
```bash
# Build production images
docker build -t freeagentics-backend:v1.0.0-alpha -f Dockerfile.backend .
docker build -t freeagentics-frontend:v1.0.0-alpha -f Dockerfile.frontend .

# Run with production config
docker-compose -f docker-compose.prod.yml up -d
```

## Configuration

### 1. Environment Variables
Create `.env` file in project root:
```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env
```

### 2. Essential Configuration
```env
# Application
APP_NAME=FreeAgentics
APP_VERSION=1.0.0-alpha
ENVIRONMENT=production
DEBUG=false

# Database
DATABASE_URL=postgresql://freeagentics:password@localhost:5432/freeagentics
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-here-minimum-32-chars
JWT_SECRET_KEY=your-jwt-secret-minimum-32-chars
JWT_ALGORITHM=RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Monitoring
ENABLE_MONITORING=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
```

### 3. Production Secrets
```bash
# Generate secure secrets
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Store in environment or secrets manager
export SECRET_KEY="generated-secret-key"
export JWT_SECRET_KEY="generated-jwt-key"
```

## Database Setup

### 1. PostgreSQL Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql
brew services start postgresql

# Create database and user
sudo -u postgres psql
```

### 2. Database Initialization
```sql
-- In PostgreSQL console
CREATE USER freeagentics WITH PASSWORD 'secure-password';
CREATE DATABASE freeagentics OWNER freeagentics;
GRANT ALL PRIVILEGES ON DATABASE freeagentics TO freeagentics;
\q
```

### 3. Run Migrations
```bash
# Initialize database schema
alembic upgrade head

# Verify migrations
alembic current
```

### 4. Seed Data (Optional)
```bash
# Load initial data
python scripts/seed_database.py

# Or use make command
make seed-db
```

## Security Configuration

### 1. SSL/TLS Setup
```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Production: Use Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

### 2. Firewall Configuration
```bash
# Allow required ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API (if needed)
sudo ufw enable
```

### 3. Security Headers
Configure in `nginx.conf`:
```nginx
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Strict-Transport-Security "max-age=31536000" always;
add_header Content-Security-Policy "default-src 'self'" always;
```

## Deployment Options

### Option 1: Single Server Deployment
```bash
# Install dependencies
make install

# Configure environment
cp .env.example .env
# Edit .env with production values

# Start services
make deploy-prod

# Verify deployment
curl http://localhost:8000/health
```

### Option 2: Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-stack.yml freeagentics

# Scale services
docker service scale freeagentics_api=3
docker service scale freeagentics_worker=5
```

### Option 3: Kubernetes Deployment
```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Deploy services
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml

# Expose services
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get all -n freeagentics
```

### Option 4: Cloud Deployment (AWS Example)
```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker build -t freeagentics .
docker tag freeagentics:latest $ECR_URI/freeagentics:v1.0.0-alpha
docker push $ECR_URI/freeagentics:v1.0.0-alpha

# Deploy with ECS/Fargate
aws ecs create-cluster --cluster-name freeagentics
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster freeagentics --service-name freeagentics-api
```

## Monitoring Setup

### 1. Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'freeagentics-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

### 2. Grafana Dashboards
```bash
# Import dashboards
curl -X POST http://admin:admin@localhost:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboards/api-metrics.json
```

### 3. Alerting Rules
```yaml
# alerts.yml
groups:
  - name: freeagentics
    rules:
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes > 1e9
        for: 5m
        annotations:
          summary: "High memory usage detected"

      - alert: APILatency
        expr: http_request_duration_seconds{quantile="0.95"} > 0.5
        for: 5m
        annotations:
          summary: "API latency is high"
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Error
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -U freeagentics -h localhost -d freeagentics

# Check logs
sudo tail -f /var/log/postgresql/postgresql-*.log
```

#### 2. Redis Connection Error
```bash
# Check Redis status
redis-cli ping

# Restart Redis
sudo systemctl restart redis

# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log
```

#### 3. Port Already in Use
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use make command
make kill-ports
```

#### 4. Memory Issues
```bash
# Check memory usage
free -h

# Monitor processes
htop

# Adjust worker processes in .env
API_WORKERS=2  # Reduce if memory constrained
```

#### 5. Permission Errors
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod -R 755 .

# Fix PostgreSQL permissions
sudo -u postgres psql -c "ALTER USER freeagentics CREATEDB;"
```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose output
uvicorn api.main:app --log-level debug
```

### Health Checks
```bash
# API health check
curl http://localhost:8000/health

# Database health check
curl http://localhost:8000/health/db

# Redis health check
curl http://localhost:8000/health/redis

# Complete system check
make health-check
```

## Quick Start Commands

### Development
```bash
# Complete setup and run
make install && make dev

# Run tests
make test

# Format code
make format

# Lint code
make lint
```

### Production
```bash
# Deploy production
make deploy-prod

# Monitor logs
make logs

# Backup database
make backup-db

# Update deployment
git pull && make update-prod
```

### Maintenance
```bash
# Stop all services
make stop

# Clean up
make clean

# Reset database
make reset-db

# Full reset
make reset
```

## Security Considerations

### Production Checklist
- [ ] Change all default passwords
- [ ] Enable SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up intrusion detection
- [ ] Enable audit logging
- [ ] Configure backup strategy
- [ ] Set up monitoring alerts
- [ ] Review security headers
- [ ] Disable debug mode
- [ ] Rotate secrets regularly

### Backup Strategy
```bash
# Automated backups
0 2 * * * /usr/local/bin/backup-freeagentics.sh

# Manual backup
pg_dump -U freeagentics freeagentics > backup_$(date +%Y%m%d).sql
```

## Support

### Getting Help
- Documentation: `/docs` directory
- API Docs: http://localhost:8000/docs
- Logs: `/var/log/freeagentics/`
- Issues: https://github.com/your-org/freeagentics/issues

### Useful Commands
```bash
# View all make commands
make help

# Check system status
make status

# Run diagnostics
make diagnose

# Generate support bundle
make support-bundle
```

---

**Version**: v1.0.0-alpha  
**Last Updated**: 2025-07-17  
**Status**: ALPHA - Not for production use without careful evaluation