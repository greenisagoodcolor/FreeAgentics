# FreeAgentics Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying FreeAgentics to production environments with enterprise-grade reliability, security, and scalability.

## Table of Contents

1. [Prerequisites](#prerequisites)
1. [Architecture Overview](#architecture-overview)
1. [Deployment Options](#deployment-options)
1. [Docker Deployment](#docker-deployment)
1. [Kubernetes Deployment](#kubernetes-deployment)
1. [CI/CD Pipeline](#cicd-pipeline)
1. [Monitoring & Alerting](#monitoring--alerting)
1. [Security Configuration](#security-configuration)
1. [Database Management](#database-management)
1. [SSL/TLS Configuration](#ssltls-configuration)
1. [Backup & Recovery](#backup--recovery)
1. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04 LTS or CentOS 8+ (recommended)
- **CPU**: Minimum 4 cores, 8 cores recommended
- **Memory**: Minimum 8GB RAM, 16GB recommended
- **Storage**: Minimum 100GB SSD, 500GB recommended
- **Network**: Minimum 1Gbps connection

### Software Dependencies

- Docker 24.0+ and Docker Compose 2.0+
- Kubernetes 1.26+ (for K8s deployment)
- PostgreSQL 15+
- Redis 7.0+
- Nginx 1.20+
- Git 2.30+

### Domain and SSL Requirements

- Registered domain name
- DNS access for domain configuration
- SSL certificate (Let's Encrypt recommended)

## Architecture Overview

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    │     (Nginx)     │
                    └─────────────────┘
                             │
                    ┌─────────────────┐
                    │   Reverse Proxy │
                    │   (SSL/TLS)     │
                    └─────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
    ┌───────────────┐                ┌───────────────┐
    │   Frontend    │                │   Backend     │
    │   (Next.js)   │                │   (FastAPI)   │
    └───────────────┘                └───────────────┘
                                              │
                          ┌───────────────────┴───────────────────┐
                          │                                       │
                  ┌───────────────┐                      ┌───────────────┐
                  │   Database    │                      │     Cache     │
                  │ (PostgreSQL)  │                      │   (Redis)     │
                  └───────────────┘                      └───────────────┘
```

## Deployment Options

### Option 1: Docker Compose (Recommended for Small-Medium Scale)

- **Pros**: Simple setup, integrated services, easy development
- **Cons**: Limited scalability, single-host deployment
- **Best for**: Production environments with moderate traffic

### Option 2: Kubernetes (Recommended for Large Scale)

- **Pros**: High availability, auto-scaling, service discovery
- **Cons**: Complex setup, requires K8s knowledge
- **Best for**: Enterprise environments with high traffic

### Option 3: Cloud-Native (AWS/GCP/Azure)

- **Pros**: Managed services, auto-scaling, global distribution
- **Cons**: Cloud vendor lock-in, higher costs
- **Best for**: Cloud-first organizations

## Docker Deployment

### 1. Environment Setup

```bash
# Create production environment file
cp .env.example .env.production

# Edit production environment variables
nano .env.production
```

#### Required Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://freeagentics:SECURE_PASSWORD@postgres:5432/freeagentics
POSTGRES_PASSWORD=SECURE_DATABASE_PASSWORD

# Redis Configuration
REDIS_PASSWORD=SECURE_REDIS_PASSWORD

# Application Security
SECRET_KEY=SECURE_SECRET_KEY_32_CHARS_MIN
JWT_SECRET=SECURE_JWT_SECRET_32_CHARS_MIN
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Domain Configuration
DOMAIN=yourdomain.com
HTTPS_ONLY=true
SECURE_COOKIES=true

# SSL Configuration
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
ALERTMANAGER_ENABLED=true

# Backup Configuration
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=30
```

### 2. SSL Certificate Setup

```bash
# Option 1: Let's Encrypt (Recommended)
./nginx/setup-letsencrypt.sh yourdomain.com

# Option 2: Custom SSL Certificate
# Place your certificates in nginx/ssl/
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem
```

### 3. Database Initialization

```bash
# Initialize database structure
docker-compose -f docker-compose.production.yml run --rm migration

# Verify database connection
docker-compose -f docker-compose.production.yml exec postgres psql -U freeagentics -d freeagentics -c "\l"
```

### 4. Production Deployment

```bash
# Deploy with automated script
./deploy-production.sh --version v1.0.0

# Or manual deployment
docker-compose -f docker-compose.production.yml up -d
```

### 5. Health Verification

```bash
# Check all services
docker-compose -f docker-compose.production.yml ps

# Test endpoints
curl -k https://yourdomain.com/health
curl -k https://yourdomain.com/api/v1/health
```

## Kubernetes Deployment

### 1. Kubernetes Manifests

Create the following Kubernetes configurations:

#### Namespace Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: freeagentics-prod
  labels:
    name: freeagentics-prod
    environment: production
```

#### Database Deployment

```yaml
# k8s/postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: freeagentics-prod
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: freeagentics
        - name: POSTGRES_USER
          value: freeagentics
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: freeagentics-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
```

### 2. Application Deployment

```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: freeagentics-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: freeagentics/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: freeagentics-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: freeagentics-secrets
              key: redis-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
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

### 3. Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: freeagentics-ingress
  namespace: freeagentics-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "10"
    nginx.ingress.kubernetes.io/rate-limit-window: "1s"
spec:
  tls:
  - hosts:
    - yourdomain.com
    secretName: freeagentics-tls
  rules:
  - host: yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 3000
```

### 4. Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f k8s/

# Verify deployment
kubectl get pods -n freeagentics-prod
kubectl get services -n freeagentics-prod
kubectl get ingress -n freeagentics-prod
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/production-deploy.yml
name: Production Deployment

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          docker-compose -f docker-compose.test.yml up --build --exit-code-from tests
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker images
        env:
          DOCKER_REGISTRY: ${{ secrets.DOCKER_REGISTRY }}
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker build -t ${DOCKER_REGISTRY}/freeagentics/backend:${GITHUB_SHA} -f Dockerfile.production .
          docker push ${DOCKER_REGISTRY}/freeagentics/backend:${GITHUB_SHA}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to production
        run: |
          # Deploy using your preferred method
          ./deploy-production.sh --version ${GITHUB_SHA}
```

## Monitoring & Alerting

### Prometheus Configuration

```yaml
# monitoring/prometheus-production.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'freeagentics-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'freeagentics-frontend'
    static_configs:
      - targets: ['frontend:3000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

rule_files:
  - "/etc/prometheus/rules/*.yml"
```

### Grafana Dashboard Setup

```bash
# Deploy monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Import dashboards
./monitoring/deploy-dashboards.sh

# Configure alerts
./monitoring/setup-alerts.sh
```

### Key Metrics to Monitor

1. **Application Health**

   - Response times
   - Error rates
   - Request throughput
   - Active connections

1. **Infrastructure Health**

   - CPU utilization
   - Memory usage
   - Disk space
   - Network I/O

1. **Database Performance**

   - Query performance
   - Connection pool usage
   - Lock contention
   - Replication lag

1. **Security Metrics**

   - Failed authentication attempts
   - Rate limiting triggers
   - SSL certificate expiration
   - Suspicious activity patterns

## Security Configuration

### 1. Network Security

```bash
# Configure firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 5432/tcp
ufw deny 6379/tcp
ufw enable
```

### 2. Container Security

```yaml
# Security contexts in docker-compose.production.yml
services:
  backend:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp
    user: "1000:1000"
```

### 3. SSL/TLS Hardening

```nginx
# nginx/snippets/ssl-params.conf
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_timeout 1d;
ssl_session_cache shared:SSL:50m;
ssl_stapling on;
ssl_stapling_verify on;
```

## Database Management

### Migration Strategy

```bash
# Pre-deployment migration check
docker-compose -f docker-compose.production.yml run --rm migration alembic check

# Run migrations
docker-compose -f docker-compose.production.yml run --rm migration alembic upgrade head

# Rollback if needed
docker-compose -f docker-compose.production.yml run --rm migration alembic downgrade -1
```

### Database Backup

```bash
# Automated backup script
./scripts/database-backup.sh backup

# Restore from backup
./scripts/database-backup.sh restore /path/to/backup.sql.gz

# Verify backup integrity
./scripts/database-backup.sh verify /path/to/backup.sql.gz
```

## SSL/TLS Configuration

### Let's Encrypt Setup

```bash
# Initial certificate request
certbot certonly --webroot -w /var/www/certbot -d yourdomain.com

# Automated renewal
0 0,12 * * * /usr/bin/certbot renew --quiet
```

### Certificate Monitoring

```bash
# Monitor certificate expiration
./nginx/monitor-ssl.sh health-check

# Setup automated alerts
./nginx/monitor-ssl.sh setup-alerts
```

## Backup & Recovery

### Automated Backup Strategy

```bash
#!/bin/bash
# scripts/backup-production.sh

# Database backup
pg_dump -h postgres -U freeagentics -d freeagentics | gzip > /backups/db-$(date +%Y%m%d-%H%M%S).sql.gz

# File system backup
tar -czf /backups/files-$(date +%Y%m%d-%H%M%S).tar.gz /app/uploads /app/logs

# Upload to cloud storage
aws s3 cp /backups/ s3://your-backup-bucket/freeagentics/ --recursive
```

### Disaster Recovery Plan

1. **Recovery Time Objective (RTO)**: 1 hour
1. **Recovery Point Objective (RPO)**: 15 minutes
1. **Backup Frequency**: Every 6 hours
1. **Backup Retention**: 30 days local, 1 year cloud

### Recovery Procedures

```bash
# 1. Restore database
gunzip -c /backups/db-latest.sql.gz | psql -h postgres -U freeagentics -d freeagentics

# 2. Restore application files
tar -xzf /backups/files-latest.tar.gz -C /

# 3. Restart services
docker-compose -f docker-compose.production.yml restart

# 4. Verify recovery
curl -k https://yourdomain.com/health
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.production.yml logs backend

# Check resource usage
docker stats

# Check disk space
df -h
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
docker-compose -f docker-compose.production.yml exec postgres psql -U freeagentics -c "\l"

# Check connection pool
docker-compose -f docker-compose.production.yml exec backend python -c "from database.session import get_db; print('DB OK')"
```

#### 3. SSL Certificate Issues

```bash
# Check certificate validity
openssl x509 -in nginx/ssl/cert.pem -text -noout

# Test SSL configuration
./nginx/test-ssl.sh yourdomain.com
```

#### 4. Performance Issues

```bash
# Check application metrics
curl -k https://yourdomain.com/metrics

# Check database performance
docker-compose -f docker-compose.production.yml exec postgres psql -U freeagentics -c "SELECT * FROM pg_stat_activity;"

# Check system resources
htop
iotop
```

### Health Check Endpoints

- **Application Health**: `https://yourdomain.com/health`
- **API Health**: `https://yourdomain.com/api/v1/health`
- **Database Health**: `https://yourdomain.com/api/v1/health/database`
- **Cache Health**: `https://yourdomain.com/api/v1/health/cache`

### Log Locations

- **Application Logs**: `/var/log/freeagentics/`
- **Nginx Logs**: `/var/log/nginx/`
- **Database Logs**: `/var/log/postgresql/`
- **Container Logs**: `docker-compose logs [service]`

## Production Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database migrations tested
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] Security hardening applied
- [ ] Load testing completed
- [ ] Documentation updated

### Post-Deployment

- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Backup verification
- [ ] Performance baseline established
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Team notification sent

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**

   - Review monitoring dashboards
   - Check backup integrity
   - Review security logs
   - Update dependencies

1. **Monthly**

   - Security vulnerability scan
   - Performance optimization review
   - Database maintenance
   - Documentation updates

1. **Quarterly**

   - Disaster recovery testing
   - Security audit
   - Capacity planning review
   - Technology stack updates

### Contact Information

- **Technical Support**: support@freeagentics.com
- **Security Issues**: security@freeagentics.com
- **Emergency Contact**: +1-XXX-XXX-XXXX

______________________________________________________________________

This guide provides a comprehensive foundation for deploying FreeAgentics in production environments. For specific deployment scenarios or additional support, please contact our technical team.
