# FreeAgentics Deployment Guide

## Overview

This guide covers deploying the FreeAgentics platform in production environments. We'll cover Docker deployment, Kubernetes orchestration, security considerations, and monitoring setup.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Security Setup](#security-setup)
6. [SSL/TLS Configuration](#ssltls-configuration)
7. [Database Setup](#database-setup)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Scaling Strategies](#scaling-strategies)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Production Requirements:**

- CPU: 4 cores
- RAM: 8GB
- Storage: 100GB SSD
- Network: 1Gbps
- Docker 20.10+
- Docker Compose 2.0+

**Recommended Production Configuration:**

- CPU: 8+ cores
- RAM: 16GB+
- Storage: 500GB SSD with backup
- Network: 10Gbps
- Kubernetes 1.25+

### Required Services

- PostgreSQL 15+
- Redis 7+
- Nginx (reverse proxy)
- SSL certificates

## Docker Deployment

### Quick Start

```bash
# Clone the repository
git clone https://github.com/freeagentics/freeagentics.git
cd freeagentics

# Create environment file
cp .env.example .env

# Edit configuration
nano .env

# Start services
docker-compose -f docker-compose.production.yml up -d
```

### Production Docker Compose

```yaml
# docker-compose.production.yml
version: "3.8"

services:
  frontend:
    image: freeagentics/web:latest
    container_name: freeagentics-web
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=https://api.yourdomain.com
      - NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - freeagentics

  api:
    image: freeagentics/api:latest
    container_name: freeagentics-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/freeagentics
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - CORS_ORIGINS=https://yourdomain.com
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - freeagentics

  postgres:
    image: postgres:15-alpine
    container_name: freeagentics-db
    restart: unless-stopped
    environment:
      - POSTGRES_DB=freeagentics
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init:/docker-entrypoint-initdb.d
    networks:
      - freeagentics

  redis:
    image: redis:7-alpine
    container_name: freeagentics-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - freeagentics

  nginx:
    image: nginx:alpine
    container_name: freeagentics-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/conf.d:/etc/nginx/conf.d
      - ./ssl:/etc/nginx/ssl
      - ./nginx/dhparam.pem:/etc/nginx/dhparam.pem
    depends_on:
      - frontend
      - api
    networks:
      - freeagentics

volumes:
  postgres_data:
  redis_data:

networks:
  freeagentics:
    driver: bridge
```

### Building Images

```bash
# Build frontend
cd web
docker build -t freeagentics/web:latest -f Dockerfile.production .

# Build API
cd ..
docker build -t freeagentics/api:latest -f Dockerfile.production .

# Push to registry
docker push freeagentics/web:latest
docker push freeagentics/api:latest
```

## Kubernetes Deployment

### Namespace Setup

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: freeagentics
```

### Frontend Deployment

```yaml
# k8s/frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: freeagentics
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
        - name: frontend
          image: freeagentics/web:latest
          ports:
            - containerPort: 3000
          env:
            - name: NODE_ENV
              value: "production"
            - name: NEXT_PUBLIC_API_URL
              value: "https://api.freeagentics.com"
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
              port: 3000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  namespace: freeagentics
spec:
  selector:
    app: frontend
  ports:
    - protocol: TCP
      port: 80
      targetPort: 3000
  type: ClusterIP
```

### API Deployment

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: freeagentics
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
        - name: api
          image: freeagentics/api:latest
          ports:
            - containerPort: 8000
          env:
            - name: ENVIRONMENT
              value: "production"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: url
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: redis-secret
                  key: url
            - name: JWT_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: jwt-secret
                  key: key
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
            initialDelaySeconds: 60
            periodSeconds: 20
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: freeagentics
spec:
  selector:
    app: api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: freeagentics-ingress
  namespace: freeagentics
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
    - hosts:
        - freeagentics.com
        - api.freeagentics.com
      secretName: freeagentics-tls
  rules:
    - host: freeagentics.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend-service
                port:
                  number: 80
    - host: api.freeagentics.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80
```

## Environment Configuration

### Required Environment Variables

```bash
# .env.production
# Application
NODE_ENV=production
ENVIRONMENT=production
LOG_LEVEL=info

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Frontend Configuration
NEXT_PUBLIC_API_URL=https://api.freeagentics.com
NEXT_PUBLIC_WS_URL=wss://api.freeagentics.com
NEXT_PUBLIC_ENV=production

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/freeagentics
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=0

# Redis
REDIS_URL=redis://redis:6379
REDIS_PASSWORD=your-redis-password

# Authentication
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=RS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Security
SECRET_KEY=your-app-secret-key
CORS_ORIGINS=https://freeagentics.com
ALLOWED_HOSTS=freeagentics.com,api.freeagentics.com

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_ENABLED=true
```

### Secrets Management

```bash
# Create Kubernetes secrets
kubectl create secret generic db-secret \
  --from-literal=url=postgresql://user:pass@postgres:5432/freeagentics \
  -n freeagentics

kubectl create secret generic redis-secret \
  --from-literal=url=redis://redis:6379 \
  -n freeagentics

kubectl create secret generic jwt-secret \
  --from-literal=key=your-jwt-secret-key \
  -n freeagentics
```

## Security Setup

### Security Headers (Nginx)

```nginx
# nginx/conf.d/security.conf
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

# Hide server version
server_tokens off;

# Limit request methods
if ($request_method !~ ^(GET|HEAD|POST|PUT|DELETE|OPTIONS)$) {
    return 405;
}
```

### Rate Limiting

```nginx
# nginx/conf.d/rate-limiting.conf
# Define rate limit zones
limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;
limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/m;

# Apply rate limits
location /api/v1/login {
    limit_req zone=auth burst=2 nodelay;
    proxy_pass http://api-service;
}

location /api/v1/prompts {
    limit_req zone=api burst=10 nodelay;
    proxy_pass http://api-service;
}
```

## SSL/TLS Configuration

### Generate SSL Certificate

```bash
# Using Let's Encrypt
certbot certonly --nginx -d freeagentics.com -d api.freeagentics.com

# Generate DH parameters
openssl dhparam -out /etc/nginx/dhparam.pem 2048
```

### Nginx SSL Configuration

```nginx
# nginx/conf.d/ssl.conf
server {
    listen 443 ssl http2;
    server_name api.freeagentics.com;

    ssl_certificate /etc/letsencrypt/live/freeagentics.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/freeagentics.com/privkey.pem;
    ssl_dhparam /etc/nginx/dhparam.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/freeagentics.com/chain.pem;

    location / {
        proxy_pass http://api-service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://api-service;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Database Setup

### PostgreSQL Initialization

```sql
-- postgres/init/01-init.sql
CREATE DATABASE freeagentics;
CREATE USER freeagentics_user WITH ENCRYPTED PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE freeagentics TO freeagentics_user;

-- Create extensions
\c freeagentics;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
```

### Database Migrations

```bash
# Run migrations
docker exec -it freeagentics-api alembic upgrade head

# Create migration
docker exec -it freeagentics-api alembic revision --autogenerate -m "Description"
```

### Backup Strategy

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="freeagentics"

# Create backup
docker exec freeagentics-db pg_dump -U postgres $DB_NAME | gzip > $BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "freeagentics-api"
    static_configs:
      - targets: ["api-service:8000"]
    metrics_path: "/metrics"

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]
```

### Grafana Dashboards

1. Import dashboard templates from `monitoring/grafana/dashboards/`
2. Configure data source to point to Prometheus
3. Set up alerts for critical metrics

### Log Aggregation

```yaml
# monitoring/fluent-bit.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: freeagentics
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         1
        Log_Level     info
        Daemon        off

    [INPUT]
        Name              tail
        Path              /var/log/containers/*freeagentics*.log
        Parser            docker
        Tag               freeagentics.*
        Refresh_Interval  5

    [OUTPUT]
        Name              forward
        Match             *
        Host              elasticsearch
        Port              9200
```

## Scaling Strategies

### Horizontal Scaling

```bash
# Scale API replicas
kubectl scale deployment api --replicas=5 -n freeagentics

# Scale frontend replicas
kubectl scale deployment frontend --replicas=3 -n freeagentics
```

### Auto-scaling Configuration

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: freeagentics
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Database Scaling

1. **Read Replicas**: Add PostgreSQL read replicas for read-heavy workloads
2. **Connection Pooling**: Use PgBouncer for connection pooling
3. **Partitioning**: Partition large tables by date or ID range

## Troubleshooting

### Common Issues

**Container won't start**

```bash
# Check logs
docker logs freeagentics-api
kubectl logs -f deployment/api -n freeagentics

# Check events
kubectl get events -n freeagentics
```

**Database connection errors**

```bash
# Test connection
docker exec -it freeagentics-api python -c "from database.session import test_connection; test_connection()"

# Check credentials
echo $DATABASE_URL | base64 -d
```

**High memory usage**

```bash
# Check resource usage
kubectl top pods -n freeagentics

# Increase limits
kubectl edit deployment api -n freeagentics
```

### Health Checks

```bash
# API health
curl https://api.freeagentics.com/health

# Frontend health
curl https://freeagentics.com/health

# Database health
docker exec freeagentics-db pg_isready
```

### Performance Optimization

1. **Enable Redis caching**: Ensure Redis is properly configured
2. **Optimize database queries**: Use EXPLAIN ANALYZE
3. **CDN for static assets**: Configure CloudFlare or similar
4. **Compress responses**: Enable gzip in Nginx

## Maintenance

### Regular Tasks

1. **Daily**: Check logs and metrics
2. **Weekly**: Run database maintenance
3. **Monthly**: Update dependencies and security patches
4. **Quarterly**: Review and optimize performance

### Update Procedure

```bash
# 1. Build new images
docker build -t freeagentics/api:v1.2.0 .

# 2. Update deployment
kubectl set image deployment/api api=freeagentics/api:v1.2.0 -n freeagentics

# 3. Monitor rollout
kubectl rollout status deployment/api -n freeagentics

# 4. Rollback if needed
kubectl rollout undo deployment/api -n freeagentics
```

## Conclusion

This deployment guide provides a comprehensive approach to deploying FreeAgentics in production. Always test changes in a staging environment first, monitor system health continuously, and maintain regular backups.

For additional support, consult the operations team or refer to the specific component documentation.
