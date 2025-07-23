# FreeAgentics Production Deployment Guide

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Step-by-Step Deployment Procedures](#step-by-step-deployment-procedures)
3. [Configuration Management](#configuration-management)
4. [SSL/TLS Setup](#ssltls-setup)
5. [Database Migration Procedures](#database-migration-procedures)
6. [Zero-Downtime Deployment Strategies](#zero-downtime-deployment-strategies)
7. [Rollback Procedures](#rollback-procedures)
8. [Deployment Verification](#deployment-verification)

## Pre-Deployment Checklist

### Code Quality Checks
- [ ] All tests passing (`make test`)
- [ ] Code coverage meets requirements (≥80%)
- [ ] Security audit completed (`make security-audit`)
- [ ] Performance benchmarks passing
- [ ] All linting checks passing (`make lint`)
- [ ] Dependencies updated and audited

### Infrastructure Readiness
- [ ] Production servers provisioned
- [ ] Load balancers configured
- [ ] SSL certificates valid and installed
- [ ] Database backups completed
- [ ] Monitoring and alerting configured
- [ ] Log aggregation set up

### Configuration Verification
- [ ] Environment variables reviewed
- [ ] API keys and secrets secured in vault
- [ ] Database connection strings verified
- [ ] Redis configuration validated
- [ ] Rate limiting configured
- [ ] CORS settings reviewed

### Team Readiness
- [ ] Deployment window scheduled
- [ ] Rollback plan documented
- [ ] On-call personnel notified
- [ ] Communication channels open
- [ ] Customer notifications sent (if needed)

## Step-by-Step Deployment Procedures

### 1. Pre-Deployment Setup

```bash
# Clone the deployment repository
git clone https://github.com/youorg/freeagentics.git
cd freeagentics

# Checkout the release tag
git checkout tags/v1.0.0

# Verify the release
git tag --verify v1.0.0
```

### 2. Build Production Images

```bash
# Build the optimized production image
docker build -f Dockerfile.production -t freeagentics:v1.0.0 .

# Build the web frontend
cd web/
docker build -f Dockerfile.production -t freeagentics-web:v1.0.0 .
cd ..

# Tag images for registry
docker tag freeagentics:v1.0.0 your-registry/freeagentics:v1.0.0
docker tag freeagentics-web:v1.0.0 your-registry/freeagentics-web:v1.0.0

# Push to registry
docker push your-registry/freeagentics:v1.0.0
docker push your-registry/freeagentics-web:v1.0.0
```

### 3. Database Migration

```bash
# Run database migrations
./scripts/deployment/migrate-database.sh production

# Verify migration status
docker-compose -f docker-compose.production.yml exec api alembic current
```

### 4. Deploy Application

```bash
# Deploy using Docker Compose (single server)
docker-compose -f docker-compose.production.yml up -d

# Or deploy using orchestration (Kubernetes)
kubectl apply -f k8s/production/

# Or deploy using Docker Swarm
docker stack deploy -c docker-compose.production.yml freeagentics
```

### 5. Health Checks

```bash
# Run deployment verification
./scripts/deployment/verify-deployment.sh

# Check service health
curl -k https://api.yourdomain.com/health
curl -k https://api.yourdomain.com/v1/system/status
```

## Configuration Management

### Environment Variables

Create `.env.production` from the template:

```bash
cp .env.production.template .env.production
```

Required environment variables:

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@db:5432/freeagentics
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis Configuration
REDIS_URL=redis://redis:6379/0
REDIS_POOL_SIZE=10

# Security
JWT_SECRET_KEY=<generate-secure-key>
JWT_REFRESH_SECRET_KEY=<generate-secure-key>
API_KEY_SALT=<generate-secure-salt>

# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/server.crt
SSL_KEY_PATH=/etc/ssl/private/server.key
FORCE_HTTPS=true

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
SENTRY_DSN=<your-sentry-dsn>

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT=100/minute
RATE_LIMIT_AUTH=20/minute

# CORS
CORS_ORIGINS=https://app.yourdomain.com,https://www.yourdomain.com
```

### Secret Management

Use a secret management solution:

```bash
# Using HashiCorp Vault
vault kv put secret/freeagentics/production \
  jwt_secret_key="$(openssl rand -base64 32)" \
  jwt_refresh_secret_key="$(openssl rand -base64 32)" \
  database_password="<secure-password>" \
  api_key_salt="$(openssl rand -base64 32)"

# Or using Docker Secrets
echo "<jwt-secret>" | docker secret create jwt_secret_key -
echo "<db-password>" | docker secret create database_password -
```

## SSL/TLS Setup

### 1. Generate SSL Certificates

For production, use certificates from a trusted CA:

```bash
# Using Let's Encrypt with Certbot
certbot certonly --standalone -d api.yourdomain.com -d app.yourdomain.com

# Certificates will be in:
# /etc/letsencrypt/live/yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/yourdomain.com/privkey.pem
```

### 2. Configure Nginx with SSL

Update `nginx/nginx.production.conf`:

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # SSL Security Headers
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    # Other security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### 3. Certificate Renewal

Set up automatic renewal:

```bash
# Add to crontab
0 0 * * * /usr/bin/certbot renew --quiet --post-hook "docker-compose -f /path/to/docker-compose.production.yml restart nginx"
```

## Database Migration Procedures

### 1. Pre-Migration Backup

```bash
# Backup current database
./scripts/deployment/backup-database.sh production

# Verify backup
pg_restore --list backup_production_$(date +%Y%m%d).dump | head -20
```

### 2. Run Migrations

```bash
# Generate migration status report
docker-compose -f docker-compose.production.yml exec api alembic history

# Run pending migrations
docker-compose -f docker-compose.production.yml exec api alembic upgrade head

# Verify migration
docker-compose -f docker-compose.production.yml exec api alembic current
```

### 3. Migration Rollback (if needed)

```bash
# Rollback to previous revision
docker-compose -f docker-compose.production.yml exec api alembic downgrade -1

# Or rollback to specific revision
docker-compose -f docker-compose.production.yml exec api alembic downgrade <revision_id>
```

## Zero-Downtime Deployment Strategies

### Blue-Green Deployment

1. **Setup Blue Environment (Current)**
```bash
# Current production running on blue
docker-compose -f docker-compose.blue.yml up -d
```

2. **Deploy to Green Environment**
```bash
# Deploy new version to green
docker-compose -f docker-compose.green.yml up -d

# Run health checks on green
./scripts/deployment/health-check.sh green
```

3. **Switch Traffic**
```bash
# Update load balancer to point to green
./scripts/deployment/switch-to-green.sh

# Monitor for issues
./scripts/deployment/monitor-deployment.sh
```

4. **Keep Blue as Backup**
```bash
# Keep blue running for quick rollback if needed
# After verification, stop blue
docker-compose -f docker-compose.blue.yml down
```

### Rolling Update (Kubernetes)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: freeagentics-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: api
        image: your-registry/freeagentics:v1.0.0
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

Deploy:
```bash
kubectl apply -f k8s/production/deployment.yaml
kubectl rollout status deployment/freeagentics-api
```

### Canary Deployment

1. **Deploy Canary Version**
```bash
# Deploy new version to small subset
docker-compose -f docker-compose.canary.yml up -d

# Configure load balancer for 10% traffic to canary
./scripts/deployment/configure-canary.sh 10
```

2. **Monitor Metrics**
```bash
# Monitor error rates and performance
./scripts/deployment/monitor-canary.sh

# If metrics are good, increase traffic
./scripts/deployment/configure-canary.sh 50
```

3. **Full Deployment**
```bash
# If canary is successful, deploy to all
./scripts/deployment/promote-canary.sh
```

## Rollback Procedures

### Immediate Rollback

```bash
# 1. Switch load balancer to previous version
./scripts/deployment/rollback-traffic.sh

# 2. Stop problematic deployment
docker-compose -f docker-compose.production.yml down

# 3. Restore previous version
docker-compose -f docker-compose.production-backup.yml up -d

# 4. Verify services
./scripts/deployment/verify-deployment.sh
```

### Database Rollback

```bash
# 1. Stop application services
docker-compose -f docker-compose.production.yml stop api worker

# 2. Rollback database migration
docker-compose -f docker-compose.production.yml exec api alembic downgrade -1

# 3. Deploy previous application version
docker checkout tags/v0.9.0
docker-compose -f docker-compose.production.yml up -d

# 4. Verify rollback
./scripts/deployment/verify-rollback.sh
```

### Emergency Rollback Script

Create `scripts/deployment/emergency-rollback.sh`:

```bash
#!/bin/bash
set -e

echo "🚨 Starting emergency rollback..."

# Save current state for debugging
docker-compose -f docker-compose.production.yml logs > rollback_logs_$(date +%Y%m%d_%H%M%S).log

# Switch to previous version
./scripts/deployment/switch-to-backup.sh

# Notify team
./scripts/deployment/notify-team.sh "Emergency rollback initiated"

echo "✅ Emergency rollback completed"
```

## Deployment Verification

### Health Check Script

Create `scripts/deployment/verify-deployment.sh`:

```bash
#!/bin/bash
set -e

echo "🔍 Verifying deployment..."

# API Health
echo "Checking API health..."
curl -f -s https://api.yourdomain.com/health || exit 1

# Database Connectivity
echo "Checking database connectivity..."
curl -f -s https://api.yourdomain.com/v1/system/status | jq '.database' || exit 1

# Redis Connectivity
echo "Checking Redis connectivity..."
curl -f -s https://api.yourdomain.com/v1/system/status | jq '.redis' || exit 1

# WebSocket Service
echo "Checking WebSocket service..."
wscat -c wss://api.yourdomain.com/ws/health || exit 1

# Monitoring Endpoints
echo "Checking monitoring endpoints..."
curl -f -s https://api.yourdomain.com/metrics || exit 1

echo "✅ All health checks passed!"
```

### Smoke Tests

```bash
#!/bin/bash
# scripts/deployment/smoke-tests.sh

echo "🔥 Running smoke tests..."

# Test authentication
TOKEN=$(curl -s -X POST https://api.yourdomain.com/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test@example.com","password":"testpass"}' | jq -r '.access_token')

# Test authenticated endpoint
curl -f -s -H "Authorization: Bearer $TOKEN" \
  https://api.yourdomain.com/v1/agents || exit 1

# Test agent creation
curl -f -s -X POST -H "Authorization: Bearer $TOKEN" \
  https://api.yourdomain.com/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"name":"smoke-test","type":"resource_collector"}' || exit 1

echo "✅ Smoke tests passed!"
```

### Monitoring Verification

```bash
# Check Prometheus metrics
curl -s http://prometheus:9090/api/v1/query?query=up | jq '.data.result'

# Check Grafana dashboards
curl -s -u admin:admin http://grafana:3000/api/dashboards/home | jq '.[]'

# Verify alerting
curl -s http://alertmanager:9093/api/v1/alerts | jq '.[]'
```

## Post-Deployment Tasks

1. **Monitor Application Metrics**
   - Check error rates
   - Monitor response times
   - Watch resource utilization
   - Review log aggregation

2. **Update Documentation**
   - Update deployment log
   - Document any issues encountered
   - Update runbooks if needed

3. **Notify Stakeholders**
   - Send deployment completion notification
   - Update status page
   - Close deployment window

4. **Schedule Post-Mortem (if issues occurred)**
   - Document what went wrong
   - Identify improvement areas
   - Update procedures

## Troubleshooting Common Issues

### Container Won't Start
```bash
# Check logs
docker-compose -f docker-compose.production.yml logs api

# Check resource constraints
docker stats

# Verify environment variables
docker-compose -f docker-compose.production.yml config
```

### Database Connection Issues
```bash
# Test database connectivity
docker-compose -f docker-compose.production.yml exec api \
  python -c "from database.session import engine; engine.execute('SELECT 1')"

# Check connection pool
docker-compose -f docker-compose.production.yml exec api \
  python -c "from database.session import engine; print(engine.pool.status())"
```

### SSL Certificate Issues
```bash
# Verify certificate
openssl x509 -in /etc/letsencrypt/live/yourdomain.com/cert.pem -text -noout

# Test SSL configuration
openssl s_client -connect api.yourdomain.com:443 -servername api.yourdomain.com
```

### Performance Issues
```bash
# Check resource usage
docker stats

# Review slow queries
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U freeagentics -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check Redis performance
docker-compose -f docker-compose.production.yml exec redis redis-cli --stat
```

## Maintenance Windows

Schedule regular maintenance windows for:
- Security updates
- Dependency updates
- Database maintenance
- Certificate renewal
- Performance optimization

Typical maintenance schedule:
- **Weekly**: Security patches (if critical)
- **Monthly**: Regular updates and optimization
- **Quarterly**: Major version updates
- **Annually**: Infrastructure review and updates
