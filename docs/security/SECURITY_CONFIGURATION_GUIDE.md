# Security Configuration Guide

## Overview

This guide provides detailed instructions for configuring security features in FreeAgentics across different environments.

## Environment Configuration

### Development Environment

```bash
# .env.development
# Security Settings (Development)
JWT_SECRET_KEY=dev-secret-key-change-in-production
JWT_ALGORITHM=HS256  # Use RS256 in production
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60  # Longer for development
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# Rate Limiting (Relaxed for Development)
RATE_LIMIT_ENABLED=true
RATE_LIMIT_ANONYMOUS=100/minute
RATE_LIMIT_AUTHENTICATED=500/minute
REDIS_URL=redis://localhost:6379/0

# Security Headers (Development)
SECURITY_HEADERS_ENABLED=true
HSTS_ENABLED=false  # Disabled for local development
CSP_REPORT_ONLY=true  # Report-only mode

# Logging
LOG_LEVEL=DEBUG
SECURITY_LOG_LEVEL=INFO
AUDIT_LOG_ENABLED=true
```

### Production Environment

```bash
# .env.production
# Security Settings (Production)
JWT_SECRET_KEY=${JWT_SECRET_KEY}  # From secure vault
JWT_PRIVATE_KEY_PATH=/etc/ssl/private/jwt-private.pem
JWT_PUBLIC_KEY_PATH=/etc/ssl/certs/jwt-public.pem
JWT_ALGORITHM=RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
JWT_KEY_ROTATION_DAYS=30

# Rate Limiting (Production)
RATE_LIMIT_ENABLED=true
RATE_LIMIT_ANONYMOUS=60/minute
RATE_LIMIT_AUTHENTICATED=300/minute
REDIS_URL=${REDIS_URL}  # From secure configuration
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_SSL=true

# Security Headers (Production)
SECURITY_HEADERS_ENABLED=true
HSTS_ENABLED=true
HSTS_MAX_AGE=31536000
HSTS_INCLUDE_SUBDOMAINS=true
HSTS_PRELOAD=true
CSP_REPORT_URI=https://api.freeagentics.com/security/csp-report
EXPECT_CT_ENABLED=true
EXPECT_CT_ENFORCE=true

# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/freeagentics.crt
SSL_KEY_PATH=/etc/ssl/private/freeagentics.key
SSL_PROTOCOLS="TLSv1.2 TLSv1.3"
SSL_CIPHERS="ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384"

# Database Security
DATABASE_URL=${DATABASE_URL}  # Encrypted connection string
DATABASE_SSL_MODE=require
AUDIT_DATABASE_URL=${AUDIT_DATABASE_URL}

# Monitoring
LOG_LEVEL=INFO
SECURITY_LOG_LEVEL=INFO
AUDIT_LOG_ENABLED=true
SENTRY_DSN=${SENTRY_DSN}
PROMETHEUS_ENABLED=true
```

## Nginx Configuration

### SSL/TLS Configuration

```nginx
# /etc/nginx/sites-available/freeagentics-api
server {
    listen 80;
    server_name api.freeagentics.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.freeagentics.com;

    # SSL Certificate
    ssl_certificate /etc/ssl/certs/freeagentics.crt;
    ssl_certificate_key /etc/ssl/private/freeagentics.key;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers off;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/ssl/certs/freeagentics-chain.crt;

    # SSL Session Cache
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

    # Security Headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Request Size Limits
    client_max_body_size 10M;
    client_body_buffer_size 128k;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;

    # Timeouts
    client_body_timeout 12;
    client_header_timeout 12;
    keepalive_timeout 15;
    send_timeout 10;

    # Proxy to Application
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # Security
        proxy_set_header X-Frame-Options "DENY";
        proxy_hide_header X-Powered-By;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health Check Endpoint (No Auth Required)
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }

    # Metrics Endpoint (Restricted)
    location /metrics {
        allow 10.0.0.0/8;  # Internal network only
        deny all;
        proxy_pass http://localhost:8000/metrics;
    }
}
```

## Redis Security Configuration

### Redis Configuration File

```conf
# /etc/redis/redis.conf

# Network Security
bind 127.0.0.1 ::1
protected-mode yes
port 6379

# Authentication
requirepass your-strong-redis-password

# SSL/TLS
tls-port 6380
tls-cert-file /etc/redis/tls/redis.crt
tls-key-file /etc/redis/tls/redis.key
tls-ca-cert-file /etc/redis/tls/ca.crt
tls-dh-params-file /etc/redis/tls/redis.dh

# Persistence Security
dir /var/lib/redis
dbfilename dump.rdb
appendonly yes
appendfilename "appendonly.aof"

# Command Renaming (Security through Obscurity)
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
rename-command CONFIG "CONFIG_e4f9c8d7"

# Memory Limits
maxmemory 2gb
maxmemory-policy allkeys-lru

# Logging
logfile /var/log/redis/redis-server.log
loglevel notice
```

## Database Security

### PostgreSQL Security Configuration

```sql
-- Database User Setup
CREATE USER freeagentics_app WITH PASSWORD 'strong-password';
CREATE USER freeagentics_readonly WITH PASSWORD 'readonly-password';

-- Database Creation
CREATE DATABASE freeagentics OWNER freeagentics_app;
CREATE DATABASE freeagentics_audit OWNER freeagentics_app;

-- Permissions
GRANT CONNECT ON DATABASE freeagentics TO freeagentics_app;
GRANT CREATE ON DATABASE freeagentics TO freeagentics_app;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO freeagentics_readonly;

-- Row Level Security
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;

CREATE POLICY agent_isolation ON agents
    FOR ALL
    TO freeagentics_app
    USING (created_by = current_setting('app.current_user_id')::uuid
           OR current_setting('app.current_user_role') = 'admin');

-- SSL Configuration
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/postgresql/server.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/postgresql/server.key';
```

### Connection String Security

```python
# Use environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')

# Parse and validate
from urllib.parse import urlparse
parsed = urlparse(DATABASE_URL)

# Add SSL parameters
if parsed.scheme == 'postgresql':
    DATABASE_URL += '?sslmode=require&sslcert=/path/to/client-cert.pem'
```

## Docker Security Configuration

### Dockerfile Security

```dockerfile
# Use specific version
FROM python:3.11-slim-bookworm

# Create non-root user
RUN groupadd -r app && useradd -r -g app app

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Security scanning
RUN pip install safety bandit
COPY . /app
WORKDIR /app
RUN safety check && bandit -r /app -ll

# Change ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Run with limited privileges
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
```

### Docker Compose Security

```yaml
version: '3.8'

services:
  api:
    build: .
    image: freeagentics-api:latest
    container_name: freeagentics-api
    restart: unless-stopped

    # Security options
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

    # Environment variables from secrets
    env_file:
      - .env.production

    # Volumes
    volumes:
      - type: tmpfs
        target: /tmp
      - type: tmpfs
        target: /app/logs

    # Network
    networks:
      - internal
      - redis
      - postgres

    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    container_name: freeagentics-redis
    restart: unless-stopped
    command: redis-server /etc/redis/redis.conf

    # Security
    security_opt:
      - no-new-privileges:true
    read_only: true

    # Volumes
    volumes:
      - ./redis.conf:/etc/redis/redis.conf:ro
      - redis-data:/data
      - type: tmpfs
        target: /tmp

    # Network
    networks:
      - redis

  postgres:
    image: postgres:15-alpine
    container_name: freeagentics-postgres
    restart: unless-stopped

    # Security
    security_opt:
      - no-new-privileges:true

    # Environment
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
      POSTGRES_USER: freeagentics
      POSTGRES_DB: freeagentics

    # Secrets
    secrets:
      - db_password

    # Volumes
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./postgresql.conf:/etc/postgresql/postgresql.conf:ro

    # Network
    networks:
      - postgres

networks:
  internal:
    driver: bridge
  redis:
    driver: bridge
    internal: true
  postgres:
    driver: bridge
    internal: true

volumes:
  redis-data:
    driver: local
  postgres-data:
    driver: local

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

## Kubernetes Security Configuration

### Pod Security Policy

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: freeagentics-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

### Network Policy

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: freeagentics-netpol
spec:
  podSelector:
    matchLabels:
      app: freeagentics-api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: freeagentics
      ports:
        - protocol: TCP
          port: 5432  # PostgreSQL
        - protocol: TCP
          port: 6379  # Redis
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 443   # HTTPS for external APIs
        - protocol: TCP
          port: 53    # DNS
        - protocol: UDP
          port: 53    # DNS
```

## Monitoring and Alerting

### Prometheus Alerts

```yaml
# prometheus-alerts.yml
groups:
  - name: security
    rules:
      - alert: HighFailedLoginRate
        expr: rate(auth_failed_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High rate of failed login attempts

      - alert: RateLimitViolations
        expr: rate(rate_limit_exceeded_total[5m]) > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High rate of rate limit violations

      - alert: SuspiciousActivity
        expr: security_suspicious_activity_total > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Suspicious activity detected
```

## Security Checklist

### Pre-Deployment

- [ ] All secrets in environment variables or vault
- [ ] SSL/TLS certificates valid and not expiring
- [ ] Database connections use SSL
- [ ] Redis authentication enabled
- [ ] Rate limiting configured
- [ ] Security headers enabled
- [ ] CORS properly configured
- [ ] Input validation in place
- [ ] Output sanitization implemented

### Post-Deployment

- [ ] Security scan completed (OWASP ZAP)
- [ ] Penetration test passed
- [ ] SSL Labs score A or higher
- [ ] Security headers validated
- [ ] Rate limiting tested
- [ ] Monitoring alerts configured
- [ ] Audit logging verified
- [ ] Backup and recovery tested
- [ ] Incident response plan ready
