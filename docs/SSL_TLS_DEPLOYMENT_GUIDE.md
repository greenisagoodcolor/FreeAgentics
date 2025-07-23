# SSL/TLS Deployment Guide for FreeAgentics

This guide provides comprehensive instructions for deploying SSL/TLS certificates and ensuring HTTPS enforcement across all FreeAgentics environments.

## Table of Contents

1. [Overview](#overview)
2. [Development Environment](#development-environment)
3. [Production Deployment](#production-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Load Balancer Configuration](#load-balancer-configuration)
6. [Certificate Management](#certificate-management)
7. [Security Best Practices](#security-best-practices)
8. [Monitoring and Alerts](#monitoring-and-alerts)
9. [Troubleshooting](#troubleshooting)

## Overview

FreeAgentics implements comprehensive SSL/TLS security with:
- Automatic HTTP to HTTPS redirect
- HSTS (HTTP Strict Transport Security) with preload
- Secure cookie flags
- Mixed content prevention
- Let's Encrypt integration for automatic certificate renewal
- Certificate monitoring and alerting
- Multi-domain and wildcard certificate support
- Zero-downtime SSL deployment

### Security Requirements

- **TLS Version**: Minimum TLS 1.2, preferred TLS 1.3
- **Cipher Suites**: Strong AEAD ciphers only (GCM, POLY1305)
- **Perfect Forward Secrecy**: ECDHE or DHE key exchange
- **HSTS**: 1-year max-age with includeSubDomains and preload
- **Certificate Validity**: 90-day certificates with 30-day renewal
- **OCSP Stapling**: Enabled for performance and privacy

## Development Environment

### Self-Signed Certificates

For local development, generate self-signed certificates:

```bash
# Using the built-in script
./scripts/setup-ssl-production.sh self-signed

# Or manually with Python
python -c "from auth.https_enforcement import generate_self_signed_cert; generate_self_signed_cert('localhost')"
```

### Docker Compose Development

```yaml
# docker-compose.override.yml
version: '3.8'

services:
  nginx:
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
    environment:
      - PRODUCTION=false
      - SSL_CERT=/etc/nginx/ssl/localhost.crt
      - SSL_KEY=/etc/nginx/ssl/localhost.key
```

### Testing HTTPS Locally

```bash
# Start services with SSL
docker-compose up -d

# Test HTTPS
curl -k https://localhost/health

# Test HTTP redirect
curl -I http://localhost/api/data
```

## Production Deployment

### Prerequisites

1. **Domain Configuration**:
   - Ensure DNS A/AAAA records point to your server
   - Configure CAA records for Let's Encrypt: `0 issue "letsencrypt.org"`

2. **Environment Variables**:
   ```bash
   export DOMAIN="freeagentics.com"
   export LETSENCRYPT_EMAIL="admin@freeagentics.com"
   export LETSENCRYPT_DOMAINS="freeagentics.com,www.freeagentics.com,api.freeagentics.com"
   export PRODUCTION=true
   ```

### Let's Encrypt Setup

#### Initial Certificate Generation

```bash
# Run the automated setup script
./scripts/setup-ssl-production.sh letsencrypt

# Or use the manual process
./nginx/setup-letsencrypt.sh
```

#### Manual Let's Encrypt Process

```bash
# 1. Install certbot
sudo apt-get update
sudo apt-get install -y certbot

# 2. Create webroot directory
mkdir -p /var/www/certbot

# 3. Obtain certificate
certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email $LETSENCRYPT_EMAIL \
  --agree-tos \
  --no-eff-email \
  --force-renewal \
  -d $DOMAIN \
  -d www.$DOMAIN \
  -d api.$DOMAIN

# 4. Copy certificates
sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem /etc/nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem /etc/nginx/ssl/key.pem
sudo chmod 600 /etc/nginx/ssl/key.pem
```

### Nginx Configuration

The production nginx configuration is in `/nginx/conf.d/ssl-freeagentics.conf`:

```nginx
server {
    listen 80;
    server_name _;

    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    # Redirect all HTTP to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name freeagentics.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # SSL Protocols
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;

    # Strong cipher suites
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

    # ... rest of configuration
}
```

### Docker Production Deployment

```bash
# Deploy with SSL
docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d

# Verify SSL deployment
docker-compose exec nginx nginx -t
docker-compose logs nginx
```

## Kubernetes Deployment

### Install cert-manager

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s
```

### Deploy Certificate Configuration

```bash
# Set environment variables
export LETSENCRYPT_EMAIL="admin@freeagentics.com"
export CLOUDFLARE_API_TOKEN="your-cloudflare-api-token"

# Apply cert-manager configuration
envsubst < k8s/cert-manager.yaml | kubectl apply -f -

# Verify certificate creation
kubectl get certificates -n freeagentics-prod
kubectl describe certificate freeagentics-tls -n freeagentics-prod
```

### Update Ingress for TLS

The ingress configuration in `k8s/ingress.yaml` includes:

```yaml
spec:
  tls:
  - hosts:
    - freeagentics.com
    - www.freeagentics.com
    - api.freeagentics.com
    secretName: freeagentics-tls
```

### Apply Ingress

```bash
kubectl apply -f k8s/ingress.yaml
```

## Load Balancer Configuration

### AWS Application Load Balancer

```bash
# Create SSL certificate in ACM
aws acm request-certificate \
  --domain-name freeagentics.com \
  --subject-alternative-names "*.freeagentics.com" \
  --validation-method DNS

# Configure ALB listener
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=$CERT_ARN \
  --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN \
  --ssl-policy ELBSecurityPolicy-TLS-1-2-2017-01
```

### Google Cloud Load Balancer

```bash
# Create managed SSL certificate
gcloud compute ssl-certificates create freeagentics-cert \
  --domains=freeagentics.com,www.freeagentics.com

# Create HTTPS load balancer
gcloud compute target-https-proxies create freeagentics-https-proxy \
  --ssl-certificates=freeagentics-cert
```

### Nginx as Load Balancer

For nginx as a load balancer with SSL termination:

```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 443 ssl http2;

    # SSL termination at load balancer
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://backend;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-SSL on;
    }
}
```

## Certificate Management

### Automatic Renewal

#### Systemd Timer (Linux)

```bash
# Create renewal service
sudo tee /etc/systemd/system/cert-renewal.service <<EOF
[Unit]
Description=Renew Let's Encrypt certificates
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/renew-letsencrypt.sh
User=root
EOF

# Create renewal timer
sudo tee /etc/systemd/system/cert-renewal.timer <<EOF
[Unit]
Description=Run cert-renewal twice daily
Requires=cert-renewal.service

[Timer]
OnCalendar=*-*-* 00,12:00:00
RandomizedDelaySec=3600
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable and start timer
sudo systemctl daemon-reload
sudo systemctl enable cert-renewal.timer
sudo systemctl start cert-renewal.timer
```

#### Cron Job

```bash
# Add to crontab
0 0,12 * * * /usr/local/bin/renew-letsencrypt.sh >> /var/log/cert-renewal.log 2>&1
```

### Certificate Monitoring

#### Command Line Check

```bash
# Check certificate expiry
openssl x509 -in /etc/nginx/ssl/cert.pem -noout -enddate

# Verify certificate chain
openssl verify -CAfile /etc/nginx/ssl/chain.pem /etc/nginx/ssl/cert.pem

# Test SSL configuration
openssl s_client -connect freeagentics.com:443 -servername freeagentics.com
```

#### Monitoring Script

```bash
#!/bin/bash
# /usr/local/bin/check-ssl-expiry.sh

CERT_FILE="/etc/nginx/ssl/cert.pem"
WARNING_DAYS=30

expiry_date=$(openssl x509 -in "$CERT_FILE" -noout -enddate | cut -d= -f2)
expiry_epoch=$(date -d "$expiry_date" +%s)
current_epoch=$(date +%s)
days_until_expiry=$(( ($expiry_epoch - $current_epoch) / 86400 ))

if [ $days_until_expiry -lt $WARNING_DAYS ]; then
    echo "WARNING: Certificate expires in $days_until_expiry days"
    # Send alert
    curl -X POST https://api.freeagentics.com/alerts/ssl-expiry \
      -H "Content-Type: application/json" \
      -d "{\"days_remaining\": $days_until_expiry}"
fi
```

## Security Best Practices

### 1. Strong Cipher Configuration

```nginx
# Only use strong ciphers
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;

# Disable weak protocols
ssl_protocols TLSv1.2 TLSv1.3;
```

### 2. HSTS Configuration

```nginx
# Enable HSTS with preload
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
```

Submit to HSTS preload list: https://hstspreload.org/

### 3. Certificate Pinning

```python
# In auth/security_headers.py
certificate_pins = {
    "freeagentics.com": [
        "sha256-ACTUAL_PIN_BASE64",
        "sha256-BACKUP_PIN_BASE64"
    ]
}
```

### 4. OCSP Stapling

```nginx
# Enable OCSP stapling
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /etc/nginx/ssl/chain.pem;
```

### 5. Secure Headers

```nginx
# Security headers
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;
```

## Monitoring and Alerts

### Prometheus Metrics

```yaml
# Certificate expiry alert
- alert: SSLCertificateExpiringSoon
  expr: ssl_cert_not_after - time() < 30 * 24 * 60 * 60
  for: 1h
  labels:
    severity: warning
  annotations:
    summary: "SSL certificate expiring soon"
    description: "SSL certificate for {{ $labels.domain }} expires in {{ $value | humanizeDuration }}"
```

### Health Checks

```bash
# SSL health endpoint
curl https://freeagentics.com/ssl-health

# Expected response
{
  "ssl_status": "active",
  "protocol": "TLSv1.3",
  "cipher": "TLS_AES_256_GCM_SHA384",
  "cert_expiry_days": 75,
  "hsts_enabled": true
}
```

## Troubleshooting

### Common Issues

#### 1. Certificate Not Trusted

```bash
# Check certificate chain
openssl s_client -showcerts -connect freeagentics.com:443

# Verify intermediate certificates are included
cat /etc/nginx/ssl/cert.pem
cat /etc/nginx/ssl/chain.pem
```

#### 2. Mixed Content Warnings

```bash
# Check for HTTP resources
grep -r "http://" /var/www/html/

# Update to use protocol-relative URLs
sed -i 's|http://|//|g' /var/www/html/index.html
```

#### 3. HSTS Not Working

```bash
# Verify HSTS header
curl -I https://freeagentics.com | grep Strict-Transport-Security

# Clear HSTS cache in browser
# Chrome: chrome://net-internals/#hsts
```

#### 4. Let's Encrypt Rate Limits

```bash
# Check rate limit status
curl https://crt.sh/?q=freeagentics.com

# Use staging environment for testing
certbot certonly --staging ...
```

### SSL Labs Testing

Test your SSL configuration:
1. Visit https://www.ssllabs.com/ssltest/
2. Enter your domain
3. Aim for A+ rating

### Security Headers Testing

Check security headers:
1. Visit https://securityheaders.com/
2. Enter your domain
3. Fix any reported issues

## Zero-Downtime SSL Deployment

### Blue-Green Deployment

```bash
# 1. Deploy new SSL configuration to green environment
kubectl apply -f k8s/green-deployment.yaml

# 2. Test SSL on green environment
curl https://green.freeagentics.com/health

# 3. Switch traffic to green
kubectl patch service freeagentics-lb -p '{"spec":{"selector":{"version":"green"}}}'

# 4. Remove blue deployment
kubectl delete deployment freeagentics-blue
```

### Rolling Update

```bash
# Update SSL configuration
kubectl set image deployment/nginx nginx=nginx:latest

# Monitor rollout
kubectl rollout status deployment/nginx

# Rollback if needed
kubectl rollout undo deployment/nginx
```

## Conclusion

This SSL/TLS deployment ensures:
- ✅ All traffic encrypted with strong ciphers
- ✅ Automatic certificate renewal
- ✅ Zero-downtime deployments
- ✅ Protection against downgrade attacks
- ✅ Compliance with security best practices
- ✅ A+ rating on SSL Labs

For additional support, consult the security team or refer to the OWASP TLS guidelines.
