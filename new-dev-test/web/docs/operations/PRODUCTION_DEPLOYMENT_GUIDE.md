# Production Deployment Guide - FreeAgentics Frontend

## Overview

This guide provides comprehensive instructions for deploying the FreeAgentics frontend to production environments. The frontend is optimized for performance, security, and scalability.

## Prerequisites

### System Requirements

- **Node.js**: 18.0.0 or higher
- **npm**: 8.0.0 or higher
- **Docker**: 20.10.0 or higher (for containerized deployment)
- **SSL Certificate**: Required for HTTPS

### Environment Variables

Create a `.env.production` file with the following variables:

```env
# API Configuration
NEXT_PUBLIC_API_URL=https://api.freeagentics.com
NEXT_PUBLIC_WS_URL=wss://api.freeagentics.com
NEXT_PUBLIC_APP_URL=https://app.freeagentics.com

# Analytics (Optional)
NEXT_PUBLIC_GA_MEASUREMENT_ID=G-XXXXXXXXXX
NEXT_PUBLIC_POSTHOG_KEY=phc_xxxxxxxxxx

# Security
NEXT_PUBLIC_ENVIRONMENT=production
```

## Deployment Methods

### Method 1: Docker Deployment (Recommended)

#### 1. Build Production Image

```bash
# Build optimized production image
docker build -f Dockerfile.production -t freeagentics-web:latest .

# Verify image size and contents
docker images freeagentics-web:latest
```

#### 2. Run Production Container

```bash
# Run with environment variables
docker run -d \
  --name freeagentics-web \
  -p 3000:3000 \
  --env-file .env.production \
  --restart unless-stopped \
  freeagentics-web:latest

# Check container health
docker ps
docker logs freeagentics-web
```

#### 3. Production Docker Compose

```yaml
version: "3.8"
services:
  web:
    image: freeagentics-web:latest
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=https://api.freeagentics.com
      - NEXT_PUBLIC_WS_URL=wss://api.freeagentics.com
      - NEXT_PUBLIC_APP_URL=https://app.freeagentics.com
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Method 2: Node.js Deployment

#### 1. Install Dependencies

```bash
# Install production dependencies
npm ci --production

# Build the application
npm run build

# Verify build
npm run start
```

#### 2. Process Management with PM2

```bash
# Install PM2 globally
npm install -g pm2

# Create PM2 configuration
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'freeagentics-web',
    script: 'npm',
    args: 'start',
    instances: 'max',
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'production',
      PORT: 3000
    }
  }]
};
EOF

# Start with PM2
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save
pm2 startup
```

### Method 3: Serverless Deployment (Vercel/Netlify)

#### Vercel Deployment

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy to Vercel
vercel --prod

# Set environment variables
vercel env add NEXT_PUBLIC_API_URL
vercel env add NEXT_PUBLIC_WS_URL
vercel env add NEXT_PUBLIC_APP_URL
```

#### Netlify Deployment

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Build and deploy
npm run build
netlify deploy --prod --dir=.next
```

## Reverse Proxy Configuration

### Nginx Configuration

```nginx
upstream freeagentics_web {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name app.freeagentics.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name app.freeagentics.com;

    # SSL Configuration
    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Static Assets
    location /_next/static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Service Worker
    location /sw.js {
        expires 0;
        add_header Cache-Control "public, max-age=0, must-revalidate";
    }

    # Main Application
    location / {
        proxy_pass http://freeagentics_web;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Apache Configuration

```apache
<VirtualHost *:80>
    ServerName app.freeagentics.com
    Redirect permanent / https://app.freeagentics.com/
</VirtualHost>

<VirtualHost *:443>
    ServerName app.freeagentics.com

    SSLEngine on
    SSLCertificateFile /path/to/ssl/cert.pem
    SSLCertificateKeyFile /path/to/ssl/key.pem

    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:3000/
    ProxyPassReverse / http://127.0.0.1:3000/

    # Security Headers
    Header always set Strict-Transport-Security "max-age=31536000; includeSubDomains"
    Header always set X-Frame-Options "DENY"
    Header always set X-Content-Type-Options "nosniff"
    Header always set X-XSS-Protection "1; mode=block"

    # Compression
    LoadModule deflate_module modules/mod_deflate.so
    <Location />
        SetOutputFilter DEFLATE
        SetEnvIfNoCase Request_URI \
            \.(?:gif|jpe?g|png)$ no-gzip dont-vary
        SetEnvIfNoCase Request_URI \
            \.(?:exe|t?gz|zip|bz2|sit|rar)$ no-gzip dont-vary
    </Location>
</VirtualHost>
```

## Performance Optimization

### CDN Configuration

```bash
# Configure CDN for static assets
# Example: Cloudflare, AWS CloudFront, or similar

# Cache Rules:
# - /_next/static/*: Cache for 1 year
# - /static/*: Cache for 1 year
# - /sw.js: No cache
# - /api/*: No cache
# - /*: Cache for 1 hour with stale-while-revalidate
```

### Database Optimization

```sql
-- Add indexes for frequently queried data
CREATE INDEX idx_user_sessions ON user_sessions(user_id, created_at);
CREATE INDEX idx_agent_status ON agents(status, updated_at);
```

## Monitoring Setup

### Health Checks

```bash
# Application health check
curl -f http://localhost:3000/api/health

# Detailed health check
curl -f http://localhost:3000/api/health?detailed=true
```

### Logging Configuration

```javascript
// Production logging setup
const winston = require("winston");

const logger = winston.createLogger({
  level: "info",
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: "error.log", level: "error" }),
    new winston.transports.File({ filename: "combined.log" }),
  ],
});

if (process.env.NODE_ENV !== "production") {
  logger.add(
    new winston.transports.Console({
      format: winston.format.simple(),
    }),
  );
}
```

### Performance Monitoring

```javascript
// Real User Monitoring setup
import { getCLS, getFID, getFCP, getLCP, getTTFB } from "web-vitals";

function sendToAnalytics(metric) {
  // Send to your analytics service
  fetch("/api/analytics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metric),
  });
}

getCLS(sendToAnalytics);
getFID(sendToAnalytics);
getFCP(sendToAnalytics);
getLCP(sendToAnalytics);
getTTFB(sendToAnalytics);
```

## Security Considerations

### Environment Security

```bash
# Set proper file permissions
chmod 600 .env.production
chown www-data:www-data .env.production

# Use secrets management
# AWS Secrets Manager, HashiCorp Vault, or similar
```

### Content Security Policy

```javascript
// Strict CSP for production
const csp = {
  "default-src": ["'self'"],
  "script-src": ["'self'", "'unsafe-inline'"],
  "style-src": ["'self'", "'unsafe-inline'"],
  "img-src": ["'self'", "data:", "https:"],
  "font-src": ["'self'", "data:"],
  "connect-src": ["'self'", "https://api.freeagentics.com"],
  "frame-ancestors": ["'none'"],
  "base-uri": ["'self'"],
  "form-action": ["'self'"],
};
```

## Backup and Recovery

### Database Backups

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump freeagentics > /backups/freeagentics_$DATE.sql
aws s3 cp /backups/freeagentics_$DATE.sql s3://freeagentics-backups/
```

### Application Backups

```bash
# Static files backup
tar -czf /backups/static_files_$(date +%Y%m%d).tar.gz ./public/uploads/
```

## Scaling Considerations

### Horizontal Scaling

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: freeagentics-web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: freeagentics-web
  template:
    metadata:
      labels:
        app: freeagentics-web
    spec:
      containers:
        - name: web
          image: freeagentics-web:latest
          ports:
            - containerPort: 3000
          env:
            - name: NODE_ENV
              value: "production"
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
```

### Load Balancing

```nginx
upstream freeagentics_web {
    least_conn;
    server web1.freeagentics.com:3000;
    server web2.freeagentics.com:3000;
    server web3.freeagentics.com:3000;
}
```

## Troubleshooting

### Common Issues

1. **Build Failures**: Check Node.js version and dependencies
2. **Memory Issues**: Increase container memory limits
3. **SSL Issues**: Verify certificate installation
4. **Performance Issues**: Check bundle size and caching

### Debug Commands

```bash
# Check application logs
docker logs freeagentics-web

# Check system resources
docker stats freeagentics-web

# Test API connectivity
curl -I https://api.freeagentics.com/health

# Run performance audit
npm run lighthouse
```

## Maintenance

### Regular Tasks

- **Weekly**: Review application logs and metrics
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Performance audit and optimization review
- **Annually**: SSL certificate renewal and security audit

### Update Process

```bash
# 1. Backup current deployment
docker tag freeagentics-web:latest freeagentics-web:backup-$(date +%Y%m%d)

# 2. Build new version
docker build -f Dockerfile.production -t freeagentics-web:latest .

# 3. Test new version
docker run --rm -p 3001:3000 freeagentics-web:latest

# 4. Deploy new version
docker stop freeagentics-web
docker rm freeagentics-web
docker run -d --name freeagentics-web -p 3000:3000 freeagentics-web:latest

# 5. Verify deployment
curl -f http://localhost:3000/api/health
```

## Support and Resources

### Documentation

- [Next.js Production Deployment](https://nextjs.org/docs/deployment)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Nginx Configuration](https://nginx.org/en/docs/)

### Monitoring Tools

- **Application Performance**: New Relic, DataDog, or Sentry
- **Infrastructure**: Prometheus, Grafana, or CloudWatch
- **Uptime**: Pingdom, UptimeRobot, or StatusCake

### Emergency Contacts

- **Development Team**: dev@freeagentics.com
- **DevOps Team**: devops@freeagentics.com
- **On-call Support**: +1-xxx-xxx-xxxx

---

_Last Updated: $(date)_
_Version: 1.0_
_Next Review: 3 months_
