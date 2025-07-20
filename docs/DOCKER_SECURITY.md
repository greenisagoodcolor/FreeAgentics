# Docker Security Configuration Guide

## Overview

FreeAgentics v0.2 implements strict security measures for Docker deployments. All hardcoded credentials have been removed and must be provided via environment variables.

## Quick Start for Development

1. **Copy the override template**:

   ```bash
   cp docker-compose.override.yml.example docker-compose.override.yml
   ```

1. **Edit docker-compose.override.yml** with your local passwords:

   ```yaml
   services:
     postgres:
       environment:
         POSTGRES_PASSWORD: your_secure_password_here
   ```

1. **Set required environment variables** in `.env`:

   ```bash
   # Database
   POSTGRES_PASSWORD=your_secure_password_here
   DATABASE_URL=postgresql://freeagentics:your_secure_password_here@localhost:5432/freeagentics

   # Redis
   REDIS_PASSWORD=your_redis_password_here

   # Security Keys (generate new ones!)
   SECRET_KEY=your_generated_secret_key_here
   JWT_SECRET=your_generated_jwt_secret_here
   ```

1. **Generate secure keys**:

   ```bash
   # Generate SECRET_KEY
   python -c "import secrets; print(secrets.token_urlsafe(64))"

   # Generate JWT_SECRET
   python -c "import secrets; print(secrets.token_urlsafe(64))"
   ```

## Security Changes in v0.2

### 1. No Hardcoded Credentials

- ❌ **Removed**: Default passwords in `docker-compose.yml`
- ✅ **Added**: Required environment variables with no fallbacks
- ✅ **Added**: Validation to prevent dev credentials in production

### 2. Database Security

- **Required**: `DATABASE_URL` environment variable
- **SSL/TLS**: Automatically enabled in production mode
- **Connection Security**: Statement timeouts, connection recycling
- **Validation**: Rejects development credentials in production

### 3. Redis Security

- **Password Protection**: Redis now requires authentication
- **Network Isolation**: Custom Docker network for service communication

### 4. Container Security

- **Non-root Users**: Production containers run as non-root
- **Read-only Filesystem**: Production containers use read-only root
- **Network Isolation**: Custom bridge network with defined subnet

## Production Deployment

### Required Environment Variables

```bash
# Database (no defaults allowed)
POSTGRES_PASSWORD=<strong-password>
DATABASE_URL=postgresql://freeagentics:<strong-password>@postgres:5432/freeagentics?sslmode=require

# Redis (no defaults allowed)
REDIS_PASSWORD=<strong-redis-password>

# Security Keys (MUST generate new ones)
SECRET_KEY=<generate-with-secrets.token_urlsafe(64)>
JWT_SECRET=<generate-with-secrets.token_urlsafe(64)>

# Production Flag
PRODUCTION=true

# API Configuration
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

### SSL/TLS Configuration

1. **Generate SSL certificates**:

   ```bash
   mkdir -p nginx/ssl
   # Add your SSL certificates:
   # - nginx/ssl/cert.pem
   # - nginx/ssl/key.pem
   ```

1. **Generate DH parameters**:

   ```bash
   openssl dhparam -out nginx/dhparam.pem 2048
   ```

### Deployment Commands

```bash
# Production deployment with all security checks
docker-compose --profile prod up -d

# Run database migrations
docker-compose --profile migrate up

# View logs
docker-compose logs -f backend
```

## Security Checklist

Before deploying to production:

- [ ] Generated new SECRET_KEY (64+ characters)
- [ ] Generated new JWT_SECRET (64+ characters)
- [ ] Set strong POSTGRES_PASSWORD (20+ characters)
- [ ] Set strong REDIS_PASSWORD (20+ characters)
- [ ] Configured SSL certificates in nginx/ssl/
- [ ] Set PRODUCTION=true in environment
- [ ] Removed all .env files from version control
- [ ] Tested with `make security-check`
- [ ] Ran `make security-audit`
- [ ] Verified no hardcoded secrets with `make check-secrets`

## Common Issues

### "DATABASE_URL environment variable is required"

- **Solution**: Set DATABASE_URL in your .env file or environment

### "POSTGRES_PASSWORD is required"

- **Solution**: Set POSTGRES_PASSWORD when running docker-compose

### "Production environment detected but using development database credentials"

- **Solution**: Change your database password from the development default

### Container fails with "Permission denied"

- **Solution**: Ensure your application can run as non-root user (UID 1000)

## Security Best Practices

1. **Rotate Credentials Regularly**

   - Database passwords every 90 days
   - JWT secrets every 180 days
   - API keys every 90 days

1. **Monitor Access**

   - Enable PostgreSQL logging
   - Monitor failed authentication attempts
   - Set up alerts for suspicious activity

1. **Backup Securely**

   - Encrypt database backups
   - Store backups in separate location
   - Test restore procedures regularly

1. **Network Security**

   - Use firewall rules to restrict access
   - Enable VPN for administrative access
   - Monitor network traffic

## Additional Resources

- [SECURITY_AUDIT_REPORT.md](../SECURITY_AUDIT_REPORT.md) - Full security audit details
- [.env.production.template](../.env.production.template) - Production environment template
- `make security-check` - Verify security configuration
- `make prod-env` - Production deployment checklist
