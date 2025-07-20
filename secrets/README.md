# FreeAgentics Secrets Management

This directory contains scripts and utilities for managing secrets in production deployments.

## Security Notice

**⚠️ IMPORTANT: Never commit actual secret values to version control!**

The files in this directory are templates and utilities. Actual secret values should be:

1. Generated using the provided scripts
1. Stored securely (environment variables, Docker secrets, vault, etc.)
1. Never committed to the repository

## Files

- `generate_secrets.py` - Script to generate secure random secrets
- `docker-secrets.yml` - Docker secrets configuration template
- `secrets-manager.py` - Production secrets management utility
- `vault-integration.py` - HashiCorp Vault integration (optional)
- `*.txt` - Template files (replace with actual secrets in production)

## Usage

### Development

```bash
# Generate development secrets
python generate_secrets.py --env development

# Load secrets from environment
source .env.development
```

### Production

```bash
# Generate production secrets
python generate_secrets.py --env production --vault

# Use Docker secrets
docker-compose -f docker-compose.production.yml up
```

### Vault Integration (Optional)

```bash
# Store secrets in Vault
python vault-integration.py store

# Retrieve secrets from Vault
python vault-integration.py retrieve
```

## Security Best Practices

1. **Rotation**: Rotate secrets regularly (recommended: every 90 days)
1. **Access Control**: Limit access to secrets to necessary services only
1. **Monitoring**: Monitor secret access and usage
1. **Backup**: Securely backup secrets with encryption
1. **Audit**: Maintain audit logs of secret operations

## Environment Variables

Required environment variables for production:

- `SECRET_KEY` - Application secret key
- `JWT_SECRET` - JWT signing secret
- `POSTGRES_PASSWORD` - Database password
- `REDIS_PASSWORD` - Redis password
- `ENCRYPTION_KEY` - Data encryption key
- `SSL_CERT_PATH` - SSL certificate path
- `SSL_KEY_PATH` - SSL private key path
