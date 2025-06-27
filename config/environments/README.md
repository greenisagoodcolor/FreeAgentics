# FreeAgentics Environment Configuration

This directory contains environment-specific configurations for different deployment scenarios.

## Overview

FreeAgentics uses environment variables to configure the application for different deployment contexts. The system supports multiple environments:

- **Development** - Local development with hot reload and debugging tools
- **Test** - Automated testing with mocked services
- **Demo** - Demonstration environment for showcasing features
- **Staging** - Pre-production testing environment
- **Production** - Live production deployment

## Environment Files

### Templates Available

- `env.example` - Complete template with all available variables
- `env.development` - Development configuration template
- `env.test` - Test environment template
- `env.production` - Production configuration template

### Creating Environment Files

1. Copy the appropriate template:

   ```bash
   cp env.development .env.development
   ```

2. Update the file with your specific values (API keys, database URLs, etc.)

3. Ensure the file is NOT committed to version control

## Configuration Categories

### 1. Application Configuration

- Frontend URLs and metadata
- Node environment settings
- Application versioning

### 2. Backend API Configuration

- Server host and port settings
- CORS configuration
- Request limits and timeouts
- Worker processes

### 3. Database Configuration

- PostgreSQL connection strings
- Connection pooling settings
- Migration and seeding options

### 4. Redis Configuration

- Cache server connection
- TTL and key prefix settings
- Connection pool configuration

### 5. LLM Configuration

- Provider selection (Anthropic, OpenAI, Ollama)
- API keys for each provider
- Model selection and fallbacks
- Token limits and parameters
- Context window management

### 6. Security Configuration

- Encryption keys and salts
- JWT settings and expiration
- Security headers
- Session management

### 7. Feature Flags

- Development tools (hot reload, debugging)
- System features (multi-agent, knowledge graph)
- Experimental features
- Demo mode settings

### 8. Resource Limits

- Agent simulation constraints
- Knowledge graph size limits
- Rate limiting configuration
- Memory and CPU limits

### 9. Monitoring and Logging

- Log levels and formats
- File logging configuration
- Metrics collection
- Error tracking (Sentry)

### 10. External Services

- AWS configuration
- Email service settings
- Third-party integrations

## Environment-Specific Notes

### Development

- Uses local Docker containers
- Hot reload enabled
- Verbose logging
- Mock services available
- No rate limiting

### Test

- Isolated test databases
- Minimal logging
- Mock LLM providers
- Deterministic settings
- Fast timeouts

### Production

- SSL/TLS required
- Secure secrets management
- Rate limiting enabled
- Error tracking active
- Optimized for performance

## Security Best Practices

1. **Secret Generation**

   ```bash
   # Generate secure random keys
   openssl rand -base64 32
   ```

2. **Key Rotation**
   - Rotate all secrets quarterly
   - Use different keys per environment
   - Never reuse development keys

3. **Access Control**
   - Limit access to production configs
   - Use secrets management services
   - Enable audit logging

4. **Storage**
   - Never commit .env files
   - Use encrypted storage
   - Implement proper backups

## Docker Integration

Environment files are referenced in Docker Compose:

```yaml
services:
  frontend:
    env_file:
      - ../../environments/.env.development
```

## Troubleshooting

### Common Issues

1. **Missing Variables**
   - Check all required variables are set
   - Reference env.example for complete list

2. **Connection Errors**
   - Verify database/Redis URLs
   - Check network configuration
   - Ensure services are running

3. **Permission Issues**
   - Set appropriate file permissions (600)
   - Verify Docker volume mounts

### Validation

Run environment validation:

```bash
# Check for missing required variables
grep -E "^[A-Z_]+=" env.example | while read var; do
  key=$(echo $var | cut -d= -f1)
  grep -q "^$key=" .env.development || echo "Missing: $key"
done
```

## References

- [Next.js Environment Variables](https://nextjs.org/docs/basic-features/environment-variables)
- [Docker Compose Environment](https://docs.docker.com/compose/environment-variables/)
- [12-Factor App Config](https://12factor.net/config)
