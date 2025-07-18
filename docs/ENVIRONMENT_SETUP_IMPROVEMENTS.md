# Environment Setup Improvements

## Overview

This document summarizes the recent improvements to the FreeAgentics development environment setup, making it easier for new developers to get started.

## Key Improvements

### 1. SQLite Fallback for Development

**What's New**: The system now automatically uses SQLite when PostgreSQL is not available in development mode.

**How It Works**:
- When `DEVELOPMENT_MODE=true` and no `DATABASE_URL` is set
- Automatically creates `freeagentics_dev.db` in the project root
- No PostgreSQL installation required for basic development

**Implementation**: See `/database/session.py` for the fallback logic

### 2. Development Environment File

**What's New**: Added `.env.development` with sensible defaults for local development.

**Features**:
- Pre-configured for SQLite fallback
- Development-only secrets (not for production!)
- Debug settings enabled by default
- Optional Redis configuration

**Usage**:
```bash
cp .env.development .env
```

### 3. Simplified Installation Process

**What's New**: One-command installation for all dependencies.

**Command**: `make install`

**What It Does**:
1. Creates Python virtual environment
2. Upgrades pip
3. Installs Python dependencies from `requirements.txt`
4. Installs Node.js dependencies for frontend
5. Handles both pyproject.toml and requirements.txt

### 4. Improved Documentation

**Updated Files**:
- `README.md` - Added SQLite fallback and environment setup sections
- `docs/ONBOARDING_GUIDE.md` - Updated development setup instructions
- `docs/DOCKER_SETUP_GUIDE.md` - Added .env.development reference

## Migration Guide

### For Existing Developers

1. Pull latest changes
2. Copy new environment file: `cp .env.development .env`
3. Run `make install` to ensure all dependencies are updated
4. Start development: `make dev`

### For New Developers

1. Clone repository
2. Copy environment file: `cp .env.development .env`
3. Install dependencies: `make install`
4. Start development: `make dev`

## Database Options

### Option 1: SQLite (Default)
- No setup required
- Automatic with `DEVELOPMENT_MODE=true`
- Perfect for quick prototyping
- Limitations: Not suitable for multi-agent testing or production

### Option 2: PostgreSQL
- Set `DATABASE_URL` in `.env`
- Full feature support
- Required for production-like testing
- Supports concurrent agents

## Environment Variables

### Critical Development Variables

```bash
# Enable development features
DEVELOPMENT_MODE=true

# Database (leave empty for SQLite)
DATABASE_URL=

# Security (development values)
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET=dev-jwt-secret-change-in-production

# Optional services
REDIS_URL=redis://localhost:6379/0

# Debug settings
DEBUG=true
DEBUG_SQL=false
LOG_LEVEL=DEBUG
```

## Troubleshooting

### Common Issues

1. **"DATABASE_URL environment variable is required"**
   - Solution: Ensure `DEVELOPMENT_MODE=true` in `.env`
   - Or: Set a valid PostgreSQL URL

2. **"Module 'inferactively-pymdp' not found"**
   - Solution: Run `make clean && make install`
   - The package is `inferactively-pymdp==0.0.7.1`

3. **"Port already in use"**
   - Solution: Run `make kill-ports`
   - Or: Change ports in `.env`

### Debug Tools

- SQL Query Logging: Set `DEBUG_SQL=true`
- Verbose Logging: Set `LOG_LEVEL=DEBUG`
- Test Verbosity: `make test --tb=long --vvv`

## Security Considerations

### Development vs Production

**Development** (`.env.development`):
- Uses hardcoded development secrets
- SQLite database without authentication
- Debug mode enabled
- CORS allows all origins

**Production**:
- Must use secure random secrets
- PostgreSQL with SSL/TLS required
- Debug mode disabled
- CORS restricted to allowed domains

### Never In Production

- Don't use `.env.development` values
- Don't use SQLite database
- Don't enable `DEVELOPMENT_MODE`
- Don't use default secrets

## Performance Notes

### SQLite Limitations
- Single writer at a time
- No true concurrent access
- Limited to single machine
- Suitable for development only

### PostgreSQL Benefits
- Full ACID compliance
- Concurrent multi-agent support
- Production-grade performance
- Horizontal scalability

## Future Improvements

1. **Docker Development Mode**
   - SQLite support in Docker containers
   - Simplified docker-compose for development

2. **Database Migrations**
   - Automatic migration for SQLite
   - Development seed data

3. **Environment Validation**
   - Pre-flight checks for configuration
   - Automatic secret generation for development

## References

- [Database Session Configuration](/database/session.py)
- [Environment Template](/.env.development)
- [Installation Script](/Makefile)
- [Main README](/README.md)