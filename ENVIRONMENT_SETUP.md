# FreeAgentics Environment Setup Guide

This guide provides detailed instructions for setting up your FreeAgentics development environment.

## Table of Contents
- [Quick Start](#quick-start)
- [Database Configuration](#database-configuration)
- [Environment Variables](#environment-variables)
- [Development Setup](#development-setup)
- [Production Setup](#production-setup)
- [Troubleshooting](#troubleshooting)

## Quick Start

The fastest way to get started with FreeAgentics:

```bash
# Clone the repository
git clone https://github.com/your-org/freeagentics.git
cd freeagentics

# Copy development environment template
cp .env.development .env

# Install dependencies and start
make install
make dev
```

This will start FreeAgentics with SQLite database (no PostgreSQL required).

## Database Configuration

FreeAgentics supports two database backends:

### SQLite (Default for Development)

SQLite is automatically used when:
- `DEVELOPMENT_MODE=true` in your `.env` file
- `DATABASE_URL` is not set or is empty

**Advantages:**
- No installation required
- Zero configuration
- Perfect for local development
- Automatic database creation

**Limitations:**
- Single-user access only
- Not suitable for production
- Limited concurrent operations
- No advanced PostgreSQL features

### PostgreSQL (Recommended for Production)

For production or multi-agent testing, use PostgreSQL:

1. **Install PostgreSQL** (if not already installed):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib

   # macOS
   brew install postgresql
   ```

2. **Create database and user**:
   ```bash
   sudo -u postgres psql
   CREATE DATABASE freeagentics_dev;
   CREATE USER freeagentics WITH PASSWORD 'your-secure-password';
   GRANT ALL PRIVILEGES ON DATABASE freeagentics_dev TO freeagentics;
   \q
   ```

3. **Configure in .env**:
   ```bash
   DATABASE_URL=postgresql://freeagentics:your-secure-password@localhost:5432/freeagentics_dev
   ```

## Environment Variables

### Essential Variables

```bash
# Core Configuration
DEVELOPMENT_MODE=true              # Enable development features and SQLite fallback
NODE_ENV=development              # Node environment (development/production)

# Database
DATABASE_URL=                     # PostgreSQL URL (leave empty for SQLite)
                                 # Format: postgresql://user:pass@host:port/dbname

# Security (MUST change in production!)
SECRET_KEY=dev-secret-key        # Session encryption key
JWT_SECRET=dev-jwt-secret        # JWT signing key

# API Configuration
API_HOST=0.0.0.0                 # API bind address
API_PORT=8000                    # API port
NEXT_PUBLIC_API_URL=http://localhost:8000  # Frontend API endpoint
```

### Optional Variables

```bash
# Redis (for caching and real-time features)
REDIS_URL=redis://localhost:6379/0

# Debugging
DEBUG=true                       # Enable debug mode
DEBUG_SQL=false                  # Log all SQL queries (verbose!)
LOG_LEVEL=DEBUG                  # Log level: DEBUG, INFO, WARNING, ERROR

# Testing
TESTING=false                    # Automatically set during tests

# Docker
COMPOSE_PROJECT_NAME=freeagentics-dev
```

### Environment File Priority

1. `.env` - Your local configuration (git-ignored)
2. `.env.development` - Development defaults (committed)
3. `.env.production` - Production defaults (committed)
4. Environment variables - Override any file setting

## Development Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git
- Make (GNU Make)

### Step-by-Step Setup

1. **Clone and Enter Directory**:
   ```bash
   git clone https://github.com/your-org/freeagentics.git
   cd freeagentics
   ```

2. **Create Environment File**:
   ```bash
   cp .env.development .env
   ```

3. **Install Dependencies**:
   ```bash
   make install
   ```
   This installs both Python and Node.js dependencies.

4. **Start Development Servers**:
   ```bash
   make dev
   ```
   - Backend API: http://localhost:8000
   - Frontend UI: http://localhost:3000
   - API Docs: http://localhost:8000/docs

### Verifying Your Setup

```bash
# Check if services are running
make status

# Run tests to verify installation
make test

# Try the Active Inference demo
make demo
```

## Production Setup

### Security First

1. **Generate Secure Keys**:
   ```bash
   # Generate SECRET_KEY
   openssl rand -hex 32

   # Generate JWT_SECRET
   openssl rand -hex 32
   ```

2. **Create Production .env**:
   ```bash
   # Required settings
   DEVELOPMENT_MODE=false
   NODE_ENV=production
   DATABASE_URL=postgresql://user:pass@host:port/dbname
   SECRET_KEY=<your-generated-key>
   JWT_SECRET=<your-generated-jwt-key>

   # Recommended
   REDIS_URL=redis://user:pass@host:6379/0
   DEBUG=false
   LOG_LEVEL=WARNING
   ```

3. **Database Setup**:
   - Use managed PostgreSQL service (AWS RDS, Google Cloud SQL, etc.)
   - Enable SSL/TLS connections
   - Configure automated backups

### Docker Deployment

```bash
# Build production image
make docker

# Run with docker-compose
docker-compose -f docker-compose.production.yml up -d
```

## Troubleshooting

### Database Connection Issues

**SQLite Issues:**
- **"Database is locked"**: Close all connections and restart
- **"No such table"**: Delete `freeagentics_dev.db` and restart
- **Permission errors**: Check file ownership of `freeagentics_dev.db`

**PostgreSQL Issues:**
- **"Connection refused"**: Check if PostgreSQL is running
- **"Authentication failed"**: Verify username/password in DATABASE_URL
- **"Database does not exist"**: Create database with `createdb` command

### Environment Variable Issues

- **Variables not loading**: Ensure `.env` file is in project root
- **Wrong values used**: Check for typos and proper quoting
- **Production using dev values**: Verify `DEVELOPMENT_MODE=false`

### Quick Fixes

```bash
# Reset everything
make reset

# Kill hanging processes
make kill-ports

# Clean and reinstall
make clean && make install

# Verbose test output
make test --tb=long --vvv
```

### Getting Help

1. Check logs in `api.log` and `frontend.log`
2. Enable `DEBUG=true` and `DEBUG_SQL=true` for detailed output
3. Run health check: `curl http://localhost:8000/health`
4. Review documentation in `/docs` directory

## Next Steps

- Run the demo: `make demo`
- Explore the API: http://localhost:8000/docs
- Read the architecture guide: `/docs/ARCHITECTURE_OVERVIEW.md`
- Join the community: [Active Inference Institute](https://www.activeinference.org/)
