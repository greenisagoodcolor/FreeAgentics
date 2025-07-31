# PostgreSQL and pgvector Setup Guide

This guide provides comprehensive instructions for setting up PostgreSQL with pgvector extension for FreeAgentics.

## Quick Start (Recommended: Docker)

```bash
# Using Docker Compose (easiest)
docker-compose up -d postgres

# Or using the setup script
./scripts/setup-db-docker.sh
```

## Option 1: Docker Setup (Recommended)

### Prerequisites

- Docker and Docker Compose installed
- 2GB free disk space

### Using Docker Compose

Create or use the existing `docker-compose.yml`:

```yaml
version: "3.8"

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: freeagentics-postgres
    environment:
      POSTGRES_USER: freeagentics
      POSTGRES_PASSWORD: freeagentics_dev
      POSTGRES_DB: freeagentics
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U freeagentics"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

### Quick Commands

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Stop PostgreSQL
docker-compose stop postgres

# View logs
docker-compose logs -f postgres

# Connect to database
docker-compose exec postgres psql -U freeagentics

# Remove all data (careful!)
docker-compose down -v
```

## Option 2: Local Installation

### macOS

```bash
# Install PostgreSQL 16
brew install postgresql@16

# Install pgvector
brew install pgvector

# Start PostgreSQL
brew services start postgresql@16

# Create database and user
createdb freeagentics
psql -d freeagentics -c "CREATE EXTENSION vector;"
```

### Ubuntu/Debian

```bash
# Add PostgreSQL APT repository
sudo sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update

# Install PostgreSQL 16
sudo apt-get install postgresql-16 postgresql-client-16

# Install pgvector
sudo apt install postgresql-16-pgvector

# Enable extension
sudo -u postgres psql -d freeagentics -c "CREATE EXTENSION vector;"
```

### Windows

1. Download PostgreSQL 16 installer from https://www.postgresql.org/download/windows/
2. Run installer and follow wizard
3. Install pgvector:
   ```powershell
   # In PowerShell as Administrator
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   nmake /f Makefile.win
   nmake /f Makefile.win install
   ```

## pgvector Extension Setup

### What is pgvector?

pgvector adds vector similarity search to PostgreSQL, enabling:

- Semantic search capabilities
- Efficient k-nearest neighbor queries
- Embedding storage and retrieval
- Cosine similarity, L2 distance, and inner product operations

### Enable pgvector

```sql
-- Connect to your database
psql -U freeagentics -d freeagentics

-- Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Version Compatibility

| PostgreSQL | pgvector | Status           |
| ---------- | -------- | ---------------- |
| 16.x       | 0.5.x+   | ✅ Recommended   |
| 15.x       | 0.5.x+   | ✅ Supported     |
| 14.x       | 0.4.x+   | ⚠️ Minimum       |
| < 14       | -        | ❌ Not supported |

## Database Configuration

### Connection URLs

Add to your `.env` file:

```bash
# Docker setup
DATABASE_URL=postgresql://freeagentics:freeagentics_dev@localhost:5432/freeagentics

# Local setup (adjust as needed)
DATABASE_URL=postgresql://username:password@localhost:5432/freeagentics

# With SSL (production)
DATABASE_URL=postgresql://user:pass@host:5432/freeagentics?sslmode=require
```

### Verify Connection

```bash
# Test connection
psql $DATABASE_URL -c "SELECT version();"

# Verify pgvector
psql $DATABASE_URL -c "SELECT vector_version();"
```

## Running Migrations

```bash
# Using Alembic
alembic upgrade head

# Or using make
make db-migrate

# Verify migrations
alembic current
```

## Sample Vector Queries

Test pgvector functionality:

```sql
-- Create a test table
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    embedding vector(3)
);

-- Insert test data
INSERT INTO items (embedding) VALUES
    ('[1,2,3]'),
    ('[4,5,6]'),
    ('[7,8,9]');

-- Find nearest neighbors
SELECT id, embedding <-> '[3,4,5]' AS distance
FROM items
ORDER BY embedding <-> '[3,4,5]'
LIMIT 2;

-- Cosine similarity
SELECT id, 1 - (embedding <=> '[3,4,5]') AS similarity
FROM items
ORDER BY embedding <=> '[3,4,5]'
LIMIT 2;
```

## Performance Tuning

### PostgreSQL Configuration

Add to `postgresql.conf`:

```ini
# Memory settings
shared_buffers = 256MB
work_mem = 4MB
maintenance_work_mem = 64MB

# Connection settings
max_connections = 100

# pgvector specific
ivfflat.probes = 10
```

### Index Creation

```sql
-- Create index for vector similarity search
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- For L2 distance
CREATE INDEX ON embeddings USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
```

## Switching Between Docker and Local

### Export from Docker

```bash
# Backup Docker database
docker-compose exec postgres pg_dump -U freeagentics freeagentics > backup.sql
```

### Import to Local

```bash
# Restore to local PostgreSQL
psql -U your_user -d freeagentics < backup.sql
```

### Update .env

```bash
# Switch DATABASE_URL between:
# Docker: postgresql://freeagentics:freeagentics_dev@localhost:5432/freeagentics
# Local: postgresql://your_user:your_pass@localhost:5432/freeagentics
```

## Troubleshooting

### Quick Diagnosis Commands

```bash
# Database health check suite
make db-check                               # Test connection
docker-compose -f docker-compose.db.yml ps  # Check container status
psql $DATABASE_URL -c "SELECT version();"   # PostgreSQL version
psql $DATABASE_URL -c "\dx"                 # List extensions
alembic current                             # Check migration status
```

### Common Issues and Solutions

#### 1. Connection Issues

**Symptom**: Connection refused, timeout, or "server doesn't exist"

```bash
# Diagnostic steps
docker-compose -f docker-compose.db.yml ps  # Check if container running
docker-compose -f docker-compose.db.yml logs postgres  # Check container logs
netstat -an | grep 5432                     # Check if port is bound
ping localhost                              # Basic network test

# Solutions by scenario:
```

**Container not running:**

```bash
# Start PostgreSQL
docker-compose -f docker-compose.db.yml up -d

# Check initialization logs
docker-compose -f docker-compose.db.yml logs -f postgres
```

**Port conflict (5432 already in use):**

```bash
# Find what's using port 5432
sudo lsof -i :5432
# Kill conflicting process or change port in docker-compose.db.yml
```

**Network/firewall issues:**

```bash
# Test with host networking
docker run --rm --network host pgvector/pgvector:pg16 \
  psql -h localhost -U freeagentics -d freeagentics -c "SELECT 1;"

# Or modify docker-compose.db.yml to use host network
```

#### 2. pgvector Extension Issues

**Symptom**: "extension 'vector' does not exist" or vector functions not working

```bash
# Diagnostic steps
psql $DATABASE_URL -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';"
psql $DATABASE_URL -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
psql $DATABASE_URL -c "SELECT vector_version();"  # Should work if installed
```

**Extension not available:**

```bash
# For Docker setup - rebuild with correct image
docker-compose -f docker-compose.db.yml down -v
docker-compose -f docker-compose.db.yml pull
docker-compose -f docker-compose.db.yml up -d

# For local setup - reinstall pgvector
# macOS: brew reinstall pgvector
# Ubuntu: sudo apt install postgresql-16-pgvector
```

**Extension not enabled in database:**

```sql
-- Connect as superuser and enable
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"

-- Verify installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Test basic functionality
SELECT '[1,2,3]'::vector;
```

#### 3. Permission and Authentication Issues

**Symptom**: "permission denied", "authentication failed", "role does not exist"

```bash
# Check user permissions
psql $DATABASE_URL -c "\du"  # List users and roles
psql $DATABASE_URL -c "\l"   # List databases and owners
```

**Authentication failures:**

```bash
# Reset password in Docker setup
docker-compose -f docker-compose.db.yml exec postgres \
  psql -U postgres -c "ALTER USER freeagentics PASSWORD 'freeagentics_dev';"

# For local setup - ensure user exists
sudo -u postgres createuser --interactive freeagentics
sudo -u postgres psql -c "ALTER USER freeagentics PASSWORD 'your_password';"
```

**Permission denied on database/tables:**

```sql
-- Grant database access
GRANT ALL PRIVILEGES ON DATABASE freeagentics TO freeagentics;

-- Grant schema access
GRANT ALL ON SCHEMA public TO freeagentics;
GRANT ALL ON ALL TABLES IN SCHEMA public TO freeagentics;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO freeagentics;

-- For future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO freeagentics;
```

#### 4. Migration Issues

**Symptom**: "relation does not exist", migration failures, schema conflicts

```bash
# Check migration status
alembic current
alembic history

# View pending migrations
alembic show head
alembic show current
```

**Schema out of sync:**

```bash
# Mark current state as baseline (DESTRUCTIVE - use carefully)
alembic stamp head

# Or reset and re-run migrations
alembic downgrade base
alembic upgrade head
```

**Migration conflicts:**

```bash
# Generate new migration from current model state
alembic revision --autogenerate -m "sync_schema"

# Manually resolve conflicts in generated migration file
# Then run: alembic upgrade head
```

#### 5. Docker-Specific Issues

**Symptom**: Container crashes, initialization failures, volume issues

```bash
# Full diagnostic
docker-compose -f docker-compose.db.yml logs postgres
docker inspect freeagentics-postgres
docker volume ls | grep postgres
```

**Initialization failures:**

```bash
# Clean restart (DESTRUCTIVE - removes all data)
docker-compose -f docker-compose.db.yml down -v
docker volume rm freeagentics_postgres_data
docker-compose -f docker-compose.db.yml up -d

# Watch initialization
docker-compose -f docker-compose.db.yml logs -f postgres
```

**Volume permission issues:**

```bash
# Fix volume ownership
docker-compose -f docker-compose.db.yml exec postgres chown -R postgres:postgres /var/lib/postgresql/data

# Or recreate volume with correct permissions
docker-compose -f docker-compose.db.yml down
docker volume rm freeagentics_postgres_data
docker-compose -f docker-compose.db.yml up -d
```

#### 6. Performance Issues

**Symptom**: Slow queries, high memory usage, connection exhaustion

```bash
# Connection diagnostics
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;"
psql $DATABASE_URL -c "SELECT state, count(*) FROM pg_stat_activity GROUP BY state;"

# Memory and performance
psql $DATABASE_URL -c "SELECT * FROM pg_stat_database WHERE datname = 'freeagentics';"
```

**Too many connections:**

```sql
-- Check connection limit
SHOW max_connections;

-- View current connections
SELECT count(*) FROM pg_stat_activity;

-- Kill idle connections (if needed)
SELECT pg_terminate_backend(pid) FROM pg_stat_activity
WHERE state = 'idle' AND state_change < now() - interval '1 hour';
```

**Slow vector operations:**

```sql
-- Check if vector indexes exist
\d+ embeddings

-- Create vector index if missing
CREATE INDEX CONCURRENTLY ON embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Update statistics
ANALYZE embeddings;
```

### Environment-Specific Troubleshooting

#### Windows Issues

**Docker Desktop not starting:**

- Enable WSL2 integration
- Ensure Hyper-V is enabled
- Check Windows Subsystem for Linux is installed

**psql command not found:**

```powershell
# Install psql via chocolatey
choco install postgresql

# Or use Docker
docker run --rm -it postgres:16 psql postgresql://user:pass@host:port/db
```

#### macOS Issues

**Homebrew PostgreSQL conflicts:**

```bash
# Stop system PostgreSQL if running
brew services stop postgresql@16

# Use only Docker for consistency
docker-compose -f docker-compose.db.yml up -d
```

**M1/ARM compatibility:**

```bash
# Use ARM-compatible image
docker pull --platform linux/arm64 pgvector/pgvector:pg16
```

#### Linux Issues

**Systemd PostgreSQL conflicts:**

```bash
# Stop system PostgreSQL
sudo systemctl stop postgresql
sudo systemctl disable postgresql

# Ensure Docker has priority on port 5432
```

**SELinux restrictions:**

```bash
# Temporarily disable SELinux
sudo setenforce 0

# Or configure SELinux for Docker volumes
sudo setsebool -P container_manage_cgroup on
```

### Recovery Procedures

#### Complete Database Reset

**DESTRUCTIVE - removes all data:**

```bash
# Stop services
make stop

# Remove all database data
docker-compose -f docker-compose.db.yml down -v
docker volume rm freeagentics_postgres_data

# Fresh start
docker-compose -f docker-compose.db.yml up -d
sleep 10  # Wait for initialization
make db-migrate

# Verify setup
make db-check
```

#### Backup Before Troubleshooting

```bash
# Create backup before making changes
timestamp=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL > backup_${timestamp}.sql

# Or compressed backup
pg_dump -Fc $DATABASE_URL > backup_${timestamp}.dump
```

#### Database Recovery from Backup

```bash
# Restore from SQL backup
psql $DATABASE_URL < backup_20250731_143000.sql

# Restore from compressed backup
pg_restore -d $DATABASE_URL backup_20250731_143000.dump

# Verify restoration
psql $DATABASE_URL -c "SELECT count(*) FROM agents;"
```

### Automated Troubleshooting Script

Create `scripts/db-troubleshoot.sh`:

```bash
#!/bin/bash
# Comprehensive database diagnostics

echo "=== FreeAgentics Database Diagnostics ==="

# Test 1: Container status
echo "1. Container Status:"
docker-compose -f docker-compose.db.yml ps

# Test 2: Connection test
echo "2. Connection Test:"
psql $DATABASE_URL -c "SELECT 'Connection OK';" 2>/dev/null || echo "❌ Connection failed"

# Test 3: pgvector availability
echo "3. pgvector Extension:"
psql $DATABASE_URL -c "SELECT vector_version();" 2>/dev/null || echo "❌ pgvector not available"

# Test 4: Migration status
echo "4. Migration Status:"
alembic current 2>/dev/null || echo "❌ Alembic not configured"

# Test 5: Table existence
echo "5. Schema Status:"
psql $DATABASE_URL -c "\dt" 2>/dev/null | grep -q agents && echo "✅ Tables exist" || echo "❌ Tables missing"

# Test 6: Performance check
echo "6. Performance Check:"
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null

echo ""
echo "Run with: bash scripts/db-troubleshoot.sh"
```

### Debug Commands

```bash
# Comprehensive diagnostics
docker-compose -f docker-compose.db.yml logs postgres  # Container logs
docker inspect freeagentics-postgres                   # Container details
psql $DATABASE_URL -c "\conninfo"                      # Connection info
psql $DATABASE_URL -c "\dx"                            # Installed extensions
psql $DATABASE_URL -c "\dt"                            # List tables
psql $DATABASE_URL -c "\du"                            # List users
psql $DATABASE_URL -c "SHOW all;"                      # All settings

# Performance diagnostics
psql $DATABASE_URL -c "SELECT * FROM pg_stat_activity;"
psql $DATABASE_URL -c "SELECT * FROM pg_stat_database WHERE datname = 'freeagentics';"
psql $DATABASE_URL -c "SELECT schemaname,tablename,n_tup_ins,n_tup_upd,n_tup_del FROM pg_stat_user_tables;"

# Vector-specific diagnostics
psql $DATABASE_URL -c "SELECT count(*) FROM pg_available_extensions WHERE name = 'vector';"
psql $DATABASE_URL -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Disk space check
docker exec freeagentics-postgres df -h
docker system df
```

## Migration Procedures

### SQLite Demo to PostgreSQL Production

This section covers migrating from SQLite in-memory/file mode to PostgreSQL with pgvector.

#### 1. Pre-Migration Assessment

```bash
# Check current setup
echo "Current DATABASE_URL: ${DATABASE_URL:-'Not set (using SQLite demo)'}"
echo "Environment: ${ENVIRONMENT:-'dev'}"

# Check if data exists in SQLite (if using file mode)
if [ -f "freeagentics.db" ]; then
    echo "SQLite database file exists"
    sqlite3 freeagentics.db ".tables"
else
    echo "Using in-memory SQLite (no persistent data)"
fi

# Check Python dependencies
python -c "import psycopg2, pgvector" 2>/dev/null && echo "✅ PostgreSQL deps installed" || echo "❌ Install: pip install psycopg2-binary pgvector"
```

#### 2. PostgreSQL Setup

```bash
# Step 1: Set up PostgreSQL with pgvector
make db-setup-docker  # or make db-setup-local

# Step 2: Wait for database to be ready
sleep 10

# Step 3: Test connection
make db-check
```

#### 3. Data Migration (if preserving existing data)

**Option A: Fresh Start (Recommended for demo -> production)**

```bash
# Update .env to use PostgreSQL
cp .env .env.backup
sed -i 's/DATABASE_URL=.*/DATABASE_URL=postgresql:\/\/freeagentics:freeagentics_dev@localhost:5432\/freeagentics/' .env

# Run migrations to create schema
make db-migrate

# Verify schema
psql $DATABASE_URL -c "\dt"
```

**Option B: Preserve Existing SQLite Data**

```bash
# 1. Export data from SQLite (if using file mode)
sqlite3 freeagentics.db ".output agents_export.csv" ".mode csv" "SELECT * FROM agents;"
sqlite3 freeagentics.db ".output coalitions_export.csv" ".mode csv" "SELECT * FROM coalitions;"
# Export other tables as needed

# 2. Set up PostgreSQL schema
export DATABASE_URL="postgresql://freeagentics:freeagentics_dev@localhost:5432/freeagentics"
make db-migrate

# 3. Import data to PostgreSQL (manual process - adapt as needed)
# Create import script: scripts/import_from_sqlite.py
python scripts/import_from_sqlite.py

# 4. Verify migration
psql $DATABASE_URL -c "SELECT count(*) FROM agents;"
```

#### 4. Environment Configuration Update

```bash
# Update .env file
cat >> .env << 'EOF'

# PostgreSQL Configuration (updated from SQLite demo)
DATABASE_URL=postgresql://freeagentics:freeagentics_dev@localhost:5432/freeagentics
ENVIRONMENT=development
DEVELOPMENT_MODE=false

# Enable pgvector features
ENABLE_PGVECTOR=true
ENABLE_PGVECTOR_SIMILARITY=true
VECTOR_DIMENSION=384

# Disable demo mode features
ENABLE_MOCK_DATA=false
LLM_PROVIDER=openai  # or your preferred provider
EOF

# Remove SQLite-specific settings
sed -i '/SQLITE_FILE=/d' .env
```

#### 5. Application Configuration Update

Update any application code that might have SQLite-specific logic:

```python
# Before (SQLite demo mode)
if not DATABASE_URL:
    engine = create_engine("sqlite:///:memory:", echo=True)

# After (PostgreSQL production)
if not DATABASE_URL:
    raise ValueError("DATABASE_URL required for production mode")
engine = create_engine(DATABASE_URL, echo=False)
```

#### 6. Migration Validation

```bash
# Test the full application stack
make install
make dev

# Verify database functionality
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/agents

# Test vector operations (if applicable)
psql $DATABASE_URL -c "SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector AS distance;"
```

#### 7. Production Hardening

```bash
# Update production settings in .env
cat >> .env << 'EOF'

# Production Security Settings
PRODUCTION=true
SECURE_COOKIES=true
HTTPS_ONLY=true
CSRF_ENABLED=true
RATE_LIMITING_ENABLED=true

# Database Production Settings
DB_POOL_SIZE=20
DB_POOL_OVERFLOW=40
DB_POOL_TIMEOUT=30
DB_POOL_PRE_PING=true

# Change default passwords
POSTGRES_PASSWORD=your_strong_production_password
JWT_SECRET_KEY=your_production_jwt_secret
SECRET_KEY=your_production_secret_key
EOF

# Update database password
docker-compose -f docker-compose.db.yml exec postgres \
  psql -U postgres -c "ALTER USER freeagentics PASSWORD 'your_strong_production_password';"

# Update DATABASE_URL with new password
export DATABASE_URL="postgresql://freeagentics:your_strong_production_password@localhost:5432/freeagentics"
```

### Development to Production Migration

For moving from development PostgreSQL to production PostgreSQL:

#### 1. Database Dump and Restore

```bash
# Create development backup
pg_dump $DEV_DATABASE_URL > dev_backup.sql

# Restore to production (after setting up production database)
psql $PROD_DATABASE_URL < dev_backup.sql

# Or use compressed format for large databases
pg_dump -Fc $DEV_DATABASE_URL > dev_backup.dump
pg_restore -d $PROD_DATABASE_URL dev_backup.dump
```

#### 2. Schema Migration

```bash
# Generate migration for any schema differences
alembic revision --autogenerate -m "production_migration"

# Apply to production
alembic -c production_alembic.ini upgrade head
```

#### 3. Production Database Configuration

```bash
# Production postgresql.conf optimizations
cat >> postgresql.conf << 'EOF'
# Memory Configuration
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Connection Configuration
max_connections = 100
shared_preload_libraries = 'vector'

# Performance Configuration
random_page_cost = 1.1
effective_io_concurrency = 200
min_wal_size = 1GB
max_wal_size = 4GB

# Logging for Production
log_destination = 'csvlog'
logging_collector = on
log_directory = 'pg_log'
log_statement = 'ddl'
log_min_duration_statement = 1000

# pgvector Configuration
ivfflat.probes = 10
EOF

# Restart PostgreSQL to apply configuration
docker-compose -f docker-compose.db.yml restart postgres
```

### Zero-Downtime Migration

For production systems requiring zero downtime:

#### 1. Blue-Green Database Setup

```bash
# Set up new database (green)
docker run -d --name freeagentics-postgres-green \
  -e POSTGRES_DB=freeagentics \
  -e POSTGRES_USER=freeagentics \
  -e POSTGRES_PASSWORD=production_password \
  -p 5433:5432 \
  pgvector/pgvector:pg16

# Migrate data
pg_dump postgresql://freeagentics:old_pass@localhost:5432/freeagentics | \
  psql postgresql://freeagentics:production_password@localhost:5433/freeagentics

# Switch application to new database
export DATABASE_URL="postgresql://freeagentics:production_password@localhost:5433/freeagentics"

# After verification, remove old database
docker stop freeagentics-postgres-blue
```

#### 2. Online Schema Migration

```bash
# Use concurrent operations where possible
psql $DATABASE_URL << 'EOF'
-- Add new columns without locking
ALTER TABLE agents ADD COLUMN new_field TEXT;

-- Create indexes concurrently
CREATE INDEX CONCURRENTLY idx_agents_new_field ON agents(new_field);

-- Update data in batches
UPDATE agents SET new_field = 'default_value' WHERE id IN (
  SELECT id FROM agents WHERE new_field IS NULL LIMIT 1000
);
EOF
```

### Migration Rollback Procedures

#### 1. Immediate Rollback

```bash
# Keep old DATABASE_URL in environment
export DATABASE_URL_BACKUP="$DATABASE_URL"
export DATABASE_URL="$PREVIOUS_DATABASE_URL"

# Restart application with old database
make stop
make dev
```

#### 2. Data Rollback

```bash
# Restore from pre-migration backup
pg_restore -d $DATABASE_URL --clean backup_pre_migration.dump

# Or use point-in-time recovery if available
```

#### 3. Schema Rollback

```bash
# Downgrade database schema
alembic downgrade -1  # Go back one migration
# or
alembic downgrade 1b4306802749  # Go back to specific revision
```

## Backup and Restore

### Backup

```bash
# Full backup
pg_dump -U freeagentics -h localhost freeagentics > freeagentics_backup.sql

# Compressed backup
pg_dump -U freeagentics -h localhost -Fc freeagentics > freeagentics_backup.dump
```

### Restore

```bash
# From SQL file
psql -U freeagentics -h localhost freeagentics < freeagentics_backup.sql

# From compressed dump
pg_restore -U freeagentics -h localhost -d freeagentics freeagentics_backup.dump
```

## Next Steps

1. Copy `.env.example` to `.env` and configure DATABASE_URL
2. Run migrations: `make db-migrate`
3. Verify setup: `make db-check`
4. Start developing!

For production deployment, see [Production Database Guide](./PRODUCTION_DATABASE.md).
