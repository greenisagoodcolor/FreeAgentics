# PostgreSQL Database Setup Guide

This guide provides comprehensive instructions for setting up PostgreSQL for FreeAgentics in production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Database Setup](#database-setup)
5. [Running Migrations](#running-migrations)
6. [Performance Optimization](#performance-optimization)
7. [Backup and Restore](#backup-and-restore)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

- Ubuntu 20.04+ / Debian 10+ / RHEL 8+ / macOS
- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- At least 2GB RAM for development, 8GB+ for production
- 20GB+ disk space for database storage

## Installation

### Option 1: Local PostgreSQL Installation

#### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### RHEL/CentOS/Fedora

```bash
# Install PostgreSQL
sudo dnf install postgresql postgresql-server postgresql-contrib

# Initialize database
sudo postgresql-setup --initdb

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### macOS

```bash
# Using Homebrew
brew install postgresql
brew services start postgresql
```

### Option 2: Docker Deployment (Recommended)

```bash
# Start PostgreSQL using Docker Compose
docker-compose up -d postgres

# Verify PostgreSQL is running
docker-compose ps postgres
```

## Configuration

### 1. Automated Setup

Run the provided setup script for automated configuration:

```bash
# For local PostgreSQL
./scripts/setup-postgresql.sh

# For Docker deployment
./scripts/setup-postgresql.sh --docker
```

This script will:
- Generate secure passwords
- Create the database and user
- Configure connection pooling
- Create the .env.production file

### 2. Manual Setup

If you prefer manual setup:

#### Create Database User and Database

```sql
-- Connect as postgres superuser
sudo -u postgres psql

-- Create user
CREATE USER freeagentics WITH PASSWORD 'your_secure_password_here';

-- Create database
CREATE DATABASE freeagentics OWNER freeagentics;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE freeagentics TO freeagentics;

-- Enable extensions
\c freeagentics
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Exit
\q
```

#### Create .env.production File

```bash
# Copy the example file
cp .env.production.example .env.production

# Edit with your database credentials
vim .env.production
```

Update the following values:
```env
DATABASE_URL=postgresql://freeagentics:your_password@localhost:5432/freeagentics
POSTGRES_PASSWORD=your_password
REDIS_PASSWORD=your_redis_password
SECRET_KEY=your_32_char_secret_key
JWT_SECRET=your_32_char_jwt_secret
```

### 3. Connection Pool Configuration

The application uses the following connection pool settings by default:

```python
# In database/connection_manager.py
pool_config = {
    "pool_size": 10,          # Number of persistent connections
    "max_overflow": 20,       # Maximum overflow connections
    "pool_timeout": 10,       # Timeout for getting connection
    "pool_pre_ping": True,    # Test connections before use
    "pool_recycle": 1800,     # Recycle connections after 30 minutes
}
```

For high-traffic production environments, adjust in .env.production:
```env
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
```

## Database Setup

### 1. Test Database Connection

```bash
python scripts/test_database_connection.py
```

This will verify:
- Basic connectivity
- Connection pooling
- Table existence
- CRUD operations

### 2. Run Migrations

```bash
# Check current migration status
alembic current

# Run all migrations
alembic upgrade head

# Verify migrations
alembic history
```

### 3. Create Indexes

```bash
# Apply performance indexes
python scripts/apply_database_indexes.py
```

This creates indexes for:
- Agent status and template queries
- Coalition performance rankings
- Knowledge graph traversal
- JSON field searches (GIN indexes)

### 4. Load Seed Data

```bash
# Load initial data (optional)
python scripts/seed_database.py
```

This creates:
- 5 example agents with different templates
- 3 active coalitions
- Sample knowledge graph nodes and edges

## Performance Optimization

### 1. PostgreSQL Configuration

Edit `/etc/postgresql/15/main/postgresql.conf`:

```conf
# Memory settings (adjust based on available RAM)
shared_buffers = 256MB          # 25% of RAM for dedicated DB server
effective_cache_size = 1GB      # 50-75% of RAM
work_mem = 4MB                  # RAM per query operation
maintenance_work_mem = 64MB     # RAM for maintenance tasks

# Connection settings
max_connections = 200           # Adjust based on pool size

# Query planner
random_page_cost = 1.1          # For SSD storage
effective_io_concurrency = 200  # For SSD storage

# Logging (for production)
log_min_duration_statement = 1000  # Log queries slower than 1s
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
```

Restart PostgreSQL after changes:
```bash
sudo systemctl restart postgresql
```

### 2. Regular Maintenance

Set up a cron job for regular maintenance:

```bash
# Edit crontab
crontab -e

# Add maintenance tasks
# Daily VACUUM ANALYZE at 2 AM
0 2 * * * psql -U freeagentics -d freeagentics -c "VACUUM ANALYZE;"

# Weekly REINDEX at 3 AM on Sunday
0 3 * * 0 psql -U freeagentics -d freeagentics -c "REINDEX DATABASE freeagentics;"
```

### 3. Monitor Performance

```sql
-- Check slow queries
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 100  -- Queries averaging > 100ms
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan;

-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Backup and Restore

### 1. Automated Backups

```bash
# Set up daily backups
./scripts/database-backup.sh

# Or use the backup script with custom location
./scripts/backup/full-backup.sh --output /backup/location
```

### 2. Manual Backup

```bash
# Full database dump
pg_dump -U freeagentics -h localhost freeagentics > freeagentics_backup_$(date +%Y%m%d_%H%M%S).sql

# Compressed backup
pg_dump -U freeagentics -h localhost -Fc freeagentics > freeagentics_backup_$(date +%Y%m%d_%H%M%S).dump
```

### 3. Restore from Backup

```bash
# From SQL file
psql -U freeagentics -h localhost freeagentics < backup_file.sql

# From compressed dump
pg_restore -U freeagentics -h localhost -d freeagentics backup_file.dump
```

## Monitoring

### 1. Health Checks

```bash
# Check database status
pg_isready -h localhost -p 5432

# Check connection count
psql -U freeagentics -d freeagentics -c "SELECT count(*) FROM pg_stat_activity;"

# Check database size
psql -U freeagentics -d freeagentics -c "SELECT pg_database_size('freeagentics');"
```

### 2. Performance Metrics

Monitor these key metrics:
- Connection pool utilization
- Query response times
- Index hit ratios
- Table bloat
- Replication lag (if using replication)

### 3. Alerts

Set up alerts for:
- Connection pool exhaustion
- Long-running queries (> 5 seconds)
- Failed backups
- Disk space usage > 80%
- High number of deadlocks

## Troubleshooting

### Common Issues

#### 1. Connection Refused

```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# Verify listening address
sudo netstat -plnt | grep 5432
```

#### 2. Authentication Failed

```bash
# Check pg_hba.conf
sudo cat /etc/postgresql/15/main/pg_hba.conf

# Ensure it has:
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     peer
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
```

#### 3. Too Many Connections

```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Terminate idle connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'freeagentics'
  AND pid <> pg_backend_pid()
  AND state = 'idle'
  AND state_change < current_timestamp - interval '10 minutes';
```

#### 4. Slow Queries

```sql
-- Enable query logging
ALTER DATABASE freeagentics SET log_min_duration_statement = 1000;

-- Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
  AND n_distinct > 100
  AND correlation < 0.1
ORDER BY n_distinct DESC;
```

### Performance Tuning Checklist

- [ ] Indexes created and being used
- [ ] VACUUM and ANALYZE running regularly
- [ ] Connection pooling properly configured
- [ ] Query timeouts set appropriately
- [ ] Monitoring and alerting in place
- [ ] Regular backups verified
- [ ] PostgreSQL configuration optimized
- [ ] Application queries optimized
- [ ] Partitioning implemented for large tables
- [ ] Read replicas configured (if needed)

## Next Steps

1. Set up monitoring with Prometheus/Grafana
2. Configure automated failover with streaming replication
3. Implement partitioning for time-series data
4. Set up connection pooling with PgBouncer for very high loads
5. Configure SSL/TLS for encrypted connections

For additional help, refer to:
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [FreeAgentics Issue Tracker](https://github.com/yourusername/freeagentics/issues)
- [Database Performance Tuning Guide](./PERFORMANCE_OPTIMIZATION_GUIDE.md)