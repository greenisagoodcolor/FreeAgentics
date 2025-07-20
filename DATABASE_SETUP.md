# FreeAgentics Database Setup Quick Reference

## SQLite (Default for Development)

### Automatic Setup
```bash
cp .env.development .env
make dev
# Database created automatically at ./freeagentics_dev.db
```

### Requirements
- `DEVELOPMENT_MODE=true` in `.env`
- `DATABASE_URL` not set or empty

### Benefits
- ✅ Zero configuration
- ✅ No installation required
- ✅ Perfect for local development
- ✅ Automatic migrations

### Limitations
- ❌ Single user only
- ❌ No concurrent access
- ❌ Not for production
- ❌ Limited performance

## PostgreSQL (Production & Advanced Development)

### Quick Setup
```bash
# 1. Install PostgreSQL
sudo apt-get install postgresql  # Ubuntu/Debian
brew install postgresql           # macOS

# 2. Create database
sudo -u postgres psql << EOF
CREATE DATABASE freeagentics_dev;
CREATE USER freeagentics WITH PASSWORD 'secure-password';
GRANT ALL PRIVILEGES ON DATABASE freeagentics_dev TO freeagentics;
EOF

# 3. Configure .env
echo "DATABASE_URL=postgresql://freeagentics:secure-password@localhost:5432/freeagentics_dev" >> .env

# 4. Run migrations
make dev
```

### Docker PostgreSQL
```bash
# Using docker-compose (includes PostgreSQL)
docker-compose up -d
```

### Connection String Format
```
postgresql://[user]:[password]@[host]:[port]/[database]
```

Examples:
- Local: `postgresql://postgres:password@localhost:5432/freeagentics_dev`
- Docker: `postgresql://postgres:postgres@postgres:5432/freeagentics`
- Cloud: `postgresql://user:pass@aws-rds-instance.region.rds.amazonaws.com:5432/freeagentics`

## Switching Between Databases

### From SQLite to PostgreSQL
```bash
# 1. Set DATABASE_URL in .env
# 2. Restart application
make kill-ports && make dev
```

### From PostgreSQL to SQLite
```bash
# 1. Remove or comment out DATABASE_URL in .env
# 2. Ensure DEVELOPMENT_MODE=true
# 3. Restart application
make kill-ports && make dev
```

## Troubleshooting

### SQLite Issues
```bash
# Database locked
rm -f freeagentics_dev.db && make dev

# Permission denied
chmod 664 freeagentics_dev.db
```

### PostgreSQL Issues
```bash
# Connection refused
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# Authentication failed
# Check DATABASE_URL credentials

# Database does not exist
createdb freeagentics_dev
```

## Database Management Commands

### Migrations
```bash
# Run migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "Description"

# Rollback
alembic downgrade -1
```

### Database Reset
```bash
# SQLite
rm -f freeagentics_dev.db && make dev

# PostgreSQL
dropdb freeagentics_dev && createdb freeagentics_dev && make dev
```

## Production Considerations

1. **Always use PostgreSQL in production**
2. **Enable SSL/TLS connections**
3. **Use connection pooling**
4. **Configure proper backups**
5. **Set appropriate connection limits**

## Environment Variables Reference

```bash
# Development with SQLite
DEVELOPMENT_MODE=true
DATABASE_URL=  # Leave empty

# Development with PostgreSQL
DEVELOPMENT_MODE=true
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# Production
DEVELOPMENT_MODE=false
DATABASE_URL=postgresql://user:pass@prod-host:5432/dbname
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
```