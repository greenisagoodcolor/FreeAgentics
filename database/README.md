# Database Configuration

## Overview

FreeAgentics uses SQLAlchemy for database operations with support for both PostgreSQL (production) and SQLite (development).

## Development Setup

### Quick Start (SQLite)

1. Copy the example environment file:
   ```bash
   cp .env.development .env
   ```

2. The application will automatically use SQLite when:
   - `DEVELOPMENT_MODE=true` is set
   - `DATABASE_URL` is not set

3. The SQLite database will be created at `./freeagentics_dev.db`

### Using PostgreSQL in Development

If you prefer PostgreSQL for development:

1. Set up a local PostgreSQL instance
2. Set the `DATABASE_URL` in your `.env` file:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/freeagentics_dev
   ```

## Production Setup

Production environments **must** use PostgreSQL:

1. Set `DEVELOPMENT_MODE=false` or remove it entirely
2. Set `DATABASE_URL` to your PostgreSQL connection string:
   ```
   DATABASE_URL=postgresql://username:password@host:port/database
   ```

3. For production security:
   - Use strong passwords
   - Enable SSL: `DATABASE_URL=postgresql://...?sslmode=require`
   - Set `PRODUCTION=true` to enable additional security checks

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DEVELOPMENT_MODE` | Enable development features | `false` | No |
| `DATABASE_URL` | Database connection string | SQLite in dev mode | Yes in production |
| `DEBUG_SQL` | Log SQL queries | `false` | No |
| `DB_POOL_SIZE` | Connection pool size | `20` | No |
| `DB_MAX_OVERFLOW` | Max overflow connections | `40` | No |
| `DB_POOL_TIMEOUT` | Pool timeout in seconds | `30` | No |

## Database Migrations

Run migrations using Alembic:

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback one revision
alembic downgrade -1
```

## Troubleshooting

### "DATABASE_URL environment variable is required" Error

This error occurs when:
- Running in production mode without `DATABASE_URL` set
- `DEVELOPMENT_MODE` is not set to `true` in development

Solution:
- For development: Set `DEVELOPMENT_MODE=true` in your `.env` file
- For production: Set `DATABASE_URL` to your PostgreSQL connection string

### SQLite Threading Errors

If you encounter "SQLite objects created in a thread can only be used in that same thread":
- This is handled automatically by setting `check_same_thread=False`
- If issues persist, consider using PostgreSQL for development

### Connection Pool Exhaustion

If you see "QueuePool limit exceeded" errors:
- Increase `DB_POOL_SIZE` and `DB_MAX_OVERFLOW`
- Check for connection leaks in your code
- Ensure sessions are properly closed