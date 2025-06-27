# FreeAgentics Database Management

This directory contains all database-related code for FreeAgentics, including:

- SQLAlchemy models
- Alembic migrations
- Seed data scripts
- Database management utilities

## Overview

FreeAgentics uses PostgreSQL as its primary database with SQLAlchemy as the ORM and Alembic for migrations.

## Database Schema

The database consists of the following main tables:

### Core Tables

1. **agents** - AI agents in the system
   - Stores agent configuration, state, beliefs
   - Tracks location, energy, and experience
   - Supports different agent types (explorer, merchant, guardian, etc.)

2. **conversations** - Communication between agents
   - Supports direct, group, broadcast, and system conversations
   - Stores metadata and context

3. **messages** - Individual messages in conversations
   - Links to conversations and sender agents
   - Supports different message types

4. **knowledge_graphs** - Agent knowledge storage
   - Stores nodes and edges as JSON
   - Supports personal, shared, and global knowledge types
   - Access control via public flag and access lists

5. **coalitions** - Multi-agent collaborations
   - Tracks coalition goals, rules, and status
   - Manages shared value pools

6. **system_logs** - System monitoring and debugging
   - Tracks all system events with context
   - Links to relevant agents, conversations, coalitions

### Junction Tables

- **conversation_participants** - Links agents to conversations
- **coalition_members** - Links agents to coalitions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file or set environment variables:

```bash
export DATABASE_URL=postgresql://user:password@localhost:5432/freeagentics_dev
export DATABASE_POOL_SIZE=5
export DATABASE_AUTO_MIGRATE=true
export DATABASE_SEED_DATA=true
```

### 3. Initialize Database

The easiest way to set up the database:

```bash
cd database
python manage.py init
```

This will:

1. Create the database
2. Run all migrations
3. Seed initial data (if DATABASE_SEED_DATA=true)

## Database Management

### Using the Management Script

The `manage.py` script provides various database operations:

```bash
# Check database connection and status
python manage.py check

# Create database
python manage.py create-db
python manage.py create-db --force  # Drop if exists

# Run migrations
python manage.py migrate

# Rollback migrations
python manage.py rollback
python manage.py rollback --revision=base  # Rollback all

# Check migration status
python manage.py status

# Seed data
python manage.py seed --env=development
python manage.py seed --env=demo

# Drop database (with confirmation)
python manage.py drop-db
```

### Using Alembic Directly

For more control, you can use Alembic commands directly:

```bash
cd database

# Create a new migration
alembic revision -m "Description of changes"

# Auto-generate migration from model changes
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback one revision
alembic downgrade -1

# View history
alembic history
```

## Development Workflow

### 1. Making Model Changes

1. Edit models in `models.py`
2. Create a migration:

   ```bash
   alembic revision --autogenerate -m "Add new field to agents"
   ```

3. Review the generated migration file
4. Apply the migration:

   ```bash
   alembic upgrade head
   ```

### 2. Working with Seed Data

The seed scripts create realistic test data:

- **Development**: 15 agents with various conversations, knowledge graphs, and coalitions
- **Demo**: Specific scenario with Resource Optimizer, Market Maker, and Information Broker agents

To reseed the database:

```bash
# Drop and recreate with fresh data
python manage.py init
```

### 3. Docker Development

When using Docker Compose:

```bash
# Start services
docker-compose -f environments/development/docker-compose.yml up -d

# Initialize database in container
docker-compose exec backend python /app/database/manage.py init

# View logs
docker-compose logs postgres
```

## Models Reference

### Agent Model

```python
agent = Agent(
    uuid="unique-id",
    name="Agent Name",
    type="explorer",  # explorer, merchant, guardian, researcher, coordinator
    status=AgentStatus.ACTIVE,
    config={...},     # Agent configuration
    state={...},      # Current state
    beliefs={...},    # Active Inference beliefs
    location="h3_hex", # H3 hexagon location
    energy_level=1.0,
    experience_points=0
)
```

### Conversation Model

```python
conversation = Conversation(
    uuid="unique-id",
    title="Conversation Title",
    type=ConversationType.DIRECT,  # direct, group, broadcast, system
    meta_data={...},
    context={...}
)
```

### Knowledge Graph Model

```python
knowledge = KnowledgeGraph(
    uuid="unique-id",
    owner=agent,
    name="Knowledge Name",
    type="personal",  # personal, shared, global
    nodes=[...],      # Graph nodes
    edges=[...],      # Graph edges
    is_public=False,
    access_list=[]    # Agent IDs with access
)
```

## Environment Configuration

### Development

- Auto-migration enabled
- Seed data enabled
- Connection pooling: 5 connections
- Echo SQL: configurable

### Production

- Manual migrations only
- No seed data
- Connection pooling: 20+ connections
- SSL required

### Testing

- Isolated test database
- Fresh schema for each test run
- No connection pooling

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check PostgreSQL is running
   - Verify DATABASE_URL is correct
   - Check firewall/network settings

2. **Role Does Not Exist**
   - Create PostgreSQL user:

     ```sql
     CREATE USER freeagentics WITH PASSWORD 'password';
     CREATE DATABASE freeagentics_dev OWNER freeagentics;
     ```

3. **Migration Conflicts**
   - Check current status: `alembic current`
   - Resolve conflicts in alembic_version table
   - Use `alembic stamp head` to mark as current

4. **Import Errors**
   - Ensure PYTHONPATH includes project root
   - Check `prepend_sys_path` in alembic.ini

### Performance Tips

1. **Indexes**: All foreign keys and commonly queried fields are indexed
2. **JSON Fields**: Use PostgreSQL's JSON operators for efficient queries
3. **Connection Pooling**: Adjust pool size based on load
4. **Monitoring**: Use system_logs table for performance tracking

## Testing

Run database tests:

```bash
# Unit tests for models
pytest tests/unit/test_models.py

# Integration tests
pytest tests/integration/test_database.py

# Test migrations
alembic upgrade head
alembic downgrade base
alembic upgrade head
```

## Security Considerations

1. **Credentials**: Never commit database credentials
2. **SQL Injection**: Use parameterized queries (SQLAlchemy handles this)
3. **Access Control**: Implement row-level security for multi-tenant scenarios
4. **Encryption**: Enable SSL for production connections
5. **Backups**: Regular automated backups in production

## Future Enhancements

- [ ] Async SQLAlchemy support
- [ ] Database partitioning for large datasets
- [ ] Read replicas for scaling
- [ ] TimescaleDB for time-series data
- [ ] Full-text search with PostgreSQL extensions
