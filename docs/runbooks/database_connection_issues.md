# Runbook: Database Connection Issues

## Alert Details
- **Alert Name**: DatabaseConnectionFailure / DatabasePoolExhausted
- **Threshold**: Connection errors > 5/min or pool utilization > 90%
- **Severity**: SEV-2 (intermittent), SEV-1 (complete failure)

## Quick Actions

### 1. Verify Database Health
```bash
# Check if PostgreSQL is running
docker ps | grep postgres
docker-compose logs --tail=50 postgres

# Test direct connection
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "SELECT 1;"

# Check connection count
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT count(*), state, application_name 
FROM pg_stat_activity 
GROUP BY state, application_name 
ORDER BY count DESC;"
```

### 2. Check Application Connection Pool
```bash
# API connection pool status
curl -s http://localhost:8000/api/v1/monitoring/database/pool | jq

# Active connections by endpoint
curl -s http://localhost:8000/api/v1/monitoring/database/connections | jq

# Recent connection errors
docker-compose logs api | grep -E "(connection|pool|database)" | grep -i error | tail -20
```

## Diagnosis Decision Tree

```
Database Connection Issues
├─ Can connect directly to DB?
│  ├─ No → Database is down
│  │  ├─ Check PostgreSQL logs
│  │  ├─ Verify disk space
│  │  └─ Check system resources
│  └─ Yes → Application issue
│     ├─ Pool exhausted?
│     │  ├─ Check pool size
│     │  ├─ Look for connection leaks
│     │  └─ Identify slow queries
│     └─ Network issue?
│        ├─ DNS resolution
│        ├─ Firewall rules
│        └─ Docker networking
│
├─ Error pattern?
│  ├─ Constant → Configuration issue
│  ├─ Intermittent → Resource contention
│  └─ Increasing → Leak or attack
│
└─ Recent changes?
   ├─ Config changes → Review settings
   ├─ Deployment → Check migrations
   └─ Traffic spike → Scale issue
```

## Mitigation Steps

### Immediate Relief (< 5 minutes)

#### 1. Reset Connection Pool
```bash
# Force close idle connections
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' 
  AND state_change < NOW() - INTERVAL '5 minutes'
  AND application_name = 'freeagentics-api';"

# Reset application pool
curl -X POST http://localhost:8000/api/v1/system/database/pool/reset

# Restart API to force new connections
docker-compose restart api
```

#### 2. Increase Connection Limits
```bash
# Temporarily increase PostgreSQL connections
docker exec freeagentics-postgres psql -U postgres -c "ALTER SYSTEM SET max_connections = 200;"
docker-compose restart postgres

# Increase application pool size
curl -X PUT http://localhost:8000/api/v1/system/config \
  -H "Content-Type: application/json" \
  -d '{
    "database_pool_size": 30,
    "database_max_overflow": 20
  }'
```

#### 3. Emergency Connection Cleanup
```bash
# Kill all non-essential connections
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE pid <> pg_backend_pid()
  AND state IN ('idle', 'idle in transaction')
  AND query NOT LIKE '%pg_stat_activity%';"

# Vacuum to free up resources
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "VACUUM;"
```

### Investigation (5-15 minutes)

#### 1. Analyze Connection Usage
```bash
# Detailed connection analysis
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT 
    application_name,
    client_addr,
    state,
    state_change,
    query_start,
    wait_event_type,
    wait_event,
    query
FROM pg_stat_activity 
WHERE state != 'idle'
ORDER BY query_start;"

# Connection duration analysis
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT 
    state,
    COUNT(*) as connections,
    AVG(NOW() - state_change) as avg_duration,
    MAX(NOW() - state_change) as max_duration
FROM pg_stat_activity
WHERE application_name = 'freeagentics-api'
GROUP BY state;"

# Check for connection leaks
curl -s http://localhost:8000/api/v1/monitoring/database/leaks | jq
```

#### 2. Identify Blocking Queries
```bash
# Find blocking queries
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;"

# Kill blocking query if necessary
# docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "SELECT pg_terminate_backend(BLOCKING_PID);"
```

#### 3. Check Resource Constraints
```bash
# Database resource usage
docker exec freeagentics-postgres psql -U postgres -c "
SELECT 
    setting AS max_connections,
    (SELECT count(*) FROM pg_stat_activity) AS current_connections,
    (SELECT count(*) FROM pg_stat_activity)::float / setting::float * 100 AS percentage_used
FROM pg_settings 
WHERE name = 'max_connections';"

# Disk space
docker exec freeagentics-postgres df -h /var/lib/postgresql/data

# Shared memory
docker exec freeagentics-postgres ipcs -m

# PostgreSQL logs for errors
docker-compose logs --tail=100 postgres | grep -E "(FATAL|ERROR|WARNING)"
```

### Root Cause Resolution

#### 1. Connection Pool Exhaustion

**Symptoms**: Pool timeout errors, all connections busy

```python
# Fix in application code
# Update database configuration
DATABASE_CONFIG = {
    'pool_size': 20,  # Increase from default
    'max_overflow': 10,  # Allow temporary connections
    'pool_timeout': 30,  # Increase timeout
    'pool_recycle': 3600,  # Recycle connections hourly
    'pool_pre_ping': True,  # Verify connections before use
}

# Add connection cleanup
async def cleanup_db_connections():
    """Regular cleanup of idle connections"""
    async with get_db() as db:
        await db.execute("""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE state = 'idle' 
            AND state_change < NOW() - INTERVAL '10 minutes'
            AND application_name = %s
        """, [APP_NAME])
```

#### 2. Connection Leaks

**Symptoms**: Gradually increasing connection count, idle connections

```python
# Detect leaks in code
# Add connection tracking
from contextlib import asynccontextmanager
import logging

connection_tracker = {}

@asynccontextmanager
async def tracked_db_connection():
    conn_id = id(asyncio.current_task())
    connection_tracker[conn_id] = {
        'start': datetime.now(),
        'stack': traceback.extract_stack()
    }
    
    try:
        async with get_db() as db:
            yield db
    finally:
        if conn_id in connection_tracker:
            duration = datetime.now() - connection_tracker[conn_id]['start']
            if duration > timedelta(seconds=30):
                logging.warning(f"Long-lived connection detected: {duration}")
            del connection_tracker[conn_id]

# Fix common leak patterns
# Bad:
def process_items(items):
    db = get_db()  # Connection never closed!
    for item in items:
        db.execute(...)

# Good:
async def process_items(items):
    async with get_db() as db:
        for item in items:
            await db.execute(...)
```

#### 3. PostgreSQL Configuration Issues

**Symptoms**: Max connection limit reached, performance degradation

```bash
# Optimize PostgreSQL settings
docker exec freeagentics-postgres psql -U postgres -c "
-- Increase connections
ALTER SYSTEM SET max_connections = 200;

-- Connection pooling settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';

-- Memory settings for connections
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- Connection timeout
ALTER SYSTEM SET statement_timeout = '30s';
ALTER SYSTEM SET idle_in_transaction_session_timeout = '5min';
"

# Apply changes
docker-compose restart postgres

# Create pgbouncer for connection pooling
cat > pgbouncer.ini << EOF
[databases]
freeagentics = host=postgres port=5432 dbname=freeagentics

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
EOF
```

#### 4. Network/DNS Issues

**Symptoms**: Intermittent connection failures, DNS resolution errors

```bash
# Test DNS resolution
docker exec freeagentics-api nslookup postgres
docker exec freeagentics-api ping -c 3 postgres

# Check Docker network
docker network inspect freeagentics_default

# Use IP instead of hostname temporarily
export DATABASE_URL="postgresql://postgres:postgres@172.18.0.2:5432/freeagentics"

# Fix Docker DNS
docker-compose down
docker network prune
docker-compose up -d
```

## Monitoring and Prevention

### Connection Pool Monitoring
```python
# Add to monitoring endpoint
@app.get("/api/v1/monitoring/database/pool/detailed")
async def get_pool_details():
    pool = app.state.db_pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total": pool.size() + pool.overflow(),
        "configuration": {
            "pool_size": pool._pool.size(),
            "max_overflow": pool._pool._max_overflow,
            "timeout": pool._pool._timeout,
            "recycle": pool._pool._recycle
        }
    }
```

### Automated Health Checks
```bash
#!/bin/bash
# /scripts/db_health_check.sh

# Configuration
WARNING_THRESHOLD=80
CRITICAL_THRESHOLD=90

# Get connection stats
STATS=$(docker exec freeagentics-postgres psql -U postgres -d freeagentics -t -c "
SELECT 
    setting::int as max_conn,
    (SELECT count(*) FROM pg_stat_activity)::int as current_conn
FROM pg_settings 
WHERE name = 'max_connections';")

MAX_CONN=$(echo $STATS | awk '{print $1}')
CURRENT_CONN=$(echo $STATS | awk '{print $2}')
PERCENTAGE=$((CURRENT_CONN * 100 / MAX_CONN))

# Alert based on thresholds
if [ $PERCENTAGE -ge $CRITICAL_THRESHOLD ]; then
    echo "CRITICAL: Database connections at ${PERCENTAGE}% (${CURRENT_CONN}/${MAX_CONN})"
    # Send alert
    curl -X POST http://localhost:8000/api/v1/alerts \
        -H "Content-Type: application/json" \
        -d "{\"severity\": \"critical\", \"message\": \"Database connections critical: ${PERCENTAGE}%\"}"
elif [ $PERCENTAGE -ge $WARNING_THRESHOLD ]; then
    echo "WARNING: Database connections at ${PERCENTAGE}% (${CURRENT_CONN}/${MAX_CONN})"
fi

# Check for idle connections
IDLE_COUNT=$(docker exec freeagentics-postgres psql -U postgres -d freeagentics -t -c "
SELECT count(*) FROM pg_stat_activity WHERE state = 'idle' AND state_change < NOW() - INTERVAL '5 minutes';")

if [ $IDLE_COUNT -gt 10 ]; then
    echo "WARNING: ${IDLE_COUNT} idle connections detected"
fi
```

### Connection Leak Detection
```sql
-- Query to run periodically
CREATE OR REPLACE VIEW connection_analysis AS
SELECT 
    application_name,
    client_addr,
    state,
    COUNT(*) as connection_count,
    AVG(EXTRACT(EPOCH FROM (NOW() - state_change))) as avg_age_seconds,
    MAX(EXTRACT(EPOCH FROM (NOW() - state_change))) as max_age_seconds,
    array_agg(pid ORDER BY state_change) as pids
FROM pg_stat_activity
WHERE application_name IS NOT NULL
GROUP BY application_name, client_addr, state
HAVING COUNT(*) > 1 OR MAX(EXTRACT(EPOCH FROM (NOW() - state_change))) > 300
ORDER BY connection_count DESC, max_age_seconds DESC;

-- Alert on potential leaks
SELECT * FROM connection_analysis 
WHERE state = 'idle' AND max_age_seconds > 600;
```

## Recovery Procedures

### Full Database Connection Recovery
```bash
#!/bin/bash
# /scripts/db_connection_recovery.sh

echo "[$(date)] Starting database connection recovery"

# 1. Kill idle connections
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' AND state_change < NOW() - INTERVAL '2 minutes';"

# 2. Reset application pools
curl -X POST http://localhost:8000/api/v1/system/database/pool/reset

# 3. Restart API services
docker-compose restart api

# 4. Verify recovery
sleep 10
HEALTH=$(curl -s http://localhost:8000/api/v1/health | jq -r '.database')

if [ "$HEALTH" = "healthy" ]; then
    echo "[$(date)] Database connection recovery successful"
else
    echo "[$(date)] Database connection recovery failed, escalating..."
    # Escalate to on-call
fi
```

## Related Documentation
- [Database Optimization Guide](../database/optimization.md)
- [Connection Pool Tuning](../performance/connection_pooling.md)
- [PostgreSQL Configuration](../database/postgresql_config.md)

---

*Last Updated: [Date]*
*Author: Platform Team*