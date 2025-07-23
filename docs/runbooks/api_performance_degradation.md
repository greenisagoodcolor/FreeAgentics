# Runbook: API Performance Degradation

## Alert Details
- **Alert Name**: APIHighLatency / APISlowResponse
- **Threshold**: p95 latency > 1s or p99 > 2s for 5 minutes
- **Severity**: SEV-3 (>1s), SEV-2 (>3s), SEV-1 (>5s or timeouts)

## Quick Actions

### 1. Verify Performance Impact
```bash
# Current API response times
curl -s http://localhost:8000/api/v1/monitoring/metrics/latency | jq '.current'

# Check request rate and error rate
curl -s http://localhost:8000/api/v1/monitoring/metrics/requests | jq

# View slow endpoints
curl -s http://localhost:8000/api/v1/monitoring/slow-queries?limit=10 | jq
```

### 2. Quick Health Check
```bash
# API health status
curl -s http://localhost:8000/api/v1/health | jq

# Active request count
curl -s http://localhost:8000/api/v1/monitoring/requests/active | jq

# Database query performance
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "SELECT query, mean_exec_time, calls FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```

## Diagnosis Decision Tree

```
API Performance Degradation
├─ Which endpoints affected?
│  ├─ All endpoints → System-wide issue
│  │  ├─ Check CPU/Memory usage
│  │  ├─ Review database performance
│  │  └─ Check network connectivity
│  └─ Specific endpoints → Targeted issue
│     ├─ Agent-related → Check coordination
│     ├─ Knowledge graph → Check query complexity
│     └─ Auth endpoints → Check token validation
│
├─ Pattern of degradation?
│  ├─ Gradual → Growing queue or leak
│  ├─ Sudden → Recent change or spike
│  └─ Periodic → Scheduled job interference
│
└─ Correlation with?
   ├─ High traffic → Scaling issue
   ├─ Deployments → Code regression
   └─ Time of day → Usage patterns
```

## Mitigation Steps

### Immediate Relief (< 5 minutes)

#### 1. Enable Circuit Breakers
```bash
# Enable circuit breaker for slow operations
curl -X PUT http://localhost:8000/api/v1/system/circuit-breaker \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "timeout_ms": 5000,
    "failure_threshold": 0.5,
    "recovery_time": 60
  }'

# Increase timeout for critical endpoints
curl -X PUT http://localhost:8000/api/v1/system/timeouts \
  -H "Content-Type: application/json" \
  -d '{"default_ms": 10000, "critical_ms": 30000}'
```

#### 2. Reduce Load
```bash
# Enable rate limiting if not already
curl -X PUT http://localhost:8000/api/v1/system/rate-limit \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "requests_per_minute": 1000}'

# Disable non-essential features
curl -X POST http://localhost:8000/api/v1/system/features/disable \
  -H "Content-Type: application/json" \
  -d '{"features": ["analytics", "recommendations"]}'
```

#### 3. Scale Resources
```bash
# Add more API instances
docker-compose up -d --scale api=3

# Increase worker threads
curl -X PUT http://localhost:8000/api/v1/system/workers \
  -H "Content-Type: application/json" \
  -d '{"count": 16}'
```

### Investigation (5-15 minutes)

#### 1. Analyze Slow Queries
```bash
# API endpoint analysis
curl -s http://localhost:8000/api/v1/monitoring/endpoints/stats | jq '.endpoints | sort_by(.avg_latency_ms) | reverse | .[:10]'

# Database slow queries
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT
  query,
  calls,
  mean_exec_time::numeric(10,2) as avg_ms,
  max_exec_time::numeric(10,2) as max_ms,
  total_exec_time::numeric(10,2) as total_ms
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC
LIMIT 20;"

# Redis slow log
docker exec freeagentics-redis redis-cli SLOWLOG GET 10
```

#### 2. Check Resource Bottlenecks
```bash
# CPU usage by process
docker stats --no-stream

# Database connections
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT
  state,
  count(*) as connections,
  max(now() - state_change) as max_duration
FROM pg_stat_activity
GROUP BY state;"

# Thread pool status
curl -s http://localhost:8000/api/v1/monitoring/threads | jq
```

#### 3. Trace Slow Requests
```bash
# Enable request tracing
curl -X PUT http://localhost:8000/api/v1/system/tracing \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "sample_rate": 0.1}'

# Get trace for slow request
curl -s http://localhost:8000/api/v1/monitoring/traces?min_duration_ms=1000 | jq '.[0]'
```

### Root Cause Resolution

#### 1. Database Performance Issues

**Symptoms**: High database CPU, slow queries, connection pool exhaustion

```bash
# Analyze query plans
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
EXPLAIN ANALYZE
SELECT * FROM your_slow_query_here;"

# Update statistics
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "ANALYZE;"

# Check missing indexes
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
SELECT
  schemaname,
  tablename,
  attname,
  n_distinct,
  correlation
FROM pg_stats
WHERE schemaname = 'public'
  AND n_distinct > 100
  AND correlation < 0.1
ORDER BY n_distinct DESC;"

# Add missing index (example)
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
CREATE INDEX CONCURRENTLY idx_agents_status_updated
ON agents(status, updated_at)
WHERE status = 'active';"
```

#### 2. Agent Coordination Bottlenecks

**Symptoms**: High latency on agent endpoints, coordination timeouts

```bash
# Check agent pool status
curl -s http://localhost:8000/api/v1/monitoring/agents/pool | jq

# Reduce coordination complexity
curl -X PUT http://localhost:8000/api/v1/system/agents/config \
  -H "Content-Type: application/json" \
  -d '{
    "max_coalition_size": 5,
    "coordination_timeout_ms": 5000,
    "parallel_execution": true
  }'

# Clear stuck coordinations
curl -X POST http://localhost:8000/api/v1/system/agents/coordination/cleanup
```

#### 3. Memory Pressure

**Symptoms**: High GC activity, memory swapping

```bash
# Check GC stats
curl -s http://localhost:8000/api/v1/monitoring/gc | jq

# Increase heap size
docker-compose down
# Edit docker-compose.yml to add:
# environment:
#   - PYTHON_GC_THRESHOLD=700,10,10
#   - PYTHONMALLOC=malloc
docker-compose up -d

# Force GC cycle
curl -X POST http://localhost:8000/api/v1/system/gc
```

#### 4. External Service Issues

**Symptoms**: Timeouts to external APIs, DNS resolution delays

```bash
# Check external service health
curl -s http://localhost:8000/api/v1/monitoring/external-services | jq

# Enable fallback mode
curl -X PUT http://localhost:8000/api/v1/system/fallback \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "cache_ttl": 3600}'

# Test DNS resolution
docker exec freeagentics-api nslookup external-api.example.com
```

## Performance Optimization Scripts

### Query Optimization Analyzer
```python
#!/usr/bin/env python3
# /scripts/query_optimizer.py

import psycopg2
import json
from datetime import datetime

conn = psycopg2.connect(
    host="localhost",
    database="freeagentics",
    user="postgres",
    password="postgres"
)

cur = conn.cursor()

# Get slow queries
cur.execute("""
    SELECT
        query,
        calls,
        mean_exec_time,
        total_exec_time
    FROM pg_stat_statements
    WHERE mean_exec_time > 100
    ORDER BY mean_exec_time DESC
    LIMIT 20
""")

slow_queries = cur.fetchall()

for query, calls, mean_time, total_time in slow_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query[:100]}...")
    print(f"Calls: {calls}, Avg: {mean_time:.2f}ms, Total: {total_time:.2f}ms")

    # Get query plan
    try:
        cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
        plan = cur.fetchone()[0]
        print(f"Cost: {plan[0]['Plan']['Total Cost']}")

        # Check for sequential scans
        if 'Seq Scan' in json.dumps(plan):
            print("WARNING: Sequential scan detected - consider adding index")
    except:
        print("Could not analyze query plan")

cur.close()
conn.close()
```

### Auto-scaling Script
```bash
#!/bin/bash
# /scripts/auto_scale_api.sh

# Configuration
LATENCY_THRESHOLD=1000  # milliseconds
SCALE_UP_THRESHOLD=3    # consecutive checks
SCALE_DOWN_THRESHOLD=10 # consecutive checks
MAX_INSTANCES=5
MIN_INSTANCES=1

# State file
STATE_FILE="/tmp/api_scale_state"
touch $STATE_FILE

# Get current latency
CURRENT_LATENCY=$(curl -s http://localhost:8000/api/v1/monitoring/metrics/latency | jq -r '.current.p95')
CURRENT_INSTANCES=$(docker-compose ps api | grep -c "Up")

# Load state
HIGH_LATENCY_COUNT=$(grep "high_count" $STATE_FILE | cut -d= -f2 || echo 0)
LOW_LATENCY_COUNT=$(grep "low_count" $STATE_FILE | cut -d= -f2 || echo 0)

if (( $(echo "$CURRENT_LATENCY > $LATENCY_THRESHOLD" | bc -l) )); then
    HIGH_LATENCY_COUNT=$((HIGH_LATENCY_COUNT + 1))
    LOW_LATENCY_COUNT=0

    if [ $HIGH_LATENCY_COUNT -ge $SCALE_UP_THRESHOLD ] && [ $CURRENT_INSTANCES -lt $MAX_INSTANCES ]; then
        NEW_INSTANCES=$((CURRENT_INSTANCES + 1))
        echo "[$(date)] Scaling up to $NEW_INSTANCES instances (latency: ${CURRENT_LATENCY}ms)"
        docker-compose up -d --scale api=$NEW_INSTANCES
        HIGH_LATENCY_COUNT=0
    fi
else
    LOW_LATENCY_COUNT=$((LOW_LATENCY_COUNT + 1))
    HIGH_LATENCY_COUNT=0

    if [ $LOW_LATENCY_COUNT -ge $SCALE_DOWN_THRESHOLD ] && [ $CURRENT_INSTANCES -gt $MIN_INSTANCES ]; then
        NEW_INSTANCES=$((CURRENT_INSTANCES - 1))
        echo "[$(date)] Scaling down to $NEW_INSTANCES instances (latency: ${CURRENT_LATENCY}ms)"
        docker-compose up -d --scale api=$NEW_INSTANCES
        LOW_LATENCY_COUNT=0
    fi
fi

# Save state
echo "high_count=$HIGH_LATENCY_COUNT" > $STATE_FILE
echo "low_count=$LOW_LATENCY_COUNT" >> $STATE_FILE
```

## Monitoring Queries

### Performance Tracking
```sql
-- Endpoint performance over time
SELECT
    date_trunc('minute', timestamp) as minute,
    endpoint,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99,
    count(*) as requests
FROM api_requests
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY minute, endpoint
ORDER BY minute DESC, p95 DESC;

-- Error rate by endpoint
SELECT
    endpoint,
    COUNT(CASE WHEN status_code >= 500 THEN 1 END)::float / COUNT(*) * 100 as error_rate,
    COUNT(*) as total_requests
FROM api_requests
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY endpoint
HAVING COUNT(*) > 10
ORDER BY error_rate DESC;
```

## Prevention Strategies

### 1. Performance Testing
```bash
# Load test with realistic traffic
ab -n 10000 -c 100 -H "Authorization: Bearer $TOKEN" \
   http://localhost:8000/api/v1/agents/

# Stress test coordination endpoint
hey -n 5000 -c 50 -m POST \
    -H "Content-Type: application/json" \
    -d '{"agent_ids": ["1", "2", "3"]}' \
    http://localhost:8000/api/v1/coordination/form
```

### 2. Proactive Monitoring
```yaml
# prometheus alerts
groups:
  - name: api_performance
    rules:
      - alert: APILatencyHigh
        expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 1
        for: 5m
        annotations:
          summary: "API p95 latency > 1s"
          runbook: "docs/runbooks/api_performance_degradation.md"
```

### 3. Configuration Best Practices
```python
# Optimal settings in .env
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
API_WORKER_THREADS=8
API_WORKER_CONNECTIONS=1000
REDIS_MAX_CONNECTIONS=50
REQUEST_TIMEOUT_SECONDS=30
CIRCUIT_BREAKER_ENABLED=true
```

## Related Documentation
- [Performance Tuning Guide](../performance/tuning.md)
- [Database Optimization](../database/optimization.md)
- [Load Testing Procedures](../testing/load_testing.md)

---

*Last Updated: [Date]*
*Author: Platform Team*
