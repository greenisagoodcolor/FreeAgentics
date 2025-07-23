# Runbook: High Memory Usage

## Alert Details
- **Alert Name**: HighMemoryUsage
- **Threshold**: Memory usage > 85% for 5 minutes
- **Severity**: SEV-2 (>85%), SEV-1 (>95%)

## Quick Actions

### 1. Verify the Alert
```bash
# Check current memory usage across containers
docker stats --no-stream

# Check system memory
free -h

# View top memory consumers
docker exec freeagentics-api top -b -n 1 -o %MEM | head -20
```

### 2. Identify Memory Hot Spots
```bash
# Check API memory profiler
curl -s http://localhost:8000/api/v1/monitoring/memory | jq

# View memory by component
curl -s http://localhost:8000/api/v1/monitoring/metrics | grep memory

# Check for memory leaks in logs
docker-compose logs api | grep -i "memory" | tail -100
```

## Diagnosis Decision Tree

```
High Memory Alert
├─ Is it gradual increase?
│  ├─ Yes → Likely memory leak
│  │  ├─ Check recent deployments
│  │  ├─ Review memory profiler data
│  │  └─ Plan rolling restart
│  └─ No → Sudden spike
│     ├─ Check concurrent operations
│     ├─ Review agent pool size
│     └─ Check for large data processing
│
├─ Which component?
│  ├─ API Service
│  │  ├─ Check request queue
│  │  ├─ Review WebSocket connections
│  │  └─ Analyze endpoint memory usage
│  ├─ Agent System
│  │  ├─ Check active agent count
│  │  ├─ Review belief state sizes
│  │  └─ Check coalition formations
│  └─ Database
│     ├─ Check query memory usage
│     ├─ Review connection count
│     └─ Analyze cache hit rates
```

## Mitigation Steps

### Immediate Actions (< 5 minutes)

#### 1. Free Memory - Conservative Approach
```bash
# Clear expired cache entries
curl -X POST http://localhost:8000/api/v1/system/cache/cleanup

# Reduce agent pool if oversized
curl -X POST http://localhost:8000/api/v1/system/agents/pool \
  -H "Content-Type: application/json" \
  -d '{"max_size": 50}'

# Force garbage collection
curl -X POST http://localhost:8000/api/v1/system/gc
```

#### 2. If Critical (>95%)
```bash
# Rolling restart with zero downtime
docker-compose up -d --scale api=2
sleep 30
docker-compose restart api
docker-compose up -d --scale api=1

# Or emergency restart (30s downtime)
docker-compose restart api
```

### Investigation Steps (5-15 minutes)

#### 1. Analyze Memory Profile
```bash
# Generate memory heap dump
curl -X POST http://localhost:8000/api/v1/monitoring/memory/dump \
  -o memory_dump_$(date +%Y%m%d_%H%M%S).json

# Analyze top memory consumers
curl -s http://localhost:8000/api/v1/monitoring/memory/top | jq '.consumers[:10]'

# Check memory growth rate
curl -s http://localhost:8000/api/v1/monitoring/metrics/memory/trend?duration=1h | jq
```

#### 2. Identify Problematic Operations
```bash
# Long-running operations
curl -s http://localhost:8000/api/v1/monitoring/operations?status=running\&duration_gt=300 | jq

# Large belief states
curl -s http://localhost:8000/api/v1/monitoring/agents/memory | jq '.agents | sort_by(.memory_mb) | reverse | .[:5]'

# WebSocket connection count
curl -s http://localhost:8000/api/v1/monitoring/websocket/stats | jq
```

### Root Cause Analysis

#### Common Causes and Solutions

1. **Memory Leak in Agent System**
   ```bash
   # Check agent lifecycle
   curl -s http://localhost:8000/api/v1/monitoring/agents/lifecycle | jq

   # Force cleanup of orphaned agents
   curl -X POST http://localhost:8000/api/v1/system/agents/cleanup
   ```

2. **Large Belief States**
   ```bash
   # Enable belief compression
   curl -X PUT http://localhost:8000/api/v1/system/config \
     -H "Content-Type: application/json" \
     -d '{"belief_compression": true, "belief_size_limit_mb": 10}'

   # Clear oversized beliefs
   curl -X POST http://localhost:8000/api/v1/system/agents/beliefs/compact
   ```

3. **Database Connection Pool**
   ```bash
   # Check connection count
   docker exec freeagentics-postgres psql -U postgres -d freeagentics \
     -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"

   # Reset connection pool
   curl -X POST http://localhost:8000/api/v1/system/db/pool/reset
   ```

4. **Redis Memory Bloat**
   ```bash
   # Check Redis memory
   docker exec freeagentics-redis redis-cli INFO memory | grep used_memory_human

   # Analyze key patterns
   docker exec freeagentics-redis redis-cli --bigkeys

   # Clean expired keys
   docker exec freeagentics-redis redis-cli EVAL "return redis.call('del', unpack(redis.call('keys', 'exp:*')))" 0
   ```

## Prevention Measures

### Configuration Tuning
```python
# Update environment variables
AGENT_POOL_MAX_SIZE=100  # Reduce from default 200
BELIEF_COMPRESSION_ENABLED=true
BELIEF_SIZE_LIMIT_MB=10
MEMORY_MONITOR_INTERVAL=60  # seconds
MEMORY_ALERT_THRESHOLD=0.85

# API memory limits
API_MAX_MEMORY_MB=4096
API_GC_THRESHOLD_MB=3072
```

### Monitoring Queries
```sql
-- Track memory growth patterns
SELECT
    date_trunc('hour', timestamp) as hour,
    avg(memory_used_mb) as avg_memory,
    max(memory_used_mb) as max_memory,
    count(*) as sample_count
FROM metrics
WHERE metric_name = 'system.memory.used'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

-- Identify memory-intensive operations
SELECT
    operation_type,
    avg(memory_delta_mb) as avg_memory_increase,
    max(memory_delta_mb) as max_memory_increase,
    count(*) as operation_count
FROM operation_metrics
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY operation_type
ORDER BY avg_memory_increase DESC
LIMIT 10;
```

## Automation Scripts

### Auto-remediation Script
```bash
#!/bin/bash
# /scripts/auto_memory_remediation.sh

MEMORY_THRESHOLD=90
CURRENT_MEMORY=$(docker stats --no-stream --format "{{.MemPerc}}" freeagentics-api | sed 's/%//')

if (( $(echo "$CURRENT_MEMORY > $MEMORY_THRESHOLD" | bc -l) )); then
    echo "[$(date)] High memory detected: ${CURRENT_MEMORY}%"

    # Try conservative cleanup first
    curl -X POST http://localhost:8000/api/v1/system/cache/cleanup
    sleep 10

    # Check again
    NEW_MEMORY=$(docker stats --no-stream --format "{{.MemPerc}}" freeagentics-api | sed 's/%//')

    if (( $(echo "$NEW_MEMORY > $MEMORY_THRESHOLD" | bc -l) )); then
        echo "[$(date)] Memory still high after cleanup: ${NEW_MEMORY}%, performing restart"
        docker-compose restart api
    else
        echo "[$(date)] Memory recovered to: ${NEW_MEMORY}%"
    fi
fi
```

### Memory Trend Analysis
```python
#!/usr/bin/env python3
# /scripts/memory_trend_analysis.py

import requests
import pandas as pd
from datetime import datetime, timedelta

# Fetch memory metrics for last 24 hours
response = requests.get(
    "http://localhost:8000/api/v1/monitoring/metrics/memory",
    params={"duration": "24h", "interval": "5m"}
)

data = response.json()
df = pd.DataFrame(data['metrics'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Analyze trend
memory_growth_rate = df['value'].diff().mean()
current_memory = df['value'].iloc[-1]
time_to_limit = (4096 - current_memory) / memory_growth_rate if memory_growth_rate > 0 else float('inf')

print(f"Current Memory: {current_memory:.1f} MB")
print(f"Growth Rate: {memory_growth_rate:.2f} MB/5min")
print(f"Time to Limit: {time_to_limit:.1f} intervals ({time_to_limit * 5:.0f} minutes)")

if time_to_limit < 12:  # Less than 1 hour
    print("WARNING: Memory limit will be reached within 1 hour!")
```

## Post-Incident Actions

1. **Update Metrics**
   - Record memory usage patterns
   - Document which mitigation worked
   - Update thresholds if needed

2. **Code Review**
   - Review recent changes for memory leaks
   - Check for unbounded data structures
   - Verify cleanup in finally blocks

3. **Configuration Updates**
   ```yaml
   # docker-compose.yml memory limits
   services:
     api:
       deploy:
         resources:
           limits:
             memory: 4G
           reservations:
             memory: 2G
   ```

## Related Documentation
- [Memory Profiler Documentation](../performance/memory_profiling.md)
- [Agent Lifecycle Management](../development/agent_lifecycle.md)
- [Performance Tuning Guide](../performance/tuning.md)

---

*Last Updated: [Date]*
*Author: Platform Team*
