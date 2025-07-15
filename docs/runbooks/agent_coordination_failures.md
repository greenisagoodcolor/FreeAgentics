# Runbook: Agent Coordination Failures

## Alert Details
- **Alert Name**: AgentCoordinationFailure / CoalitionFormationTimeout
- **Threshold**: Coordination success rate < 80% or timeout rate > 20%
- **Severity**: SEV-3 (degraded), SEV-2 (< 50% success), SEV-1 (complete failure)

## Quick Actions

### 1. Check Agent System Status
```bash
# Overall agent system health
curl -s http://localhost:8000/api/v1/monitoring/agents/health | jq

# Active agents and their status
curl -s http://localhost:8000/api/v1/monitoring/agents | jq '.agents[] | {id, status, last_heartbeat, memory_mb}'

# Coordination metrics
curl -s http://localhost:8000/api/v1/monitoring/coordination/stats | jq

# Recent coordination failures
curl -s http://localhost:8000/api/v1/monitoring/coordination/failures?limit=10 | jq
```

### 2. Quick Diagnostics
```bash
# Check agent pool status
curl -s http://localhost:8000/api/v1/monitoring/agents/pool | jq

# View active coalitions
curl -s http://localhost:8000/api/v1/monitoring/coalitions | jq

# Recent coordination logs
docker-compose logs api | grep -E "(coordination|coalition|agent)" | grep -E "(ERROR|WARN)" | tail -50
```

## Diagnosis Decision Tree

```
Agent Coordination Failures
├─ Individual agent failures?
│  ├─ Yes → Agent-specific issue
│  │  ├─ Check agent memory/CPU
│  │  ├─ Review agent logs
│  │  └─ Restart failed agents
│  └─ No → System-wide issue
│     ├─ Check message queue
│     ├─ Review coordination algorithm
│     └─ Check resource constraints
│
├─ Failure pattern?
│  ├─ Timeouts → Performance issue
│  │  ├─ Check belief state size
│  │  ├─ Review coalition size
│  │  └─ Analyze computation time
│  ├─ Deadlocks → Coordination logic
│  │  ├─ Check circular dependencies
│  │  └─ Review locking mechanism
│  └─ Crashes → Memory/stability
│     ├─ Check OOM kills
│     └─ Review stack traces
│
└─ Correlation with?
   ├─ Load → Scaling issue
   ├─ Time → Scheduled interference
   └─ Features → Specific capability
```

## Mitigation Steps

### Immediate Relief (< 5 minutes)

#### 1. Reset Stuck Coordinations
```bash
# Clear stuck coordination tasks
curl -X POST http://localhost:8000/api/v1/system/coordination/reset

# Force cleanup of orphaned coalitions
curl -X POST http://localhost:8000/api/v1/system/coalitions/cleanup

# Reset coordination locks
curl -X POST http://localhost:8000/api/v1/system/coordination/locks/clear
```

#### 2. Reduce Coordination Complexity
```bash
# Temporarily reduce coalition size limits
curl -X PUT http://localhost:8000/api/v1/system/config \
  -H "Content-Type: application/json" \
  -d '{
    "max_coalition_size": 3,
    "coordination_timeout_ms": 10000,
    "belief_compression": true,
    "parallel_coordination": false
  }'

# Disable complex coordination strategies
curl -X PUT http://localhost:8000/api/v1/system/coordination/strategies \
  -H "Content-Type: application/json" \
  -d '{"enabled": ["simple", "greedy"], "disabled": ["optimal", "game_theoretic"]}'
```

#### 3. Restart Agent Subsystems
```bash
# Restart agent manager only
curl -X POST http://localhost:8000/api/v1/system/agents/restart

# If severe, rolling restart of API
docker-compose up -d --scale api=2
sleep 30
docker-compose restart api
docker-compose up -d --scale api=1
```

### Investigation (5-15 minutes)

#### 1. Analyze Coordination Patterns
```bash
# Coordination success/failure breakdown
curl -s http://localhost:8000/api/v1/monitoring/coordination/analysis | jq

# Time-based analysis
curl -s "http://localhost:8000/api/v1/monitoring/coordination/timeseries?interval=5m&duration=1h" | jq

# Agent participation patterns
curl -s http://localhost:8000/api/v1/monitoring/agents/participation | jq '.agents | sort_by(.failure_rate) | reverse | .[:10]'
```

#### 2. Check Resource Utilization
```bash
# Agent memory usage
curl -s http://localhost:8000/api/v1/monitoring/agents/resources | jq '.agents | sort_by(.memory_mb) | reverse | .[:10]'

# Belief state sizes
curl -s http://localhost:8000/api/v1/monitoring/agents/beliefs | jq '.agents | sort_by(.belief_size_kb) | reverse | .[:10]'

# PyMDP computation metrics
curl -s http://localhost:8000/api/v1/monitoring/pymdp/stats | jq
```

#### 3. Debug Specific Failures
```bash
# Get detailed trace of recent failure
FAILURE_ID=$(curl -s http://localhost:8000/api/v1/monitoring/coordination/failures?limit=1 | jq -r '.[0].id')
curl -s "http://localhost:8000/api/v1/monitoring/coordination/trace/$FAILURE_ID" | jq

# Check agent logs for specific coordination
COORDINATION_ID="coord_123"
docker-compose logs api | grep "$COORDINATION_ID"

# Enable debug logging temporarily
curl -X PUT http://localhost:8000/api/v1/system/logging \
  -H "Content-Type: application/json" \
  -d '{"level": "DEBUG", "components": ["coordination", "agents"]}'
```

### Root Cause Resolution

#### 1. Memory/Belief State Issues

**Symptoms**: Large belief states, OOM errors, slow coordination

```python
# Fix: Implement belief compression
async def compress_belief_state(agent_id: str):
    """Compress agent belief state to reduce memory"""
    agent = await get_agent(agent_id)
    
    # Implement belief pruning
    if agent.belief_state.size > BELIEF_SIZE_LIMIT:
        # Keep only recent and high-value beliefs
        agent.belief_state = prune_beliefs(
            agent.belief_state,
            max_size=BELIEF_SIZE_LIMIT,
            strategy="recency_and_value"
        )
    
    # Enable compression
    agent.belief_state = compress_beliefs(agent.belief_state)
    await save_agent(agent)

# Add to configuration
BELIEF_COMPRESSION_ENABLED = True
BELIEF_SIZE_LIMIT_MB = 10
BELIEF_PRUNING_STRATEGY = "adaptive"
```

#### 2. Coordination Deadlocks

**Symptoms**: Agents waiting on each other, circular dependencies

```python
# Fix: Implement deadlock detection and resolution
class CoordinationManager:
    async def detect_deadlocks(self):
        """Detect coordination deadlocks using wait-for graph"""
        wait_graph = await self.build_wait_graph()
        cycles = find_cycles(wait_graph)
        
        for cycle in cycles:
            # Break deadlock by timing out oldest coordination
            oldest = min(cycle, key=lambda c: c.start_time)
            await self.timeout_coordination(oldest.id)
            
    async def coordinate_with_timeout(self, agents, timeout_ms=5000):
        """Coordination with automatic timeout"""
        try:
            return await asyncio.wait_for(
                self._coordinate(agents),
                timeout=timeout_ms / 1000
            )
        except asyncio.TimeoutError:
            # Fallback to simple strategy
            return await self.simple_coordinate(agents)
```

#### 3. Message Queue Overload

**Symptoms**: High message latency, queue backlog

```bash
# Check Redis queue sizes
docker exec freeagentics-redis redis-cli LLEN coordination_queue
docker exec freeagentics-redis redis-cli LLEN agent_messages

# Clear old messages
docker exec freeagentics-redis redis-cli EVAL "
local keys = redis.call('keys', 'coord:msg:*')
local expired = 0
for i=1,#keys do
    local ttl = redis.call('ttl', keys[i])
    if ttl < 0 then
        redis.call('del', keys[i])
        expired = expired + 1
    end
end
return expired" 0

# Increase queue processing workers
curl -X PUT http://localhost:8000/api/v1/system/workers \
  -H "Content-Type: application/json" \
  -d '{"coordination_workers": 8, "message_workers": 16}'
```

#### 4. Coalition Formation Failures

**Symptoms**: Coalitions not forming, size constraints violated

```python
# Fix: Implement adaptive coalition formation
class AdaptiveCoalitionFormation:
    async def form_coalition(self, agents, objective):
        """Adaptively form coalitions based on current conditions"""
        # Start with optimal size
        ideal_size = self.calculate_ideal_size(objective)
        
        # Try formation with backoff
        for attempt in range(3):
            try:
                coalition = await self._try_form(
                    agents[:ideal_size], 
                    objective,
                    timeout=5000 * (attempt + 1)
                )
                return coalition
            except CoalitionFormationError:
                # Reduce size and retry
                ideal_size = max(2, ideal_size - 1)
                
        # Fallback to pairs
        return await self.form_pair_coalition(agents[:2], objective)
```

## Monitoring and Automation

### Coordination Health Dashboard
```python
# Add comprehensive monitoring endpoint
@app.get("/api/v1/monitoring/coordination/dashboard")
async def coordination_dashboard():
    return {
        "health_score": calculate_coordination_health(),
        "metrics": {
            "success_rate": await get_success_rate("1h"),
            "avg_duration_ms": await get_avg_duration("1h"),
            "timeout_rate": await get_timeout_rate("1h"),
            "deadlock_count": await get_deadlock_count("1h")
        },
        "active": {
            "coordinations": await get_active_coordinations(),
            "coalitions": await get_active_coalitions(),
            "blocked_agents": await get_blocked_agents()
        },
        "trends": {
            "success_trend": await get_metric_trend("success_rate", "24h"),
            "duration_trend": await get_metric_trend("duration", "24h")
        }
    }
```

### Auto-remediation Script
```bash
#!/bin/bash
# /scripts/coordination_auto_remediation.sh

# Configuration
SUCCESS_THRESHOLD=70
TIMEOUT_THRESHOLD=30
CHECK_INTERVAL=60

while true; do
    # Get current metrics
    METRICS=$(curl -s http://localhost:8000/api/v1/monitoring/coordination/stats)
    SUCCESS_RATE=$(echo $METRICS | jq -r '.success_rate')
    TIMEOUT_RATE=$(echo $METRICS | jq -r '.timeout_rate')
    
    # Check if intervention needed
    if (( $(echo "$SUCCESS_RATE < $SUCCESS_THRESHOLD" | bc -l) )); then
        echo "[$(date)] Low success rate: ${SUCCESS_RATE}%, applying remediation"
        
        # Clear stuck coordinations
        curl -X POST http://localhost:8000/api/v1/system/coordination/reset
        
        # Reduce complexity
        curl -X PUT http://localhost:8000/api/v1/system/config \
          -H "Content-Type: application/json" \
          -d '{"max_coalition_size": 3, "coordination_timeout_ms": 10000}'
        
    elif (( $(echo "$TIMEOUT_RATE > $TIMEOUT_THRESHOLD" | bc -l) )); then
        echo "[$(date)] High timeout rate: ${TIMEOUT_RATE}%, increasing timeouts"
        
        # Increase timeouts
        curl -X PUT http://localhost:8000/api/v1/system/config \
          -H "Content-Type: application/json" \
          -d '{"coordination_timeout_ms": 15000}'
    fi
    
    sleep $CHECK_INTERVAL
done
```

### Performance Analysis
```sql
-- Coordination performance analysis
WITH coordination_stats AS (
    SELECT 
        date_trunc('minute', started_at) as minute,
        COUNT(*) as total,
        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
        COUNT(CASE WHEN status = 'timeout' THEN 1 END) as timeouts,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failures,
        AVG(duration_ms) as avg_duration,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration
    FROM coordination_events
    WHERE started_at > NOW() - INTERVAL '1 hour'
    GROUP BY minute
)
SELECT 
    minute,
    total,
    ROUND(successful::numeric / total * 100, 2) as success_rate,
    ROUND(timeouts::numeric / total * 100, 2) as timeout_rate,
    ROUND(avg_duration::numeric, 2) as avg_duration_ms,
    ROUND(p95_duration::numeric, 2) as p95_duration_ms
FROM coordination_stats
ORDER BY minute DESC;

-- Agent performance in coordinations
SELECT 
    a.agent_id,
    a.agent_type,
    COUNT(DISTINCT ce.coordination_id) as coordinations,
    SUM(CASE WHEN ce.status = 'success' THEN 1 ELSE 0 END) as successes,
    AVG(ce.duration_ms) as avg_duration,
    SUM(CASE WHEN ce.status = 'timeout' THEN 1 ELSE 0 END) as timeouts
FROM coordination_events ce
JOIN coordination_participants cp ON ce.id = cp.coordination_id
JOIN agents a ON cp.agent_id = a.id
WHERE ce.started_at > NOW() - INTERVAL '1 hour'
GROUP BY a.agent_id, a.agent_type
ORDER BY timeouts DESC, avg_duration DESC
LIMIT 20;
```

## Prevention Strategies

### 1. Coordination Testing
```python
# Load test coordination system
async def test_coordination_under_load():
    """Test coordination with various loads and agent counts"""
    test_scenarios = [
        {"agents": 10, "coalition_size": 3, "concurrent": 5},
        {"agents": 50, "coalition_size": 5, "concurrent": 10},
        {"agents": 100, "coalition_size": 10, "concurrent": 20}
    ]
    
    for scenario in test_scenarios:
        agents = await create_test_agents(scenario["agents"])
        tasks = []
        
        for _ in range(scenario["concurrent"]):
            coalition = random.sample(agents, scenario["coalition_size"])
            tasks.append(coordinate_agents(coalition))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_rate = sum(1 for r in results if not isinstance(r, Exception)) / len(results)
        
        print(f"Scenario {scenario}: Success rate {success_rate:.2%}")
```

### 2. Configuration Best Practices
```python
# Optimal configuration for stability
COORDINATION_CONFIG = {
    # Size limits
    "max_coalition_size": 5,  # Keep small for reliability
    "min_coalition_size": 2,
    
    # Timeouts
    "coordination_timeout_ms": 5000,
    "message_timeout_ms": 1000,
    "lock_timeout_ms": 30000,
    
    # Performance
    "parallel_coordination": True,
    "max_concurrent_coordinations": 50,
    "belief_compression": True,
    "belief_size_limit_mb": 10,
    
    # Resilience
    "retry_attempts": 3,
    "retry_backoff_ms": 1000,
    "circuit_breaker_enabled": True,
    "circuit_breaker_threshold": 0.5
}
```

### 3. Regular Maintenance
```bash
# Daily maintenance tasks
0 2 * * * /scripts/cleanup_old_coordinations.sh
0 3 * * * /scripts/compress_agent_beliefs.sh
0 4 * * * /scripts/analyze_coordination_patterns.sh

# Cleanup script example
#!/bin/bash
# /scripts/cleanup_old_coordinations.sh

# Remove old coordination records
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "
DELETE FROM coordination_events 
WHERE started_at < NOW() - INTERVAL '7 days'
  AND status IN ('success', 'failed');"

# Clean Redis
docker exec freeagentics-redis redis-cli EVAL "
local count = 0
for _, key in ipairs(redis.call('keys', 'coord:*')) do
    if redis.call('ttl', key) == -1 then
        redis.call('expire', key, 86400)
        count = count + 1
    end
end
return count" 0
```

## Related Documentation
- [Agent Architecture Guide](../architecture/agents.md)
- [Coalition Formation Algorithms](../development/coalition_formation.md)
- [PyMDP Integration](../development/pymdp_integration.md)

---

*Last Updated: [Date]*
*Author: Platform Team*