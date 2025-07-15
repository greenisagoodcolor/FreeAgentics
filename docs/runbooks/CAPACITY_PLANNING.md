# FreeAgentics Capacity Planning Runbook

## Overview
This runbook provides guidelines for capacity planning, performance monitoring, and scaling decisions for the FreeAgentics production environment.

## Current System Baseline

### Hardware Configuration
```yaml
Environment: Production
CPU: 8 cores (Intel Xeon or equivalent)
RAM: 16 GB
Storage: 500 GB SSD
Network: 1 Gbps
Load Balancer: Nginx
```

### Service Resource Allocation
```yaml
Backend (FastAPI):
  CPU: 2 cores
  Memory: 4 GB
  Replicas: 2

Frontend (Next.js):
  CPU: 1 core
  Memory: 2 GB
  Replicas: 2

PostgreSQL:
  CPU: 2 cores
  Memory: 4 GB
  Storage: 200 GB

Redis:
  CPU: 0.5 cores
  Memory: 2 GB
  Storage: 10 GB

Nginx:
  CPU: 0.5 cores
  Memory: 1 GB

Monitoring Stack:
  CPU: 1 core
  Memory: 3 GB
  Storage: 50 GB
```

## Performance Metrics Baseline

### Current Performance Indicators
```yaml
API Response Times:
  50th percentile: 150ms
  95th percentile: 500ms
  99th percentile: 1000ms

Throughput:
  Requests per second: 100
  Concurrent users: 50
  Active agents: 500

Resource Utilization:
  CPU average: 40%
  Memory average: 60%
  Disk I/O: 20%
  Network: 10%

Database Performance:
  Connection pool usage: 30%
  Query time average: 50ms
  Slow queries (>1s): <1%
```

## Scaling Thresholds

### Scale-Up Triggers
```yaml
CPU Utilization:
  Warning: >70% for 10 minutes
  Critical: >85% for 5 minutes
  Action: Add CPU cores or scale horizontally

Memory Utilization:
  Warning: >80% for 10 minutes
  Critical: >90% for 5 minutes
  Action: Add memory or optimize applications

Disk Usage:
  Warning: >80% used
  Critical: >90% used
  Action: Add storage or clean up data

Database Connections:
  Warning: >70% of max connections
  Critical: >85% of max connections
  Action: Optimize queries or scale database

Response Time:
  Warning: 95th percentile >1s
  Critical: 95th percentile >3s
  Action: Scale backend or optimize code

Error Rate:
  Warning: >2% for 5 minutes
  Critical: >5% for 2 minutes
  Action: Investigate and scale if needed
```

### Scale-Down Triggers
```yaml
Resource Utilization:
  CPU: <30% for 30 minutes
  Memory: <50% for 30 minutes
  Network: <20% for 30 minutes
  
Performance Indicators:
  Response time: 95th percentile <200ms for 1 hour
  Error rate: <0.5% for 1 hour
  Queue depth: <10 for 1 hour
```

## Growth Projections

### User Growth Scenarios

#### Conservative Growth (20% annually)
```yaml
Year 1:
  Users: 60 (current: 50)
  Agents: 600 (current: 500)
  Requests/sec: 120 (current: 100)
  
Year 2:
  Users: 72
  Agents: 720
  Requests/sec: 144
  
Year 3:
  Users: 86
  Agents: 860
  Requests/sec: 173
```

#### Moderate Growth (50% annually)
```yaml
Year 1:
  Users: 75
  Agents: 750
  Requests/sec: 150
  
Year 2:
  Users: 113
  Agents: 1125
  Requests/sec: 225
  
Year 3:
  Users: 169
  Agents: 1688
  Requests/sec: 338
```

#### Aggressive Growth (100% annually)
```yaml
Year 1:
  Users: 100
  Agents: 1000
  Requests/sec: 200
  
Year 2:
  Users: 200
  Agents: 2000
  Requests/sec: 400
  
Year 3:
  Users: 400
  Agents: 4000
  Requests/sec: 800
```

## Scaling Strategies

### Horizontal Scaling

#### Backend Services
```bash
# Scale backend instances
docker-compose up -d --scale backend=4

# Monitor performance impact
./scripts/performance-monitor.sh

# Load balancer will automatically distribute traffic
```

#### Database Scaling
```bash
# Read replicas for PostgreSQL
# Add to docker-compose.yml:
postgres-replica:
  image: postgres:15
  environment:
    POSTGRES_MASTER_SERVICE: postgres
    POSTGRES_REPLICA_USER: replica
  command: |
    bash -c "
    until pg_basebackup -h postgres -D /var/lib/postgresql/data -U replica -v -P -W
    do
      echo 'Waiting for master to connect...'
      sleep 1s
    done
    echo 'archive_mode = on' >> /var/lib/postgresql/data/postgresql.conf
    echo 'archive_command = cp %p /var/lib/postgresql/data/pg_wal/%f' >> /var/lib/postgresql/data/postgresql.conf
    postgres
    "
```

### Vertical Scaling

#### Memory Scaling
```bash
# Update docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 8G  # Increased from 4G
        reservations:
          memory: 4G  # Increased from 2G

# Apply changes
docker-compose up -d
```

#### CPU Scaling
```bash
# Update docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '4.0'  # Increased from 2.0
        reservations:
          cpus: '2.0'  # Increased from 1.0

# Apply changes
docker-compose up -d
```

## Capacity Planning Formulas

### Backend Capacity Calculation
```python
# Calculate required backend instances
def calculate_backend_capacity(target_rps, current_rps_per_instance=50):
    """
    Calculate number of backend instances needed
    
    Args:
        target_rps: Target requests per second
        current_rps_per_instance: Current capacity per instance
    
    Returns:
        Number of instances needed (with 20% buffer)
    """
    base_instances = math.ceil(target_rps / current_rps_per_instance)
    buffer_instances = math.ceil(base_instances * 0.2)
    return base_instances + buffer_instances

# Example usage
target_load = 300  # RPS
required_instances = calculate_backend_capacity(target_load)
print(f"Required backend instances: {required_instances}")
```

### Database Capacity Calculation
```python
# Calculate database resource requirements
def calculate_db_capacity(num_agents, growth_factor=1.5):
    """
    Calculate database capacity requirements
    
    Args:
        num_agents: Number of agents in system
        growth_factor: Growth buffer factor
    
    Returns:
        Dictionary with resource requirements
    """
    base_storage_gb = num_agents * 0.001  # 1MB per agent
    base_memory_gb = max(4, num_agents * 0.002)  # 2MB per agent, min 4GB
    base_connections = max(20, num_agents * 0.1)  # 0.1 connections per agent
    
    return {
        'storage_gb': base_storage_gb * growth_factor,
        'memory_gb': base_memory_gb * growth_factor,
        'max_connections': int(base_connections * growth_factor)
    }

# Example usage
agent_count = 2000
db_requirements = calculate_db_capacity(agent_count)
print(f"Database requirements: {db_requirements}")
```

### Redis Capacity Calculation
```python
# Calculate Redis memory requirements
def calculate_redis_capacity(concurrent_users, session_size_kb=100):
    """
    Calculate Redis memory requirements
    
    Args:
        concurrent_users: Peak concurrent users
        session_size_kb: Average session data size in KB
    
    Returns:
        Required memory in GB
    """
    base_memory_mb = concurrent_users * session_size_kb / 1024
    # Add 50% buffer for other cached data
    total_memory_mb = base_memory_mb * 1.5
    return max(1, total_memory_mb / 1024)  # Minimum 1GB

# Example usage
peak_users = 200
redis_memory_gb = calculate_redis_capacity(peak_users)
print(f"Redis memory requirement: {redis_memory_gb:.1f}GB")
```

## Monitoring and Alerting for Capacity

### Key Metrics to Track
```yaml
System Metrics:
  - CPU utilization (per core and average)
  - Memory usage (application and system)
  - Disk usage (space and I/O)
  - Network throughput and latency

Application Metrics:
  - Request rate and response times
  - Error rates and types
  - Queue depths and processing times
  - Active connections and sessions

Business Metrics:
  - Active users and agents
  - Feature usage patterns
  - Peak load times and durations
  - Growth rates and trends
```

### Capacity Alerts
```yaml
Capacity Warnings:
  - CPU >70% for 10 minutes
  - Memory >80% for 10 minutes
  - Disk >80% full
  - Response time >1s for 95th percentile
  - Error rate >2% for 5 minutes

Capacity Critical:
  - CPU >85% for 5 minutes
  - Memory >90% for 5 minutes
  - Disk >90% full
  - Response time >3s for 95th percentile
  - Error rate >5% for 2 minutes

Growth Alerts:
  - 20% increase in any metric over 7 days
  - Approaching scaling thresholds
  - Resource utilization trends
```

## Performance Testing

### Load Testing Scripts
```bash
# Backend load test
./scripts/load-test-backend.sh --users 100 --duration 300s

# Database stress test
./scripts/stress-test-database.sh --connections 50 --duration 600s

# End-to-end performance test
./scripts/e2e-performance-test.sh --scenario production
```

### Benchmark Results Format
```yaml
Test Configuration:
  Duration: 5 minutes
  Concurrent Users: 100
  Target RPS: 200

Results:
  Actual RPS: 195
  Response Times:
    50th: 150ms
    95th: 450ms
    99th: 800ms
  Error Rate: 0.2%
  
Resource Usage:
  CPU: 65%
  Memory: 72%
  Network: 15%
  
Bottlenecks:
  - Database connection pool saturation at 85%
  - Memory allocation spikes during peak load
```

## Scaling Procedures

### Emergency Scaling (< 15 minutes)
```bash
# Quick horizontal scale
docker-compose up -d --scale backend=4 --scale frontend=3

# Increase resource limits (requires restart)
# Edit docker-compose.yml and apply
docker-compose up -d
```

### Planned Scaling (Maintenance Window)
```bash
# 1. Schedule maintenance window
# 2. Backup current configuration
cp docker-compose.yml docker-compose.yml.backup

# 3. Update configuration with new resource limits
# 4. Test changes in staging environment
# 5. Apply changes during maintenance window
docker-compose up -d

# 6. Monitor and validate performance
./scripts/validate-performance.sh
```

### Database Scaling Procedure
```bash
# 1. Create read replica
docker-compose up -d postgres-replica

# 2. Update application configuration to use read replica
# 3. Monitor replication lag
docker exec postgres-replica psql -c "SELECT pg_last_wal_replay_lsn();"

# 4. Load balance read queries
# Update application connection string for read operations
```

## Cost Optimization

### Resource Right-Sizing
```bash
# Monitor actual resource usage
./scripts/resource-usage-report.sh --period 30days

# Identify over-provisioned services
# Adjust resource limits based on actual usage + buffer

# Example: Reduce frontend memory if usage <50%
services:
  frontend:
    deploy:
      resources:
        limits:
          memory: 1G  # Reduced from 2G
```

### Storage Optimization
```bash
# Implement data retention policies
# Clean up old logs and backups
find logs/ -mtime +30 -delete
find backups/ -mtime +90 -delete

# Database maintenance
docker exec postgres psql -U freeagentics -c "VACUUM FULL;"
docker exec postgres psql -U freeagentics -c "REINDEX DATABASE freeagentics;"
```

## Capacity Planning Calendar

### Monthly Reviews
- [ ] Review growth metrics vs projections
- [ ] Analyze resource utilization trends
- [ ] Update capacity forecasts
- [ ] Plan for seasonal traffic patterns

### Quarterly Assessments
- [ ] Comprehensive performance testing
- [ ] Review and update scaling thresholds
- [ ] Evaluate new technologies for scaling
- [ ] Update disaster recovery capacity

### Annual Planning
- [ ] Multi-year capacity projections
- [ ] Budget planning for infrastructure growth
- [ ] Technology roadmap alignment
- [ ] Vendor contract renewals

---

**Note**: All capacity planning decisions should be based on data-driven analysis and include appropriate safety margins for unexpected growth or peak loads.

*Last Updated: [Date]*