# Scaling and Performance Optimization Guide

## Overview

This guide provides comprehensive procedures for scaling and optimizing the performance of the FreeAgentics system, including horizontal and vertical scaling, performance tuning, capacity planning, and optimization strategies.

## Table of Contents

1. [Scaling Architecture](#scaling-architecture)
2. [Horizontal Scaling](#horizontal-scaling)
3. [Vertical Scaling](#vertical-scaling)
4. [Database Scaling](#database-scaling)
5. [Performance Monitoring](#performance-monitoring)
6. [Performance Optimization](#performance-optimization)
7. [Capacity Planning](#capacity-planning)
8. [Auto-scaling Configuration](#auto-scaling-configuration)
9. [Load Testing](#load-testing)
10. [Performance Troubleshooting](#performance-troubleshooting)

## Scaling Architecture

### 1. Current Architecture Overview

#### System Components
```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (NGINX)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                API Service Cluster                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  API Pod 1  │  │  API Pod 2  │  │  API Pod 3  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Worker Service Cluster                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Worker Pod 1│  │ Worker Pod 2│  │ Worker Pod 3│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ PostgreSQL  │  │    Redis    │  │  File Store │        │
│  │  (Master)   │  │   Cluster   │  │    (S3)     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 2. Scaling Strategies

#### Scaling Dimensions
- **Horizontal Scaling**: Adding more instances
- **Vertical Scaling**: Increasing resource allocation
- **Database Scaling**: Read replicas and sharding
- **Cache Scaling**: Distributed caching
- **Geographic Scaling**: Multi-region deployment

#### Scaling Triggers
```yaml
scaling_triggers:
  cpu_utilization:
    scale_up: 70%
    scale_down: 30%
  memory_utilization:
    scale_up: 80%
    scale_down: 40%
  response_time:
    scale_up: 500ms
    scale_down: 200ms
  queue_depth:
    scale_up: 100
    scale_down: 10
```

## Horizontal Scaling

### 1. API Service Scaling

#### Kubernetes Horizontal Pod Autoscaler (HPA)
```yaml
# api-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: freeagentics
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

#### Docker Compose Scaling
```bash
# Scale API service
docker-compose up -d --scale api=5

# Scale worker service
docker-compose up -d --scale workers=3

# Verify scaling
docker-compose ps
```

#### Manual Scaling Commands
```bash
# Scale API deployment
kubectl scale deployment api-deployment --replicas=10

# Scale worker deployment
kubectl scale deployment worker-deployment --replicas=5

# Check scaling status
kubectl get pods -l app=api
kubectl get hpa
```

### 2. Worker Service Scaling

#### Queue-Based Scaling
```yaml
# worker-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: worker-hpa
  namespace: freeagentics
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: worker-deployment
  minReplicas: 2
  maxReplicas: 15
  metrics:
  - type: External
    external:
      metric:
        name: redis_queue_length
        selector:
          matchLabels:
            queue: task_queue
      target:
        type: AverageValue
        averageValue: "10"
```

#### Custom Metrics Scaling
```bash
# Configure custom metrics
./scripts/scaling/configure-custom-metrics.sh

# Deploy custom metrics API
kubectl apply -f custom-metrics-api.yaml

# Test custom metrics
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/freeagentics/pods/*/redis_queue_length"
```

### 3. Load Balancer Configuration

#### NGINX Load Balancer
```nginx
# nginx.conf
upstream api_backend {
    least_conn;
    server api-1:8000 max_fails=3 fail_timeout=30s;
    server api-2:8000 max_fails=3 fail_timeout=30s;
    server api-3:8000 max_fails=3 fail_timeout=30s;
    server api-4:8000 max_fails=3 fail_timeout=30s backup;
}

server {
    listen 80;
    server_name api.freeagentics.io;

    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}
```

#### HAProxy Configuration
```bash
# haproxy.cfg
global
    daemon
    maxconn 4096
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    
frontend api_frontend
    bind *:80
    default_backend api_servers
    
backend api_servers
    balance roundrobin
    option httpchk GET /health
    server api1 api-1:8000 check
    server api2 api-2:8000 check
    server api3 api-3:8000 check
    server api4 api-4:8000 check backup
```

## Vertical Scaling

### 1. Resource Optimization

#### Container Resource Limits
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: freeagentics/api:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: WORKERS
          value: "4"
        - name: MAX_CONNECTIONS
          value: "1000"
```

#### JVM Optimization (if using Java components)
```bash
# Java application optimization
export JAVA_OPTS="-Xms2g -Xmx4g -XX:NewRatio=3 -XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# Monitor JVM performance
jstat -gc $PID 1s
jstack $PID
```

#### Python Application Optimization
```python
# gunicorn configuration
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 30
keepalive = 2
```

### 2. Database Vertical Scaling

#### PostgreSQL Optimization
```bash
# PostgreSQL configuration tuning
./scripts/scaling/optimize-postgresql.sh

# Key parameters
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.7
wal_buffers = 16MB
default_statistics_target = 100
```

#### Redis Optimization
```bash
# Redis configuration
./scripts/scaling/optimize-redis.sh

# Key settings
maxmemory 1gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
tcp-keepalive 60
timeout 0
```

## Database Scaling

### 1. Read Replicas

#### PostgreSQL Read Replicas
```bash
# Create read replica
./scripts/scaling/create-read-replica.sh

# Configure read replica
./scripts/scaling/configure-read-replica.sh --master-host db-master --replica-host db-replica-1

# Test read replica
psql -h db-replica-1 -U postgres -c "SELECT pg_is_in_recovery();"
```

#### Application Read/Write Splitting
```python
# Database connection routing
class DatabaseRouter:
    def __init__(self):
        self.write_db = create_connection("postgresql://user:pass@db-master:5432/db")
        self.read_db = create_connection("postgresql://user:pass@db-replica:5432/db")
    
    def get_connection(self, operation='read'):
        if operation == 'write':
            return self.write_db
        else:
            return self.read_db

# Usage
router = DatabaseRouter()
read_conn = router.get_connection('read')
write_conn = router.get_connection('write')
```

### 2. Database Sharding

#### Horizontal Sharding Strategy
```python
# Sharding implementation
class ShardRouter:
    def __init__(self):
        self.shards = {
            'shard1': create_connection("postgresql://user:pass@shard1:5432/db"),
            'shard2': create_connection("postgresql://user:pass@shard2:5432/db"),
            'shard3': create_connection("postgresql://user:pass@shard3:5432/db")
        }
    
    def get_shard(self, key):
        shard_id = hash(key) % len(self.shards)
        return list(self.shards.values())[shard_id]
    
    def execute_query(self, key, query):
        shard = self.get_shard(key)
        return shard.execute(query)
```

#### Database Partitioning
```sql
-- Table partitioning
CREATE TABLE agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create partitions
CREATE TABLE agents_2024_q1 PARTITION OF agents
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE agents_2024_q2 PARTITION OF agents
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
```

### 3. Connection Pooling

#### PgBouncer Configuration
```bash
# PgBouncer setup
./scripts/scaling/setup-pgbouncer.sh

# pgbouncer.ini
[databases]
freeagentics = host=db-master port=5432 dbname=freeagentics

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
server_reset_query = DISCARD ALL
max_client_conn = 100
default_pool_size = 20
```

## Performance Monitoring

### 1. Application Performance Monitoring

#### Prometheus Metrics
```python
# Application metrics
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('freeagentics_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('freeagentics_request_duration_seconds', 'Request latency')
ACTIVE_CONNECTIONS = Gauge('freeagentics_active_connections', 'Active connections')

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    
    response = await call_next(request)
    
    REQUEST_LATENCY.observe(time.time() - start_time)
    return response
```

#### Custom Performance Monitoring
```bash
# Performance monitoring script
./scripts/monitoring/performance-monitor.sh

# Collect system metrics
./scripts/monitoring/collect-system-metrics.sh

# Generate performance report
./scripts/monitoring/generate-performance-report.sh --period 24h
```

### 2. Database Performance Monitoring

#### PostgreSQL Performance Queries
```sql
-- Monitor slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check database connections
SELECT count(*) as active_connections
FROM pg_stat_activity
WHERE state = 'active';

-- Monitor table bloat
SELECT schemaname, tablename, n_dead_tup, n_live_tup
FROM pg_stat_user_tables
WHERE n_dead_tup > 0
ORDER BY n_dead_tup DESC;
```

#### Redis Performance Monitoring
```bash
# Redis performance monitoring
redis-cli INFO memory
redis-cli INFO stats
redis-cli INFO clients

# Monitor Redis slowlog
redis-cli SLOWLOG GET 10
```

### 3. Network Performance Monitoring

#### Network Metrics Collection
```bash
# Network monitoring
./scripts/monitoring/network-monitor.sh

# Bandwidth monitoring
iftop -i eth0 -t -s 10

# Connection monitoring
netstat -an | grep :8000 | wc -l
```

## Performance Optimization

### 1. Application Code Optimization

#### Database Query Optimization
```python
# Optimized database queries
class OptimizedQueries:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_agents_with_stats(self, limit=100):
        # Use single query with joins instead of N+1 queries
        query = """
        SELECT a.id, a.name, a.status, 
               COUNT(t.id) as task_count,
               AVG(t.duration) as avg_duration
        FROM agents a
        LEFT JOIN tasks t ON a.id = t.agent_id
        GROUP BY a.id, a.name, a.status
        ORDER BY a.created_at DESC
        LIMIT %s
        """
        return self.db.execute(query, (limit,))
    
    def get_agent_performance(self, agent_id):
        # Use prepared statement
        query = """
        SELECT success_rate, avg_response_time, total_tasks
        FROM agent_performance_view
        WHERE agent_id = $1
        """
        return self.db.execute(query, (agent_id,))
```

#### Caching Strategy
```python
# Multi-level caching
import redis
from functools import wraps

redis_client = redis.Redis(host='redis-cluster', port=6379, decode_responses=True)

def cache_result(expiry=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(cache_key, expiry, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_result(expiry=1800)
def get_agent_stats(agent_id):
    # Expensive database operation
    return calculate_agent_performance(agent_id)
```

### 2. Database Optimization

#### Index Optimization
```sql
-- Create performance indexes
CREATE INDEX CONCURRENTLY idx_agents_status_created 
ON agents(status, created_at);

CREATE INDEX CONCURRENTLY idx_tasks_agent_id_status 
ON tasks(agent_id, status) 
WHERE status IN ('pending', 'processing');

-- Partial index for active agents
CREATE INDEX CONCURRENTLY idx_agents_active 
ON agents(id, created_at) 
WHERE status = 'active';
```

#### Query Optimization
```sql
-- Optimize frequent queries
EXPLAIN ANALYZE SELECT * FROM agents WHERE status = 'active';

-- Use materialized views for complex aggregations
CREATE MATERIALIZED VIEW agent_performance_summary AS
SELECT 
    agent_id,
    COUNT(*) as total_tasks,
    AVG(duration) as avg_duration,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)::float / COUNT(*) as success_rate
FROM tasks
GROUP BY agent_id;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW CONCURRENTLY agent_performance_summary;
```

### 3. Memory Optimization

#### Application Memory Management
```python
# Memory optimization techniques
import gc
import psutil
import weakref

class MemoryOptimizer:
    def __init__(self):
        self.object_pool = weakref.WeakValueDictionary()
        self.memory_threshold = 0.8  # 80% memory usage
    
    def get_pooled_object(self, key, factory):
        if key in self.object_pool:
            return self.object_pool[key]
        
        obj = factory()
        self.object_pool[key] = obj
        return obj
    
    def check_memory_usage(self):
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self.memory_threshold:
            self.cleanup_memory()
    
    def cleanup_memory(self):
        # Force garbage collection
        gc.collect()
        
        # Clear object pool
        self.object_pool.clear()
        
        # Clear application caches
        self.clear_caches()
```

## Capacity Planning

### 1. Resource Forecasting

#### Capacity Planning Script
```bash
# Capacity planning analysis
./scripts/capacity/analyze-capacity.sh --period 30d

# Generate capacity forecast
./scripts/capacity/forecast-capacity.sh --horizon 90d

# Recommend scaling actions
./scripts/capacity/recommend-scaling.sh
```

#### Resource Utilization Analysis
```python
# Resource utilization forecasting
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class CapacityPlanner:
    def __init__(self):
        self.models = {}
    
    def analyze_trends(self, metric_data, days_ahead=30):
        # Prepare data
        df = pd.DataFrame(metric_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Train model
        X = df[['hour', 'day_of_week']].values
        y = df['value'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast
        future_hours = np.arange(0, days_ahead * 24)
        future_dow = np.repeat(np.arange(7), days_ahead * 24 // 7)
        
        predictions = model.predict(np.column_stack([future_hours % 24, future_dow]))
        
        return {
            'forecast': predictions,
            'max_predicted': np.max(predictions),
            'avg_predicted': np.mean(predictions),
            'recommendation': self.get_recommendation(predictions)
        }
    
    def get_recommendation(self, predictions):
        max_utilization = np.max(predictions)
        if max_utilization > 0.8:
            return "Scale up recommended"
        elif max_utilization < 0.3:
            return "Scale down possible"
        else:
            return "Current capacity sufficient"
```

### 2. Cost Optimization

#### Resource Cost Analysis
```bash
# Cost analysis
./scripts/capacity/cost-analysis.sh

# Optimize resource allocation
./scripts/capacity/optimize-resources.sh

# Generate cost report
./scripts/capacity/generate-cost-report.sh --period monthly
```

#### Right-sizing Recommendations
```python
# Resource right-sizing
class ResourceOptimizer:
    def __init__(self):
        self.cpu_utilization_threshold = 0.7
        self.memory_utilization_threshold = 0.8
    
    def analyze_resource_usage(self, metrics):
        recommendations = []
        
        for service, data in metrics.items():
            cpu_avg = np.mean(data['cpu_usage'])
            memory_avg = np.mean(data['memory_usage'])
            
            if cpu_avg < 0.3 and memory_avg < 0.4:
                recommendations.append({
                    'service': service,
                    'action': 'downsize',
                    'current_cpu': data['cpu_limit'],
                    'recommended_cpu': data['cpu_limit'] * 0.7,
                    'current_memory': data['memory_limit'],
                    'recommended_memory': data['memory_limit'] * 0.7
                })
            elif cpu_avg > 0.8 or memory_avg > 0.9:
                recommendations.append({
                    'service': service,
                    'action': 'upsize',
                    'current_cpu': data['cpu_limit'],
                    'recommended_cpu': data['cpu_limit'] * 1.5,
                    'current_memory': data['memory_limit'],
                    'recommended_memory': data['memory_limit'] * 1.3
                })
        
        return recommendations
```

## Auto-scaling Configuration

### 1. Kubernetes Auto-scaling

#### Vertical Pod Autoscaler (VPA)
```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: api-vpa
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: api-deployment
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
```

#### Cluster Autoscaler
```yaml
# cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/freeagentics-cluster
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
```

### 2. Custom Auto-scaling

#### Custom Metrics Auto-scaling
```python
# Custom auto-scaling logic
import kubernetes
from kubernetes import client, config

class CustomAutoscaler:
    def __init__(self):
        config.load_incluster_config()
        self.apps_v1 = client.AppsV1Api()
        self.metrics_api = client.CustomObjectsApi()
    
    def get_queue_depth(self):
        # Get queue depth from Redis
        import redis
        redis_client = redis.Redis(host='redis-cluster')
        return redis_client.llen('task_queue')
    
    def scale_workers(self, target_replicas):
        # Scale worker deployment
        body = {'spec': {'replicas': target_replicas}}
        self.apps_v1.patch_namespaced_deployment_scale(
            name='worker-deployment',
            namespace='freeagentics',
            body=body
        )
    
    def autoscale_based_on_queue(self):
        queue_depth = self.get_queue_depth()
        current_replicas = self.get_current_replicas('worker-deployment')
        
        # Scale based on queue depth
        if queue_depth > 100:
            target_replicas = min(current_replicas + 2, 20)
        elif queue_depth < 10:
            target_replicas = max(current_replicas - 1, 2)
        else:
            target_replicas = current_replicas
        
        if target_replicas != current_replicas:
            self.scale_workers(target_replicas)
```

## Load Testing

### 1. Load Testing Strategy

#### Load Test Planning
```bash
# Load test planning
./scripts/load-testing/plan-load-test.sh

# Define test scenarios
./scripts/load-testing/define-scenarios.sh

# Prepare test environment
./scripts/load-testing/prepare-test-env.sh
```

#### Load Test Scenarios
```yaml
# load-test-scenarios.yaml
scenarios:
  - name: normal_load
    description: "Normal operating conditions"
    virtual_users: 100
    duration: 300s
    ramp_up: 60s
    
  - name: peak_load
    description: "Peak traffic conditions"
    virtual_users: 500
    duration: 600s
    ramp_up: 120s
    
  - name: stress_test
    description: "Stress testing to find breaking point"
    virtual_users: 1000
    duration: 1800s
    ramp_up: 300s
    
  - name: spike_test
    description: "Sudden traffic spikes"
    virtual_users: 200
    duration: 900s
    spike_users: 800
    spike_duration: 60s
```

### 2. Load Testing Tools

#### K6 Load Testing
```javascript
// k6-load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 200 }, // Ramp to 200 users
    { duration: '5m', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
  },
};

export default function() {
  let response = http.get('https://api.freeagentics.io/api/v1/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  sleep(1);
}
```

#### JMeter Load Testing
```bash
# JMeter load test
./scripts/load-testing/jmeter-test.sh

# Run JMeter test
jmeter -n -t freeagentics-load-test.jmx -l results.jtl -e -o report/

# Generate HTML report
jmeter -g results.jtl -o html-report/
```

### 3. Performance Benchmarking

#### API Performance Benchmarking
```bash
# API benchmarking
./scripts/benchmarking/api-benchmark.sh

# Apache Bench testing
ab -n 10000 -c 100 -H "Authorization: Bearer token" https://api.freeagentics.io/api/v1/agents

# wrk testing
wrk -t12 -c400 -d30s --header "Authorization: Bearer token" https://api.freeagentics.io/api/v1/agents
```

#### Database Performance Benchmarking
```bash
# Database benchmarking
./scripts/benchmarking/db-benchmark.sh

# pgbench testing
pgbench -i -s 10 freeagentics
pgbench -c 10 -j 2 -t 10000 freeagentics
```

## Performance Troubleshooting

### 1. Common Performance Issues

#### High CPU Usage
```bash
# Identify CPU bottlenecks
./scripts/troubleshooting/cpu-analysis.sh

# Profile application
./scripts/troubleshooting/profile-cpu.sh

# Optimize CPU-intensive code
./scripts/troubleshooting/optimize-cpu.sh
```

#### Memory Leaks
```bash
# Memory leak detection
./scripts/troubleshooting/memory-leak-detection.sh

# Memory profiling
./scripts/troubleshooting/memory-profiling.sh

# Memory optimization
./scripts/troubleshooting/optimize-memory.sh
```

#### Database Performance Issues
```bash
# Database performance analysis
./scripts/troubleshooting/db-performance-analysis.sh

# Query optimization
./scripts/troubleshooting/optimize-queries.sh

# Index optimization
./scripts/troubleshooting/optimize-indexes.sh
```

### 2. Performance Debugging

#### Application Profiling
```python
# Python profiling
import cProfile
import pstats
import io

def profile_function(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        print(s.getvalue())
        return result
    return wrapper

@profile_function
def slow_function():
    # Function to profile
    pass
```

#### Database Query Analysis
```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
SELECT pg_reload_conf();

-- Analyze slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
WHERE mean_time > 1000  -- Queries taking more than 1 second
ORDER BY mean_time DESC;

-- Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
AND n_distinct > 100
AND correlation < 0.1;
```

### 3. Performance Monitoring and Alerting

#### Real-time Performance Monitoring
```bash
# Real-time monitoring
./scripts/monitoring/realtime-performance.sh

# Performance alerts
./scripts/monitoring/setup-performance-alerts.sh

# Performance dashboard
./scripts/monitoring/create-performance-dashboard.sh
```

#### Performance Metrics Collection
```python
# Performance metrics collector
import time
import psutil
import threading

class PerformanceCollector:
    def __init__(self):
        self.metrics = {}
        self.running = False
    
    def start_collection(self):
        self.running = True
        threading.Thread(target=self._collect_metrics).start()
    
    def _collect_metrics(self):
        while self.running:
            self.metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_io': psutil.disk_io_counters()._asdict(),
                'network_io': psutil.net_io_counters()._asdict(),
                'load_avg': psutil.getloadavg()
            }
            time.sleep(1)
    
    def get_metrics(self):
        return self.metrics
```

## Best Practices

### 1. Scaling Best Practices

#### Scaling Guidelines
- **Scale gradually**: Avoid sudden scaling changes
- **Monitor during scaling**: Watch for issues during scale operations
- **Test scaling**: Regularly test auto-scaling behavior
- **Plan for failure**: Design for graceful degradation

#### Performance Best Practices
- **Profile before optimizing**: Identify actual bottlenecks
- **Optimize the database**: Often the primary bottleneck
- **Use caching effectively**: Implement multi-level caching
- **Monitor continuously**: Set up comprehensive monitoring

### 2. Resource Management

#### Resource Allocation
- **Right-size resources**: Avoid over-provisioning
- **Use resource limits**: Set appropriate limits in containers
- **Monitor resource usage**: Track utilization trends
- **Plan for growth**: Anticipate future needs

#### Cost Optimization
- **Regular cost reviews**: Monitor and optimize costs
- **Use spot instances**: For non-critical workloads
- **Implement auto-scaling**: Automatically adjust resources
- **Schedule scaling**: Scale down during low-traffic periods

## Scaling Checklist

### Pre-Scaling Checklist
- [ ] Assess current performance metrics
- [ ] Identify scaling bottlenecks
- [ ] Plan scaling strategy
- [ ] Prepare monitoring and alerting
- [ ] Test scaling procedures
- [ ] Notify team of scaling activities

### During Scaling
- [ ] Monitor system performance
- [ ] Watch for errors or issues
- [ ] Verify auto-scaling behavior
- [ ] Check resource utilization
- [ ] Monitor user impact
- [ ] Document any issues

### Post-Scaling
- [ ] Verify system stability
- [ ] Confirm performance improvements
- [ ] Update capacity planning
- [ ] Review scaling effectiveness
- [ ] Update documentation
- [ ] Schedule follow-up review

---

**Document Information:**
- **Version**: 1.0
- **Last Updated**: January 2024
- **Next Review**: April 2024
- **Owner**: Performance Engineering Team
- **Approved By**: Engineering Manager

**Performance Targets:**
- **API Response Time**: < 500ms (95th percentile)
- **System Availability**: 99.9%
- **Auto-scaling Response**: < 3 minutes
- **Database Query Time**: < 100ms (average)