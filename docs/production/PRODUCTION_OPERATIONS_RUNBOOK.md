# FreeAgentics Production Operations Runbook

## Overview

This runbook provides comprehensive operational procedures for managing FreeAgentics in production environments. It covers deployment, monitoring, troubleshooting, and emergency response procedures.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Deployment Procedures](#deployment-procedures)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Emergency Response](#emergency-response)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Performance Optimization](#performance-optimization)
8. [Security Operations](#security-operations)
9. [Backup and Recovery](#backup-and-recovery)
10. [Capacity Planning](#capacity-planning)

## System Architecture

### Production Environment Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                            │
│                       (Nginx + SSL)                             │
└─────────────────────────┬───────────────────────────────────────┘
                         │
┌─────────────────────────┴───────────────────────────────────────┐
│                     Service Mesh                                │
│                    (Istio Gateway)                              │
└─────────────┬───────────────────────────────────┬───────────────┘
             │                                   │
┌─────────────┴─────────────┐         ┌─────────────┴─────────────┐
│       Frontend            │         │       Backend             │
│    (Next.js + Node.js)    │         │    (FastAPI + Python)     │
│                           │         │                           │
│  - Static Assets          │         │  - API Endpoints          │
│  - Server-Side Rendering  │         │  - Agent Management       │
│  - Client-Side Hydration  │         │  - Coalition Logic        │
└───────────────────────────┘         └─────────────┬─────────────┘
                                                   │
                                      ┌─────────────┴─────────────┐
                                      │                           │
                            ┌─────────┴─────────┐    ┌─────────────┴─────────┐
                            │   Database        │    │      Cache           │
                            │ (PostgreSQL 15)   │    │   (Redis 7)          │
                            │                   │    │                      │
                            │ - User Data       │    │ - Session Store      │
                            │ - Agent State     │    │ - API Cache          │
                            │ - Coalition Data  │    │ - Rate Limiting      │
                            └───────────────────┘    └──────────────────────┘
```

### Key Components

#### Application Layer
- **Frontend**: Next.js application serving the user interface
- **Backend**: FastAPI application providing REST API and WebSocket endpoints
- **Service Mesh**: Istio for traffic management, security, and observability

#### Data Layer
- **PostgreSQL**: Primary database for persistent data
- **Redis**: Cache and session store
- **Object Storage**: Static assets and file uploads (S3-compatible)

#### Infrastructure Layer
- **Kubernetes**: Container orchestration platform
- **Docker**: Container runtime
- **Nginx**: Reverse proxy and load balancer
- **Istio**: Service mesh for advanced traffic management

#### Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notification
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation

## Deployment Procedures

### Standard Deployment Process

#### 1. Pre-Deployment Checklist

```bash
# Check system health
curl -f https://freeagentics.com/health
curl -f https://freeagentics.com/api/v1/health

# Verify monitoring is operational
curl -f https://freeagentics.com/prometheus/-/healthy
curl -f https://freeagentics.com/grafana/api/health

# Check database connectivity
kubectl exec -n freeagentics-prod deployment/backend -- python -c "
from database.session import get_db_session
import asyncio
async def test():
    async with get_db_session() as session:
        result = await session.execute('SELECT 1')
        print('Database OK:', result.scalar())
asyncio.run(test())
"

# Verify adequate resources
kubectl top nodes
kubectl top pods -n freeagentics-prod
```

#### 2. Deployment Execution

```bash
# Using zero-downtime deployment script
./scripts/deployment/zero-downtime-deploy.sh \
  --version v1.2.3 \
  --strategy blue-green \
  --env production

# Alternative: Using Kubernetes deployment script
./k8s/deploy-k8s-enhanced.sh \
  --version v1.2.3 \
  --strategy blue-green \
  --namespace freeagentics-prod
```

#### 3. Post-Deployment Verification

```bash
# Verify deployment status
kubectl get pods -n freeagentics-prod
kubectl get deployments -n freeagentics-prod

# Check application health
curl -f https://freeagentics.com/health
curl -f https://freeagentics.com/api/v1/health

# Verify database migrations
kubectl exec -n freeagentics-prod deployment/backend -- alembic current

# Check metrics endpoint
curl -f https://freeagentics.com/metrics

# Verify critical user flows
# (Run automated smoke tests)
```

### Rollback Procedures

#### 1. Immediate Rollback

```bash
# Automatic rollback (if deployment script detects failure)
# The zero-downtime deployment script includes automatic rollback

# Manual rollback using Kubernetes
kubectl rollout undo deployment/backend -n freeagentics-prod
kubectl rollout undo deployment/frontend -n freeagentics-prod

# Verify rollback status
kubectl rollout status deployment/backend -n freeagentics-prod
kubectl rollout status deployment/frontend -n freeagentics-prod
```

#### 2. Database Rollback

```bash
# Check current database revision
kubectl exec -n freeagentics-prod deployment/backend -- alembic current

# Rollback database migration (if needed)
kubectl exec -n freeagentics-prod deployment/backend -- alembic downgrade -1

# Verify database state
kubectl exec -n freeagentics-prod deployment/backend -- alembic current
```

#### 3. Traffic Rollback

```bash
# If using Istio service mesh
kubectl patch virtualservice freeagentics-vs -n freeagentics-prod --type=merge -p '
{
  "spec": {
    "http": [
      {
        "route": [
          {
            "destination": {
              "host": "backend-stable"
            },
            "weight": 100
          }
        ]
      }
    ]
  }
}
'
```

## Monitoring and Alerting

### Key Metrics to Monitor

#### Application Metrics
- **Request Rate**: `rate(http_requests_total[5m])`
- **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m])`
- **Response Time**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- **Active Connections**: `freeagentics_active_connections`

#### Infrastructure Metrics
- **CPU Usage**: `(1 - rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100`
- **Memory Usage**: `(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100`
- **Disk Usage**: `(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100`
- **Network I/O**: `rate(node_network_receive_bytes_total[5m])`

#### Business Metrics
- **User Registrations**: `increase(freeagentics_user_registrations_total[5m])`
- **Agent Interactions**: `increase(freeagentics_agent_interactions_total[5m])`
- **Coalition Formations**: `increase(freeagentics_coalition_formations_total[5m])`

### Alert Rules

#### Critical Alerts
```yaml
# High Error Rate
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value | humanizePercentage }}"

# High Response Time
- alert: HighResponseTime
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High response time detected"
    description: "95th percentile response time is {{ $value | humanizeDuration }}"

# Database Connection Issues
- alert: DatabaseConnectionIssues
  expr: up{job="postgres"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Database connection issues"
    description: "PostgreSQL database is unreachable"
```

#### Warning Alerts
```yaml
# High Memory Usage
- alert: HighMemoryUsage
  expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 80
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High memory usage"
    description: "Memory usage is {{ $value | humanizePercentage }}"

# High CPU Usage
- alert: HighCPUUsage
  expr: (1 - rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100 > 80
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High CPU usage"
    description: "CPU usage is {{ $value | humanizePercentage }}"
```

### Grafana Dashboards

#### System Overview Dashboard
- **Panels**: Request rate, error rate, response time, active users
- **Time Range**: Last 24 hours
- **Refresh**: 30 seconds

#### Application Performance Dashboard
- **Panels**: Endpoint performance, database queries, cache hit rate
- **Time Range**: Last 6 hours
- **Refresh**: 15 seconds

#### Infrastructure Dashboard
- **Panels**: CPU, memory, disk, network, Kubernetes resources
- **Time Range**: Last 12 hours
- **Refresh**: 1 minute

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. High Response Times

**Symptoms:**
- 95th percentile response time > 500ms
- User complaints about slow page loads
- Timeout errors in logs

**Investigation Steps:**
```bash
# Check current response times
curl -s https://freeagentics.com/prometheus/api/v1/query?query=histogram_quantile\(0.95,rate\(http_request_duration_seconds_bucket\[5m\]\)\)

# Check database performance
kubectl exec -n freeagentics-prod deployment/backend -- psql -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
"

# Check for slow queries
kubectl logs -n freeagentics-prod deployment/backend | grep -i "slow"

# Check system resources
kubectl top nodes
kubectl top pods -n freeagentics-prod
```

**Solutions:**
1. **Database Optimization:**
   ```bash
   # Add missing indexes
   kubectl exec -n freeagentics-prod deployment/backend -- python -c "
   from database.query_optimizer import suggest_indexes
   suggest_indexes()
   "
   
   # Analyze query plans
   kubectl exec -n freeagentics-prod deployment/backend -- psql -c "
   EXPLAIN ANALYZE SELECT * FROM agents WHERE status = 'active';
   "
   ```

2. **Scale Application:**
   ```bash
   # Horizontal scaling
   kubectl scale deployment/backend --replicas=6 -n freeagentics-prod
   
   # Vertical scaling (update resource limits)
   kubectl patch deployment backend -n freeagentics-prod -p '
   {
     "spec": {
       "template": {
         "spec": {
           "containers": [
             {
               "name": "backend",
               "resources": {
                 "requests": {
                   "memory": "1Gi",
                   "cpu": "500m"
                 },
                 "limits": {
                   "memory": "2Gi",
                   "cpu": "1000m"
                 }
               }
             }
           ]
         }
       }
     }
   }
   '
   ```

3. **Enable Caching:**
   ```bash
   # Check cache hit rate
   kubectl exec -n freeagentics-prod deployment/redis -- redis-cli info stats | grep keyspace
   
   # Clear cache if needed
   kubectl exec -n freeagentics-prod deployment/redis -- redis-cli flushdb
   ```

#### 2. Database Connection Issues

**Symptoms:**
- "Connection refused" errors
- Database timeout errors
- Application unable to start

**Investigation Steps:**
```bash
# Check database pod status
kubectl get pods -n freeagentics-prod -l app=postgres

# Check database logs
kubectl logs -n freeagentics-prod deployment/postgres

# Test database connectivity
kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "SELECT 1;"

# Check connection pool status
kubectl exec -n freeagentics-prod deployment/backend -- python -c "
from database.connection_manager import get_connection_pool_status
print(get_connection_pool_status())
"
```

**Solutions:**
1. **Restart Database:**
   ```bash
   kubectl rollout restart deployment/postgres -n freeagentics-prod
   kubectl rollout status deployment/postgres -n freeagentics-prod
   ```

2. **Check Database Resources:**
   ```bash
   kubectl describe pod -n freeagentics-prod -l app=postgres
   kubectl top pod -n freeagentics-prod -l app=postgres
   ```

3. **Verify Database Configuration:**
   ```bash
   kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "
   SELECT name, setting FROM pg_settings 
   WHERE name IN ('max_connections', 'shared_buffers', 'work_mem');
   "
   ```

#### 3. Memory Issues

**Symptoms:**
- Out of Memory (OOM) kills
- High memory usage alerts
- Application crashes

**Investigation Steps:**
```bash
# Check memory usage
kubectl top pods -n freeagentics-prod
kubectl describe nodes

# Check for OOM kills
kubectl get events -n freeagentics-prod --field-selector reason=OOMKilling

# Memory profiling
kubectl exec -n freeagentics-prod deployment/backend -- python -c "
import tracemalloc
tracemalloc.start()
# Application code here
current, peak = tracemalloc.get_traced_memory()
print(f'Current memory usage: {current / 1024 / 1024:.2f} MB')
print(f'Peak memory usage: {peak / 1024 / 1024:.2f} MB')
"
```

**Solutions:**
1. **Increase Memory Limits:**
   ```bash
   kubectl patch deployment backend -n freeagentics-prod -p '
   {
     "spec": {
       "template": {
         "spec": {
           "containers": [
             {
               "name": "backend",
               "resources": {
                 "limits": {
                   "memory": "2Gi"
                 }
               }
             }
           ]
         }
       }
     }
   }
   '
   ```

2. **Enable Memory Optimization:**
   ```bash
   # Enable Python memory optimization
   kubectl set env deployment/backend PYTHONMALLOC=malloc -n freeagentics-prod
   
   # Enable agent memory optimization
   kubectl set env deployment/backend AGENT_MEMORY_OPTIMIZATION=true -n freeagentics-prod
   ```

#### 4. Certificate Issues

**Symptoms:**
- SSL/TLS errors
- Certificate expiration warnings
- Browser security warnings

**Investigation Steps:**
```bash
# Check certificate expiration
echo | openssl s_client -servername freeagentics.com -connect freeagentics.com:443 2>/dev/null | openssl x509 -noout -dates

# Check certificate chain
echo | openssl s_client -servername freeagentics.com -connect freeagentics.com:443 2>/dev/null | openssl x509 -noout -text

# Check cert-manager status (if using)
kubectl get certificates -n freeagentics-prod
kubectl describe certificate freeagentics-tls -n freeagentics-prod
```

**Solutions:**
1. **Renew Certificate:**
   ```bash
   # If using cert-manager
   kubectl annotate certificate freeagentics-tls -n freeagentics-prod cert-manager.io/force-renew=true
   
   # If using Let's Encrypt manually
   certbot renew --dry-run
   certbot renew
   ```

2. **Update Certificate:**
   ```bash
   # Update Kubernetes secret
   kubectl create secret tls freeagentics-tls -n freeagentics-prod \
     --cert=/path/to/cert.pem \
     --key=/path/to/key.pem \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

## Emergency Response

### Incident Response Procedures

#### 1. Immediate Response (0-5 minutes)

**Acknowledge the Alert:**
```bash
# Check system status
curl -f https://freeagentics.com/health

# Check critical metrics
kubectl get pods -n freeagentics-prod
kubectl get nodes
```

**Assess Impact:**
- Determine if this is a partial or complete outage
- Check if users are affected
- Identify which components are failing

**Initial Communication:**
- Notify stakeholders via Slack/Teams
- Update status page if applicable
- Gather on-call team if needed

#### 2. Investigation (5-15 minutes)

**Gather Information:**
```bash
# Check application logs
kubectl logs -n freeagentics-prod deployment/backend --tail=100

# Check system events
kubectl get events -n freeagentics-prod --sort-by=.metadata.creationTimestamp

# Check resource usage
kubectl top pods -n freeagentics-prod
kubectl top nodes
```

**Identify Root Cause:**
- Recent deployments or changes
- Resource constraints
- External dependencies
- Security incidents

#### 3. Mitigation (15-30 minutes)

**Immediate Actions:**
```bash
# Scale up if resource constrained
kubectl scale deployment/backend --replicas=10 -n freeagentics-prod

# Rollback if deployment related
kubectl rollout undo deployment/backend -n freeagentics-prod

# Restart if application issue
kubectl rollout restart deployment/backend -n freeagentics-prod
```

**Traffic Management:**
```bash
# Route traffic away from failing pods
kubectl patch service backend -n freeagentics-prod -p '
{
  "spec": {
    "selector": {
      "app": "backend",
      "health": "healthy"
    }
  }
}
'

# Enable maintenance mode if needed
kubectl patch configmap nginx-config -n freeagentics-prod -p '
{
  "data": {
    "maintenance": "true"
  }
}
'
```

#### 4. Recovery (30+ minutes)

**Restore Service:**
- Implement permanent fix
- Verify functionality
- Gradually restore traffic
- Monitor for issues

**Post-Incident:**
- Document timeline
- Identify lessons learned
- Update procedures
- Schedule post-mortem

### Disaster Recovery

#### 1. Database Recovery

**Backup Restoration:**
```bash
# List available backups
aws s3 ls s3://freeagentics-backups/database/

# Restore from backup
kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "
DROP DATABASE freeagentics;
CREATE DATABASE freeagentics;
"

# Restore data
kubectl exec -i -n freeagentics-prod deployment/postgres -- psql freeagentics < backup.sql
```

**Point-in-Time Recovery:**
```bash
# If using WAL-G or similar
kubectl exec -n freeagentics-prod deployment/postgres -- wal-g backup-fetch /var/lib/postgresql/data LATEST
```

#### 2. Complete System Recovery

**Infrastructure Recovery:**
```bash
# Restore Kubernetes manifests
kubectl apply -f k8s/

# Restore secrets
kubectl apply -f backup/secrets.yaml

# Restore persistent volumes
kubectl apply -f backup/persistent-volumes.yaml
```

**Application Recovery:**
```bash
# Deploy application
./k8s/deploy-k8s-enhanced.sh --version latest

# Verify functionality
./scripts/deployment/smoke-tests.sh
```

## Maintenance Procedures

### Scheduled Maintenance

#### 1. System Updates

**Monthly System Updates:**
```bash
# Update Kubernetes nodes
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
# Perform OS updates on node
kubectl uncordon <node-name>

# Update container images
kubectl set image deployment/backend backend=freeagentics/backend:latest -n freeagentics-prod
kubectl set image deployment/frontend frontend=freeagentics/frontend:latest -n freeagentics-prod
```

#### 2. Database Maintenance

**Weekly Database Maintenance:**
```bash
# Update statistics
kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "ANALYZE;"

# Vacuum database
kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "VACUUM FULL;"

# Check database health
kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "
SELECT 
  schemaname,
  tablename,
  n_tup_ins,
  n_tup_upd,
  n_tup_del,
  n_dead_tup
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000;
"
```

#### 3. Log Rotation

**Daily Log Rotation:**
```bash
# Rotate application logs
kubectl exec -n freeagentics-prod deployment/backend -- logrotate /etc/logrotate.d/freeagentics

# Archive old logs
kubectl exec -n freeagentics-prod deployment/backend -- tar -czf /var/log/archive/app-$(date +%Y%m%d).tar.gz /var/log/freeagentics/
```

### Capacity Planning

#### 1. Resource Monitoring

**Weekly Capacity Review:**
```bash
# Generate capacity report
kubectl top nodes
kubectl top pods --all-namespaces
kubectl get nodes -o custom-columns=NAME:.metadata.name,CPU:.status.capacity.cpu,MEMORY:.status.capacity.memory

# Check resource requests vs limits
kubectl describe nodes | grep -A 5 "Allocated resources"
```

#### 2. Scaling Decisions

**Scaling Triggers:**
- CPU usage > 70% for 10 minutes
- Memory usage > 80% for 5 minutes
- Response time > 500ms for 5 minutes
- Error rate > 1% for 2 minutes

**Scaling Actions:**
```bash
# Horizontal Pod Autoscaler
kubectl autoscale deployment backend --cpu-percent=70 --min=3 --max=20 -n freeagentics-prod

# Vertical Pod Autoscaler
kubectl apply -f k8s/autoscaling-enhanced.yaml
```

### Performance Optimization

#### 1. Application Performance

**Database Optimization:**
```bash
# Identify slow queries
kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
WHERE mean_time > 100
ORDER BY total_time DESC
LIMIT 10;
"

# Add indexes for slow queries
kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "
CREATE INDEX CONCURRENTLY idx_agents_status ON agents(status);
CREATE INDEX CONCURRENTLY idx_coalitions_created_at ON coalitions(created_at);
"
```

**Cache Optimization:**
```bash
# Check cache hit rate
kubectl exec -n freeagentics-prod deployment/redis -- redis-cli info stats | grep keyspace_hits

# Optimize cache configuration
kubectl patch configmap redis-config -n freeagentics-prod -p '
{
  "data": {
    "maxmemory": "2gb",
    "maxmemory-policy": "allkeys-lru"
  }
}
'
```

#### 2. Infrastructure Performance

**Network Optimization:**
```bash
# Check network latency
kubectl exec -n freeagentics-prod deployment/backend -- ping -c 5 postgres

# Optimize connection pooling
kubectl patch configmap backend-config -n freeagentics-prod -p '
{
  "data": {
    "DATABASE_POOL_SIZE": "20",
    "DATABASE_MAX_OVERFLOW": "10"
  }
}
'
```

### Security Operations

#### 1. Security Monitoring

**Daily Security Checks:**
```bash
# Check for security alerts
kubectl get events -n freeagentics-prod --field-selector type=Warning

# Review access logs
kubectl logs -n freeagentics-prod deployment/backend | grep -i "unauthorized\|forbidden\|attack"

# Check certificate status
kubectl get certificates -n freeagentics-prod
```

#### 2. Security Updates

**Monthly Security Updates:**
```bash
# Update base images
docker pull python:3.11-slim
docker pull node:18-alpine
docker pull nginx:alpine

# Scan for vulnerabilities
trivy image freeagentics/backend:latest
trivy image freeagentics/frontend:latest

# Update dependencies
pip-audit requirements.txt
npm audit
```

### Backup and Recovery

#### 1. Backup Procedures

**Daily Automated Backups:**
```bash
# Database backup
kubectl exec -n freeagentics-prod deployment/postgres -- pg_dump -U freeagentics freeagentics | gzip > /backup/db-$(date +%Y%m%d).sql.gz

# Application data backup
kubectl exec -n freeagentics-prod deployment/backend -- tar -czf /backup/app-data-$(date +%Y%m%d).tar.gz /app/data

# Configuration backup
kubectl get configmaps -n freeagentics-prod -o yaml > /backup/configmaps-$(date +%Y%m%d).yaml
kubectl get secrets -n freeagentics-prod -o yaml > /backup/secrets-$(date +%Y%m%d).yaml
```

#### 2. Recovery Testing

**Monthly Recovery Tests:**
```bash
# Test database recovery
kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "CREATE DATABASE test_restore;"
kubectl exec -i -n freeagentics-prod deployment/postgres -- psql test_restore < /backup/db-latest.sql
kubectl exec -n freeagentics-prod deployment/postgres -- psql -c "DROP DATABASE test_restore;"

# Test application recovery
kubectl apply -f backup/test-namespace.yaml
kubectl apply -f backup/configmaps-latest.yaml -n test
kubectl apply -f backup/secrets-latest.yaml -n test
```

## Contact Information

### Emergency Contacts

- **Primary On-Call**: +1-XXX-XXX-XXXX
- **Secondary On-Call**: +1-XXX-XXX-XXXX
- **Engineering Manager**: +1-XXX-XXX-XXXX
- **Platform Team Lead**: +1-XXX-XXX-XXXX

### Communication Channels

- **Slack**: #freeagentics-alerts
- **Teams**: FreeAgentics Operations
- **Email**: ops@freeagentics.com
- **Status Page**: https://status.freeagentics.com

### Escalation Matrix

| Severity | Response Time | Notification |
|----------|---------------|--------------|
| Critical | 5 minutes | On-call + Manager |
| High | 15 minutes | On-call |
| Medium | 1 hour | On-call (business hours) |
| Low | Next business day | Email |

## Appendices

### A. Useful Commands

```bash
# Quick health check
kubectl get pods -n freeagentics-prod
curl -f https://freeagentics.com/health

# Check logs
kubectl logs -n freeagentics-prod deployment/backend --tail=50
kubectl logs -n freeagentics-prod deployment/frontend --tail=50

# Scale applications
kubectl scale deployment/backend --replicas=5 -n freeagentics-prod
kubectl scale deployment/frontend --replicas=3 -n freeagentics-prod

# Check resource usage
kubectl top pods -n freeagentics-prod
kubectl top nodes

# Port forward for debugging
kubectl port-forward -n freeagentics-prod deployment/backend 8000:8000
kubectl port-forward -n freeagentics-prod deployment/postgres 5432:5432
```

### B. Configuration Templates

See separate configuration files in the `config/` directory:
- `prometheus.yml` - Prometheus configuration
- `grafana-datasources.yml` - Grafana datasources
- `alertmanager.yml` - Alert routing configuration
- `istio-gateway.yml` - Istio gateway configuration

### C. Monitoring Queries

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# CPU usage
rate(container_cpu_usage_seconds_total[5m])

# Memory usage
container_memory_usage_bytes / container_spec_memory_limit_bytes
```

---

**Document Version**: 1.0.0
**Last Updated**: 2024-01-15
**Next Review**: 2024-02-15
**Owner**: Platform Team
**Reviewers**: Engineering Team, Security Team