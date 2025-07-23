# FreeAgentics Operational Runbook

## Overview

This runbook provides comprehensive operational procedures for managing, monitoring, and troubleshooting the FreeAgentics multi-agent AI platform in production environments.

## System Architecture Quick Reference

### Core Components
- **API Gateway**: Nginx reverse proxy with SSL termination
- **Application Server**: Python FastAPI with multi-agent coordination
- **Database**: PostgreSQL with Redis for caching
- **Monitoring**: Prometheus, Grafana, ELK stack
- **Security**: JWT authentication, RBAC, rate limiting

### Key Services
- **Agent Manager**: `/agents/agent_manager.py`
- **Coalition Coordinator**: `/agents/coalition_coordinator.py`
- **Authentication Service**: `/auth/security_implementation.py`
- **Security Monitoring**: `/observability/security_monitoring.py`
- **Rate Limiting**: `/api/middleware/rate_limiter.py`

## Daily Operations

### Morning Checklist

1. **System Health Check**
   ```bash
   # Check API health
   curl -s http://localhost:8000/health | jq

   # Check database connectivity
   psql -h localhost -U freeagentics -d freeagentics -c "SELECT 1;"

   # Check Redis connectivity
   redis-cli ping
   ```

2. **Review Overnight Logs**
   ```bash
   # Check error logs
   tail -100 /var/log/freeagentics/error.log

   # Check security audit logs
   tail -100 /var/log/freeagentics/security_audit.log

   # Check performance metrics
   curl -s http://localhost:9090/api/v1/query?query=up | jq
   ```

3. **Monitor Key Metrics**
   - CPU usage < 80%
   - Memory usage < 85%
   - Disk usage < 90%
   - Active agents count
   - Response times < 2s
   - Error rates < 1%

### Evening Checklist

1. **Performance Review**
   ```bash
   # Generate daily performance report
   python scripts/monitoring/generate_daily_report.py

   # Check slow queries
   psql -h localhost -U freeagentics -d freeagentics -c "
   SELECT query, mean_time, calls
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;"
   ```

2. **Security Review**
   ```bash
   # Check authentication failures
   grep "authentication_failed" /var/log/freeagentics/security_audit.log | wc -l

   # Check rate limiting violations
   grep "rate_limit_exceeded" /var/log/freeagentics/security_audit.log | wc -l

   # Run security validation
   python scripts/security/validate_security_posture.py
   ```

3. **Backup Verification**
   ```bash
   # Verify database backup
   ls -la /backups/postgresql/$(date +%Y%m%d)*

   # Test backup restoration (on staging)
   ./scripts/backup/test_backup_restoration.sh
   ```

## Deployment Procedures

### Pre-Deployment Checklist

1. **Code Quality Gates**
   ```bash
   # Run all tests
   pytest --cov=. --cov-report=html

   # Security tests
   pytest tests/security/ -v

   # Performance tests
   pytest tests/performance/ -v

   # Linting and formatting
   flake8 .
   black --check .
   mypy .
   ```

2. **Environment Preparation**
   ```bash
   # Update environment variables
   source .env.production

   # Check dependencies
   pip-audit

   # Validate configuration
   python scripts/validate_config.py
   ```

### Blue-Green Deployment

1. **Prepare Green Environment**
   ```bash
   # Deploy to green environment
   ./scripts/deployment/deploy-production.sh green

   # Run smoke tests
   ./scripts/deployment/smoke-tests.sh green

   # Verify health
   curl -s http://green.freeagentics.com/health
   ```

2. **Switch Traffic**
   ```bash
   # Update load balancer
   ./scripts/deployment/switch-traffic.sh green

   # Monitor during switch
   watch -n 1 'curl -s http://api.freeagentics.com/health | jq .status'
   ```

3. **Post-Deployment Verification**
   ```bash
   # Verify all endpoints
   ./scripts/deployment/verify-deployment.sh

   # Check error rates
   curl -s http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])
   ```

### Rollback Procedures

1. **Immediate Rollback**
   ```bash
   # Switch traffic back to blue
   ./scripts/deployment/switch-traffic.sh blue

   # Verify rollback
   curl -s http://api.freeagentics.com/health
   ```

2. **Database Rollback** (if needed)
   ```bash
   # Restore database from backup
   ./scripts/backup/restore-database.sh $(date -d "1 hour ago" +%Y%m%d_%H%M%S)

   # Verify data integrity
   python scripts/verify_data_integrity.py
   ```

## Monitoring and Alerting

### Key Metrics to Monitor

#### System Metrics
- **CPU Usage**: Target < 80%, Alert > 90%
- **Memory Usage**: Target < 85%, Alert > 95%
- **Disk Usage**: Target < 90%, Alert > 95%
- **Network I/O**: Monitor for unusual spikes

#### Application Metrics
- **Response Time**: Target < 2s, Alert > 5s
- **Error Rate**: Target < 1%, Alert > 5%
- **Active Agents**: Monitor for unusual patterns
- **Queue Length**: Target < 100, Alert > 500

#### Security Metrics
- **Authentication Failures**: Alert > 10/minute
- **Rate Limit Violations**: Alert > 50/minute
- **Security Events**: Monitor for patterns
- **SSL Certificate Expiry**: Alert 30 days before

### Alert Response Procedures

#### High CPU Usage
```bash
# Check top processes
top -p $(pgrep -f "python.*main.py")

# Check agent activity
curl -s http://localhost:8000/api/v1/system/metrics | jq .agent_count

# Scale horizontally if needed
kubectl scale deployment freeagentics-api --replicas=5
```

#### High Memory Usage
```bash
# Check memory usage by process
ps aux --sort=-%mem | head -20

# Check for memory leaks
python scripts/memory_profiler.py

# Restart if necessary
sudo systemctl restart freeagentics
```

#### Database Connection Issues
```bash
# Check database status
sudo systemctl status postgresql

# Check connection pool
psql -h localhost -U freeagentics -d freeagentics -c "SELECT * FROM pg_stat_activity;"

# Check disk space
df -h /var/lib/postgresql
```

#### Security Alerts
```bash
# Check security logs
tail -100 /var/log/freeagentics/security_audit.log

# Check authentication failures
grep "authentication_failed" /var/log/freeagentics/security_audit.log | tail -20

# Block malicious IPs if needed
sudo iptables -A INPUT -s <malicious_ip> -j DROP
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Agent Not Responding
```bash
# Check agent status
curl -s http://localhost:8000/api/v1/agents/<agent_id>/status

# Check agent logs
grep "agent_id:<agent_id>" /var/log/freeagentics/agents.log

# Restart agent if needed
curl -X POST http://localhost:8000/api/v1/agents/<agent_id>/restart
```

#### Slow Response Times
```bash
# Check database performance
psql -h localhost -U freeagentics -d freeagentics -c "
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;"

# Check Redis performance
redis-cli --latency-history

# Check thread pool utilization
curl -s http://localhost:8000/api/v1/system/metrics | jq .thread_pool_utilization
```

#### Authentication Issues
```bash
# Check JWT configuration
python -c "import jwt; print(jwt.get_unverified_header('TOKEN'))"

# Verify key rotation
ls -la /etc/freeagentics/keys/

# Check Redis for blacklisted tokens
redis-cli keys "blacklist:*"
```

#### Rate Limiting False Positives
```bash
# Check rate limiting configuration
cat /etc/freeagentics/rate_limiting.yaml

# Check Redis rate limiting data
redis-cli keys "rate_limit:*"

# Adjust limits if needed
vim /etc/freeagentics/rate_limiting.yaml
sudo systemctl reload freeagentics
```

### Performance Optimization

#### Database Optimization
```sql
-- Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;

-- Check for unused indexes
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0;

-- Analyze table statistics
ANALYZE;
```

#### Memory Optimization
```bash
# Check memory usage by component
python scripts/memory_profiler.py

# Optimize garbage collection
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# Adjust agent memory limits
vim /etc/freeagentics/agent_config.yaml
```

#### Thread Pool Optimization
```bash
# Check thread pool metrics
curl -s http://localhost:8000/api/v1/system/metrics | jq '.thread_pools'

# Adjust thread pool sizes
vim /etc/freeagentics/threading_config.yaml

# Restart service
sudo systemctl restart freeagentics
```

## Security Operations

### Security Incident Response

1. **Immediate Response**
   ```bash
   # Block suspicious IPs
   sudo iptables -A INPUT -s <malicious_ip> -j DROP

   # Check current attacks
   grep "security_event" /var/log/freeagentics/security_audit.log | tail -50

   # Scale down if under attack
   kubectl scale deployment freeagentics-api --replicas=1
   ```

2. **Investigation**
   ```bash
   # Generate security report
   python scripts/security/generate_incident_report.py

   # Check authentication patterns
   python scripts/security/analyze_auth_patterns.py

   # Review access logs
   grep "$(date +%Y-%m-%d)" /var/log/nginx/access.log | grep "40[0-9]\|50[0-9]"
   ```

3. **Recovery**
   ```bash
   # Update security rules
   vim /etc/freeagentics/security_rules.yaml

   # Rotate JWT keys
   python scripts/security/rotate_jwt_keys.py

   # Update rate limiting
   vim /etc/freeagentics/rate_limiting.yaml
   ```

### Security Maintenance

#### Weekly Security Tasks
```bash
# Update security dependencies
pip-audit --fix

# Run security tests
pytest tests/security/ -v

# Check SSL certificate status
openssl x509 -in /etc/ssl/certs/freeagentics.crt -text -noout

# Review security logs
python scripts/security/weekly_security_report.py
```

#### Monthly Security Review
```bash
# Full security audit
python scripts/security/comprehensive_audit.py

# Update security documentation
vim docs/security/SECURITY_PROCEDURES.md

# Test incident response procedures
./scripts/security/test_incident_response.sh

# Review and update security policies
vim /etc/freeagentics/security_policies.yaml
```

## Backup and Recovery

### Database Backup
```bash
# Daily backup
pg_dump -h localhost -U freeagentics -d freeagentics | \
  gzip > /backups/postgresql/freeagentics_$(date +%Y%m%d_%H%M%S).sql.gz

# Verify backup
gunzip -t /backups/postgresql/freeagentics_$(date +%Y%m%d_%H%M%S).sql.gz
```

### Application Backup
```bash
# Backup configuration
tar -czf /backups/config/freeagentics_config_$(date +%Y%m%d).tar.gz /etc/freeagentics/

# Backup logs
tar -czf /backups/logs/freeagentics_logs_$(date +%Y%m%d).tar.gz /var/log/freeagentics/
```

### Disaster Recovery

1. **Full System Recovery**
   ```bash
   # Restore from backup
   ./scripts/backup/full_system_restore.sh

   # Verify system integrity
   python scripts/verify_system_integrity.py

   # Update DNS if needed
   # (Update DNS records to point to recovery site)
   ```

2. **Database Recovery**
   ```bash
   # Stop application
   sudo systemctl stop freeagentics

   # Restore database
   gunzip -c /backups/postgresql/freeagentics_backup.sql.gz | \
     psql -h localhost -U freeagentics -d freeagentics

   # Start application
   sudo systemctl start freeagentics
   ```

## Performance Monitoring

### Key Performance Indicators

#### Response Time Monitoring
```bash
# Check API response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/health

# Check database response times
psql -h localhost -U freeagentics -d freeagentics -c "\timing on" -c "SELECT 1;"
```

#### Throughput Monitoring
```bash
# Check requests per second
curl -s http://localhost:9090/api/v1/query?query=rate(http_requests_total[1m])

# Check agent processing rate
curl -s http://localhost:8000/api/v1/system/metrics | jq .agent_processing_rate
```

#### Resource Utilization
```bash
# Check CPU utilization
sar -u 1 10

# Check memory utilization
free -h

# Check disk I/O
iostat -x 1 10
```

### Performance Tuning

#### Database Performance
```sql
-- Update table statistics
ANALYZE;

-- Check for long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';

-- Optimize configuration
-- Edit /etc/postgresql/postgresql.conf
-- shared_buffers = 256MB
-- effective_cache_size = 1GB
-- work_mem = 4MB
```

#### Application Performance
```bash
# Profile application
python -m cProfile -o profile.stats main.py

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Optimize thread pools
vim /etc/freeagentics/threading_config.yaml
```

## Maintenance Windows

### Scheduled Maintenance

#### Weekly Maintenance (Sundays 2:00 AM)
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Rotate logs
sudo logrotate -f /etc/logrotate.d/freeagentics

# Clean up old backups
find /backups -type f -mtime +30 -delete

# Restart services
sudo systemctl restart nginx
sudo systemctl restart freeagentics
```

#### Monthly Maintenance (First Sunday 2:00 AM)
```bash
# Update Python dependencies
pip install -r requirements.txt --upgrade

# Run database maintenance
psql -h localhost -U freeagentics -d freeagentics -c "VACUUM ANALYZE;"

# Update SSL certificates
certbot renew

# Security audit
python scripts/security/monthly_audit.py
```

### Emergency Maintenance

#### Critical Security Patch
```bash
# Apply security patch
git pull origin main
pip install -r requirements.txt

# Run security tests
pytest tests/security/ -v

# Deploy immediately
./scripts/deployment/emergency-deploy.sh
```

#### Performance Emergency
```bash
# Scale up resources
kubectl scale deployment freeagentics-api --replicas=10

# Enable emergency mode
export FREEAGENTICS_EMERGENCY_MODE=true

# Monitor closely
watch -n 10 'curl -s http://localhost:8000/api/v1/system/metrics | jq .response_time'
```

## Contact Information

### On-Call Rotation
- **Primary**: ops-primary@freeagentics.com
- **Secondary**: ops-secondary@freeagentics.com
- **Escalation**: ops-manager@freeagentics.com

### Emergency Contacts
- **Security Team**: security@freeagentics.com
- **Database Team**: dba@freeagentics.com
- **Infrastructure Team**: infra@freeagentics.com

### External Vendors
- **Cloud Provider**: [Provider Support]
- **CDN Provider**: [CDN Support]
- **Monitoring Service**: [Monitoring Support]

## Documentation References

- **Architecture Overview**: `/docs/ARCHITECTURE_OVERVIEW.md`
- **Security Documentation**: `/docs/security/`
- **API Documentation**: `/docs/api/`
- **Deployment Guide**: `/docs/production/DEPLOYMENT_GUIDE.md`
- **Monitoring Guide**: `/docs/monitoring/MONITORING_GUIDE.md`

---

**Document Version**: 1.0
**Last Updated**: January 16, 2025
**Next Review**: February 16, 2025
**Maintained By**: Operations Team

---

*This runbook is a living document. Update it as procedures change and new operational knowledge is gained.*
