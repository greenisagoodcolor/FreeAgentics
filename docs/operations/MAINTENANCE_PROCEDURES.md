# Maintenance Procedures

## Overview

This document outlines regular maintenance procedures for the FreeAgentics system, including database maintenance, log management, certificate renewal, dependency updates, and scheduled maintenance windows.

## Table of Contents

1. [Maintenance Schedule Overview](#maintenance-schedule-overview)
1. [Database Maintenance](#database-maintenance)
1. [Log Rotation and Management](#log-rotation-and-management)
1. [Certificate Management](#certificate-management)
1. [Dependency Updates](#dependency-updates)
1. [System Health Checks](#system-health-checks)
1. [Performance Optimization](#performance-optimization)
1. [Maintenance Windows](#maintenance-windows)
1. [Emergency Maintenance](#emergency-maintenance)
1. [Maintenance Automation](#maintenance-automation)

## Maintenance Schedule Overview

### Daily Tasks

- Database transaction log cleanup
- Log file rotation
- Health check monitoring
- Backup verification

### Weekly Tasks

- Database statistics update
- Cache optimization
- Security scan
- Performance metrics review

### Monthly Tasks

- Database full maintenance
- SSL certificate check
- Dependency security audit
- Storage cleanup

### Quarterly Tasks

- Major dependency updates
- Infrastructure review
- Security audit
- Performance baseline update

## Database Maintenance

### PostgreSQL Maintenance

#### Daily Maintenance

```bash
# Run via cron at 03:00 UTC daily
/usr/local/bin/database-daily-maintenance.sh

# Tasks performed:
# - VACUUM on high-activity tables
# - Update table statistics
# - Check for bloat
# - Archive old WAL files
```

#### Weekly Maintenance

```bash
# Run Sunday 04:00 UTC
/usr/local/bin/database-weekly-maintenance.sh

# Tasks performed:
# - VACUUM ANALYZE on all tables
# - REINDEX on frequently updated indexes
# - Check and alert on unused indexes
# - Generate performance report
```

#### Monthly Full Maintenance

```bash
# Run first Sunday of month at 02:00 UTC
/usr/local/bin/database-monthly-maintenance.sh

# Tasks performed:
# - VACUUM FULL on selected tables
# - REINDEX DATABASE
# - Update all statistics
# - Analyze query performance
# - Clean up orphaned data
```

### Database Health Monitoring

```bash
# Check database health
./scripts/maintenance/check-db-health.sh

# Monitor long-running queries
./scripts/maintenance/monitor-queries.sh

# Check table bloat
./scripts/maintenance/check-bloat.sh
```

### Common Database Issues and Solutions

| Issue | Detection | Solution |
|-------|-----------|----------|
| Table bloat | Size > 2x expected | VACUUM FULL or pg_repack |
| Slow queries | Duration > 1s | Analyze and optimize |
| Lock contention | Waiting queries | Review transaction logic |
| Connection exhaustion | Connection count > 80% | Increase pool size or optimize |

## Log Rotation and Management

### Application Logs

#### Log Rotation Configuration

```bash
# /etc/logrotate.d/freeagentics
/var/log/freeagentics/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 freeagentics freeagentics
    sharedscripts
    postrotate
        docker exec freeagentics-backend kill -USR1 1
    endscript
}
```

#### Log Management Tasks

1. **Daily Log Rotation**

   ```bash
   # Automatic via logrotate
   logrotate -f /etc/logrotate.d/freeagentics
   ```

1. **Log Archival**

   ```bash
   # Archive logs older than 30 days
   ./scripts/maintenance/archive-logs.sh --days 30
   ```

1. **Log Analysis**

   ```bash
   # Generate daily error report
   ./scripts/maintenance/analyze-logs.sh --report error

   # Performance analysis
   ./scripts/maintenance/analyze-logs.sh --report performance
   ```

### System Logs

```bash
# Clean system logs
./scripts/maintenance/clean-system-logs.sh

# Archive audit logs
./scripts/maintenance/archive-audit-logs.sh
```

## Certificate Management

### SSL Certificate Monitoring

```bash
# Check certificate expiration
./scripts/maintenance/check-certificates.sh

# Sample output:
# Domain: api.freeagentics.io
# Issuer: Let's Encrypt
# Expires: 2024-03-15 (45 days)
# Status: OK
```

### Certificate Renewal Process

#### Automated Renewal (Let's Encrypt)

```bash
# Runs via cron every day at 02:30 UTC
/usr/local/bin/certbot renew --quiet

# Post-renewal hook
--deploy-hook "docker exec freeagentics-nginx nginx -s reload"
```

#### Manual Renewal Process

```bash
# 1. Generate new certificate
./scripts/maintenance/renew-certificate.sh --domain api.freeagentics.io

# 2. Verify new certificate
./scripts/maintenance/verify-certificate.sh --cert /etc/letsencrypt/live/api.freeagentics.io/fullchain.pem

# 3. Deploy certificate
./scripts/maintenance/deploy-certificate.sh

# 4. Reload services
docker-compose exec nginx nginx -s reload
```

### Certificate Backup

```bash
# Backup all certificates
./scripts/maintenance/backup-certificates.sh

# Restore certificates
./scripts/maintenance/restore-certificates.sh --date 20240115
```

## Dependency Updates

### Security Updates

#### Daily Security Check

```bash
# Check for security vulnerabilities
./scripts/maintenance/security-check.sh

# Python dependencies
pip-audit --desc

# Node.js dependencies
npm audit

# Docker images
docker scout cves
```

#### Weekly Dependency Update Process

1. **Review Security Advisories**

   ```bash
   ./scripts/maintenance/check-advisories.sh
   ```

1. **Test Updates in Staging**

   ```bash
   # Create update branch
   git checkout -b deps/weekly-update-$(date +%Y%m%d)

   # Update dependencies
   ./scripts/maintenance/update-dependencies.sh --security-only

   # Run tests
   make test
   ```

1. **Deploy Updates**

   ```bash
   # If tests pass, deploy to production
   ./scripts/deployment/deploy.sh --version deps/weekly-update
   ```

### Major Version Updates

#### Quarterly Update Process

1. **Planning Phase**

   - Review changelog for breaking changes
   - Update test suite for new features
   - Plan rollback strategy

1. **Testing Phase**

   ```bash
   # Create dedicated test environment
   ./scripts/maintenance/create-update-env.sh

   # Apply updates
   ./scripts/maintenance/major-update.sh --component python

   # Run comprehensive tests
   ./scripts/testing/comprehensive-test.sh
   ```

1. **Deployment Phase**

   - Schedule maintenance window
   - Notify users
   - Deploy with rollback capability

### Dependency Update Schedule

| Component | Security Updates | Minor Updates | Major Updates |
|-----------|-----------------|---------------|---------------|
| Python packages | Daily check, immediate | Weekly | Quarterly |
| Node packages | Daily check, immediate | Weekly | Quarterly |
| Docker base images | Weekly | Monthly | Quarterly |
| System packages | Daily check, immediate | Monthly | Semi-annual |

## System Health Checks

### Automated Health Monitoring

```bash
# Runs every 5 minutes via cron
/usr/local/bin/health-check.sh

# Checks:
# - Service availability
# - Response times
# - Error rates
# - Resource usage
```

### Manual Health Verification

```bash
# Comprehensive system check
./scripts/maintenance/system-health-check.sh --full

# Component-specific checks
./scripts/maintenance/check-api-health.sh
./scripts/maintenance/check-database-health.sh
./scripts/maintenance/check-redis-health.sh
```

### Performance Metrics Collection

```bash
# Collect performance baseline
./scripts/maintenance/collect-metrics.sh --duration 1h

# Generate performance report
./scripts/maintenance/performance-report.sh --period weekly
```

## Performance Optimization

### Database Performance Tuning

#### Weekly Analysis

```bash
# Analyze slow queries
./scripts/maintenance/analyze-slow-queries.sh

# Optimize indexes
./scripts/maintenance/optimize-indexes.sh

# Update table statistics
./scripts/maintenance/update-statistics.sh
```

#### Monthly Optimization

```bash
# Full performance analysis
./scripts/maintenance/performance-analysis.sh --comprehensive

# Apply recommended optimizations
./scripts/maintenance/apply-optimizations.sh
```

### Application Performance

#### Cache Optimization

```bash
# Analyze cache hit rates
./scripts/maintenance/analyze-cache.sh

# Clear stale cache entries
./scripts/maintenance/clean-cache.sh

# Optimize cache configuration
./scripts/maintenance/optimize-cache-config.sh
```

#### Resource Optimization

```bash
# Check resource usage trends
./scripts/maintenance/resource-trends.sh --period 30d

# Optimize container resources
./scripts/maintenance/optimize-resources.sh
```

## Maintenance Windows

### Scheduled Maintenance Windows

#### Regular Windows

- **Weekly**: Sunday 03:00-04:00 UTC (low-impact tasks)
- **Monthly**: First Sunday 02:00-05:00 UTC (database maintenance)
- **Quarterly**: Announced 2 weeks in advance (major updates)

#### Maintenance Window Process

1. **Pre-Maintenance (T-24 hours)**

   ```bash
   # Send notification
   ./scripts/maintenance/notify-users.sh --window "2024-01-21 02:00 UTC"

   # Prepare maintenance environment
   ./scripts/maintenance/prepare-maintenance.sh
   ```

1. **During Maintenance**

   ```bash
   # Enable maintenance mode
   ./scripts/maintenance/enable-maintenance-mode.sh

   # Perform maintenance tasks
   ./scripts/maintenance/run-maintenance.sh --plan monthly

   # Verify system health
   ./scripts/maintenance/verify-health.sh
   ```

1. **Post-Maintenance**

   ```bash
   # Disable maintenance mode
   ./scripts/maintenance/disable-maintenance-mode.sh

   # Send completion notification
   ./scripts/maintenance/notify-completion.sh

   # Monitor for issues
   ./scripts/maintenance/post-maintenance-monitor.sh --duration 1h
   ```

### Maintenance Mode Configuration

```nginx
# Maintenance mode configuration
location / {
    if (-f /var/www/maintenance.flag) {
        return 503;
    }
    # Normal processing
}

error_page 503 @maintenance;
location @maintenance {
    root /var/www/maintenance;
    try_files /index.html =503;
}
```

## Emergency Maintenance

### Emergency Response Procedures

1. **Issue Detection**

   ```bash
   # Automated detection via monitoring
   # Manual detection via:
   ./scripts/maintenance/emergency-check.sh
   ```

1. **Initial Response**

   ```bash
   # Assess severity
   ./scripts/maintenance/assess-issue.sh

   # Enable emergency maintenance if needed
   ./scripts/maintenance/emergency-maintenance.sh --enable
   ```

1. **Resolution Process**

   - Identify root cause
   - Implement fix
   - Test resolution
   - Deploy fix
   - Monitor results

1. **Post-Incident**

   ```bash
   # Generate incident report
   ./scripts/maintenance/incident-report.sh --incident-id EM-2024-001

   # Update runbooks
   ./scripts/maintenance/update-runbooks.sh
   ```

### Emergency Contacts

| Role | Primary | Backup | Escalation |
|------|---------|--------|------------|
| DBA | On-call DBA | DBA Team Lead | CTO |
| DevOps | On-call DevOps | DevOps Lead | CTO |
| Security | Security Lead | CISO | CEO |

## Maintenance Automation

### Automated Maintenance Scripts

Located in `/home/green/FreeAgentics/scripts/maintenance/`

#### Core Scripts

- `auto-maintenance.sh` - Main automation orchestrator
- `health-monitor.sh` - Continuous health monitoring
- `auto-optimize.sh` - Automatic optimization
- `alert-handler.sh` - Maintenance alert management

### Automation Configuration

```yaml
# /etc/freeagentics/maintenance.yaml
maintenance:
  automated_tasks:
    - name: daily_vacuum
      schedule: "0 3 * * *"
      script: database-daily-maintenance.sh
      alert_on_failure: true

    - name: log_rotation
      schedule: "0 0 * * *"
      script: rotate-logs.sh
      alert_on_failure: false

    - name: certificate_check
      schedule: "0 2 * * *"
      script: check-certificates.sh
      alert_on_failure: true
      alert_threshold_days: 30

  notifications:
    slack_webhook: ${SLACK_WEBHOOK}
    email_list: ops@freeagentics.io

  thresholds:
    disk_usage_percent: 80
    memory_usage_percent: 85
    cpu_usage_percent: 90
    database_connections_percent: 80
```

### Monitoring Integration

```bash
# Prometheus metrics for maintenance
freeagentics_maintenance_last_run{task="vacuum"} 1705312800
freeagentics_maintenance_duration_seconds{task="vacuum"} 245
freeagentics_maintenance_status{task="vacuum"} 1
```

## Appendix

### A. Maintenance Checklist Templates

#### Daily Checklist

- [ ] Verify all backups completed successfully
- [ ] Check system health dashboard
- [ ] Review error logs
- [ ] Verify certificate expiration > 30 days
- [ ] Check disk space utilization
- [ ] Monitor database connections

#### Weekly Checklist

- [ ] Run database VACUUM ANALYZE
- [ ] Update security patches
- [ ] Review performance metrics
- [ ] Clean up old logs
- [ ] Verify monitoring alerts
- [ ] Test backup restoration

#### Monthly Checklist

- [ ] Full database maintenance
- [ ] Security audit
- [ ] Dependency updates
- [ ] Performance optimization
- [ ] Disaster recovery test
- [ ] Update documentation

### B. Troubleshooting Guide

| Issue | Symptoms | Quick Fix | Full Resolution |
|-------|----------|-----------|-----------------|
| High DB CPU | Slow queries, high load | Kill long queries | Optimize queries |
| Memory pressure | OOM errors, swapping | Restart services | Tune memory settings |
| Disk full | Write errors | Clean logs/temp | Add storage |
| Connection limit | Connection refused | Increase pool | Optimize connections |

### C. Maintenance Resources

- Scripts: `/home/green/FreeAgentics/scripts/maintenance/`
- Logs: `/var/log/freeagentics/maintenance/`
- Documentation: `/home/green/FreeAgentics/docs/operations/`
- Monitoring: https://monitoring.freeagentics.io/maintenance

______________________________________________________________________

Last Updated: January 2024
Next Review: April 2024
