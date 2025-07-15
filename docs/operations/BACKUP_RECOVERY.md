# Backup and Recovery Procedures

## Overview

This document outlines the comprehensive backup and recovery procedures for the FreeAgentics system, including backup strategies, recovery procedures, and disaster recovery planning.

## Table of Contents

1. [Backup Strategy](#backup-strategy)
2. [Backup Types and Schedules](#backup-types-and-schedules)
3. [RTO/RPO Targets](#rtorpo-targets)
4. [Backup Procedures](#backup-procedures)
5. [Recovery Procedures](#recovery-procedures)
6. [Testing Procedures](#testing-procedures)
7. [Disaster Recovery](#disaster-recovery)
8. [Monitoring and Alerts](#monitoring-and-alerts)
9. [Compliance and Retention](#compliance-and-retention)

## Backup Strategy

### Overview
Our backup strategy follows the 3-2-1 rule:
- **3** copies of important data (1 primary + 2 backups)
- **2** different storage media types
- **1** offsite backup copy

### Infrastructure Components
- **PostgreSQL Database**: Primary data store
- **Redis Cache**: Session and cache data
- **Application Files**: Code, configuration, and static assets
- **SSL Certificates**: Security certificates and keys
- **Environment Configuration**: Production settings and secrets

## Backup Types and Schedules

### Database Backups (PostgreSQL)

#### Full Backups
- **Frequency**: Daily at 02:00 UTC
- **Retention**: 30 days
- **Location**: Primary: `/var/backups/freeagentics/daily/`
- **Offsite**: AWS S3 bucket `freeagentics-backups-prod`

#### Incremental Backups
- **Frequency**: Every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)
- **Retention**: 7 days
- **Method**: PostgreSQL WAL archiving
- **Location**: `/var/backups/freeagentics/incremental/`

#### Point-in-Time Recovery (PITR)
- **Method**: Continuous WAL archiving
- **Retention**: 7 days of WAL files
- **Recovery granularity**: Any point within the last 7 days

### Redis Backups

#### RDB Snapshots
- **Frequency**: Every 6 hours
- **Retention**: 48 hours
- **Location**: `/var/backups/freeagentics/redis/`

#### AOF (Append Only File)
- **Status**: Enabled
- **Sync Policy**: Every second (everysec)
- **Location**: `/data/redis/appendonly.aof`

### Application and Configuration Backups

#### Code Repository
- **Method**: Git repository with tags for releases
- **Backup**: Automated mirror to secondary Git server
- **Frequency**: On every push to main branch

#### Configuration Files
- **Frequency**: Daily
- **Items**:
  - Environment files (encrypted)
  - Nginx configurations
  - Docker configurations
  - SSL certificates
- **Location**: `/var/backups/freeagentics/config/`

### Backup Schedule Summary

| Component | Type | Frequency | Retention | Primary Location | Offsite |
|-----------|------|-----------|-----------|------------------|---------|
| PostgreSQL | Full | Daily 02:00 UTC | 30 days | Local disk | AWS S3 |
| PostgreSQL | Incremental | Every 6 hours | 7 days | Local disk | AWS S3 |
| PostgreSQL | WAL | Continuous | 7 days | Local disk | AWS S3 |
| Redis | RDB | Every 6 hours | 48 hours | Local disk | - |
| Redis | AOF | Continuous | Current | Local disk | - |
| Config | Full | Daily | 7 days | Local disk | Git |
| Code | Full | On push | Forever | Git primary | Git mirror |

## RTO/RPO Targets

### Recovery Time Objective (RTO)
- **Critical Systems**: 30 minutes
- **Full System**: 2 hours
- **Complete Infrastructure**: 4 hours

### Recovery Point Objective (RPO)
- **Database**: 15 minutes (with WAL archiving)
- **Redis Cache**: 1 hour (acceptable data loss)
- **Configuration**: 24 hours
- **Application Code**: 0 (version controlled)

### Service Level Targets

| Service | RTO | RPO | Priority |
|---------|-----|-----|----------|
| API Core | 30 min | 15 min | Critical |
| Database | 1 hour | 15 min | Critical |
| Cache | 2 hours | 1 hour | High |
| Web Frontend | 30 min | 0 | High |
| Monitoring | 2 hours | 1 hour | Medium |

## Backup Procedures

### Automated Database Backup Script

Located at: `/home/green/FreeAgentics/scripts/backup/database-backup.sh`

```bash
# Run daily full backup
./scripts/backup/database-backup.sh full

# Run incremental backup
./scripts/backup/database-backup.sh incremental

# Verify backup integrity
./scripts/backup/database-backup.sh verify <backup-file>
```

### Manual Backup Procedures

#### 1. Emergency Database Backup
```bash
# Create immediate backup
sudo -u postgres pg_dump freeagentics > /tmp/emergency-backup-$(date +%Y%m%d-%H%M%S).sql

# Compress and move to backup location
gzip /tmp/emergency-backup-*.sql
mv /tmp/emergency-backup-*.sql.gz /var/backups/freeagentics/emergency/
```

#### 2. Redis Backup
```bash
# Force Redis RDB snapshot
redis-cli -a $REDIS_PASSWORD BGSAVE

# Copy snapshot to backup location
cp /data/redis/dump.rdb /var/backups/freeagentics/redis/dump-$(date +%Y%m%d-%H%M%S).rdb
```

#### 3. Configuration Backup
```bash
# Run configuration backup script
./scripts/backup/config-backup.sh

# Verify backup
./scripts/backup/config-backup.sh verify
```

### Offsite Backup Sync

```bash
# Sync to AWS S3 (runs every 4 hours via cron)
./scripts/backup/offsite-sync.sh

# Manual sync for specific backup
./scripts/backup/offsite-sync.sh --file /path/to/backup.gz
```

## Recovery Procedures

### Database Recovery

#### From Full Backup
```bash
# Stop application services
docker-compose stop backend-prod web

# Restore database
./scripts/backup/database-restore.sh full /var/backups/freeagentics/daily/backup-20240115.sql.gz

# Restart services
docker-compose up -d backend-prod web
```

#### Point-in-Time Recovery
```bash
# Restore to specific timestamp
./scripts/backup/database-restore.sh pitr "2024-01-15 14:30:00 UTC"

# Verify recovery
./scripts/backup/database-restore.sh verify
```

### Redis Recovery

```bash
# Stop Redis
docker-compose stop redis

# Restore from RDB snapshot
./scripts/backup/redis-restore.sh /var/backups/freeagentics/redis/dump-20240115.rdb

# Start Redis
docker-compose up -d redis
```

### Application Recovery

```bash
# Rollback to previous version
./scripts/deployment/rollback.sh --version v1.2.3

# Restore from Git tag
git checkout v1.2.3
./scripts/deployment/deploy.sh
```

### Full System Recovery

For complete disaster recovery:

```bash
# 1. Restore infrastructure
./scripts/recovery/infrastructure-restore.sh

# 2. Restore database
./scripts/recovery/database-restore.sh --latest

# 3. Restore Redis
./scripts/recovery/redis-restore.sh --latest

# 4. Deploy application
./scripts/recovery/application-deploy.sh --version stable

# 5. Verify system health
./scripts/recovery/health-check.sh --comprehensive
```

## Testing Procedures

### Monthly Recovery Test

1. **Test Environment Setup**
   ```bash
   # Create test environment
   ./scripts/testing/create-test-env.sh
   ```

2. **Backup Restoration Test**
   ```bash
   # Test database restoration
   ./scripts/testing/test-db-restore.sh
   
   # Test Redis restoration
   ./scripts/testing/test-redis-restore.sh
   
   # Test configuration restoration
   ./scripts/testing/test-config-restore.sh
   ```

3. **Verification Steps**
   - Database integrity check
   - Application functionality test
   - Data consistency validation
   - Performance benchmarks

4. **Test Report Generation**
   ```bash
   ./scripts/testing/generate-recovery-report.sh
   ```

### Quarterly Disaster Recovery Drill

1. **Scenario Planning**
   - Complete data center failure
   - Database corruption
   - Ransomware attack
   - Human error (accidental deletion)

2. **Execution Steps**
   - Follow disaster recovery runbook
   - Document actual RTO/RPO
   - Identify improvement areas

3. **Post-Drill Actions**
   - Update procedures based on findings
   - Train team on identified gaps
   - Update automation scripts

## Disaster Recovery

### Disaster Scenarios and Procedures

#### 1. Database Corruption
```bash
# Immediate response
./scripts/recovery/database-corruption-recovery.sh

# Steps:
# 1. Isolate corrupted database
# 2. Restore from last known good backup
# 3. Apply WAL logs to minimize data loss
# 4. Verify data integrity
```

#### 2. Complete System Failure
```bash
# Full disaster recovery
./scripts/recovery/full-disaster-recovery.sh

# Steps:
# 1. Provision new infrastructure
# 2. Restore all components from offsite backups
# 3. Update DNS and load balancers
# 4. Verify system functionality
```

#### 3. Ransomware Attack
```bash
# Ransomware recovery
./scripts/recovery/ransomware-recovery.sh

# Steps:
# 1. Isolate affected systems
# 2. Assess backup integrity
# 3. Restore from clean backups
# 4. Implement additional security measures
```

### Recovery Priority Order

1. **Phase 1 - Critical Services (0-30 minutes)**
   - Database (read-only mode)
   - Core API endpoints
   - Authentication services

2. **Phase 2 - Full Services (30-120 minutes)**
   - Database (read-write mode)
   - All API endpoints
   - Web interface
   - Redis cache

3. **Phase 3 - Supporting Services (2-4 hours)**
   - Monitoring and alerting
   - Backup services
   - Analytics
   - Scheduled jobs

## Monitoring and Alerts

### Backup Monitoring

#### Automated Checks
- Backup completion status
- Backup file size validation
- Backup integrity verification
- Storage space monitoring

#### Alert Thresholds
- Failed backup: Immediate alert
- Backup size < 90% of average: Warning
- Storage space < 20%: Critical alert
- Offsite sync failure: High priority alert

### Recovery Testing Alerts
- Monthly test overdue: Warning
- Test failure: Critical alert
- RTO/RPO exceeded: High priority alert

## Compliance and Retention

### Data Retention Policy

| Data Type | Retention Period | Compliance Requirement |
|-----------|-----------------|------------------------|
| Database Backups | 30 days | GDPR Article 5(1)(e) |
| WAL Archives | 7 days | Business continuity |
| Redis Snapshots | 48 hours | Performance optimization |
| Audit Logs | 1 year | SOC 2 Type II |
| Configuration | 90 days | Change management |

### Compliance Checklist

- [ ] GDPR compliance for EU data
- [ ] Data encryption at rest
- [ ] Access control and audit trails
- [ ] Regular compliance audits
- [ ] Data deletion procedures

### Backup Security

1. **Encryption**
   - All backups encrypted with AES-256
   - Separate key management system
   - Key rotation every 90 days

2. **Access Control**
   - Role-based access to backup systems
   - Multi-factor authentication required
   - Audit logging for all access

3. **Testing**
   - Quarterly security assessment
   - Annual penetration testing
   - Regular access reviews

## Appendix

### A. Contact Information

| Role | Name | Contact | Escalation |
|------|------|---------|------------|
| DBA Lead | On-call | ops@freeagentics.io | +1-555-0100 |
| DevOps Lead | On-call | devops@freeagentics.io | +1-555-0101 |
| Security Lead | On-call | security@freeagentics.io | +1-555-0102 |

### B. Tool Locations

- Backup Scripts: `/home/green/FreeAgentics/scripts/backup/`
- Recovery Scripts: `/home/green/FreeAgentics/scripts/recovery/`
- Test Scripts: `/home/green/FreeAgentics/scripts/testing/`
- Documentation: `/home/green/FreeAgentics/docs/operations/`

### C. External Resources

- AWS S3 Console: https://console.aws.amazon.com/s3/
- Monitoring Dashboard: https://monitoring.freeagentics.io
- Runbook Wiki: https://wiki.freeagentics.io/runbooks

---

Last Updated: January 2024
Next Review: April 2024