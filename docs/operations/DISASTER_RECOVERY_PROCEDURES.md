# FreeAgentics Disaster Recovery Procedures

## Table of Contents
1. [Overview](#overview)
2. [Disaster Scenarios](#disaster-scenarios)
3. [Recovery Procedures](#recovery-procedures)
4. [3-2-1 Backup Strategy](#3-2-1-backup-strategy)
5. [Automated Backup System](#automated-backup-system)
6. [Testing and Validation](#testing-and-validation)
7. [Emergency Contacts](#emergency-contacts)

## Overview

This document outlines the comprehensive disaster recovery procedures for the FreeAgentics production environment. Our disaster recovery strategy is built on the 3-2-1 backup principle and includes automated backup systems, multiple recovery scenarios, and regular testing procedures.

### Recovery Objectives
- **RTO (Recovery Time Objective)**: 30 minutes
- **RPO (Recovery Point Objective)**: 15 minutes
- **Data Retention**: 365 days for archives
- **Verification Frequency**: Weekly automated tests

## Disaster Scenarios

### 1. Database Corruption
**Impact**: PostgreSQL database becomes corrupted or inconsistent

**Recovery Steps**:
```bash
# 1. Stop application services
docker-compose -f docker-compose.production.yml stop backend

# 2. Verify database corruption
docker exec freeagentics-postgres pg_isready
psql -h localhost -U freeagentics -d freeagentics -c "SELECT 1;"

# 3. Restore from latest backup
./scripts/backup/disaster-recovery.sh database-corruption --point-in-time "YYYY-MM-DD HH:MM:SS"

# 4. Verify restoration
./scripts/backup/verify-database.sh

# 5. Restart services
docker-compose -f docker-compose.production.yml up -d
```

### 2. Ransomware Attack
**Impact**: System files encrypted by ransomware

**Recovery Steps**:
```bash
# 1. Isolate affected systems
./scripts/security/isolate-system.sh

# 2. Assess impact
./scripts/backup/disaster-recovery.sh ransomware --dry-run

# 3. Restore from clean backup (pre-infection)
./scripts/backup/disaster-recovery.sh ransomware --backup-source s3 --force

# 4. Apply security patches
./scripts/security/apply-emergency-patches.sh

# 5. Restore services with enhanced monitoring
./scripts/backup/restore-with-monitoring.sh
```

### 3. Complete System Failure
**Impact**: Total infrastructure failure

**Recovery Steps**:
```bash
# 1. Provision new infrastructure
terraform apply -var-file=disaster-recovery.tfvars

# 2. Restore all components
./scripts/backup/disaster-recovery.sh full-system --backup-source s3

# 3. Verify all services
./scripts/health/comprehensive-health-check.sh

# 4. Update DNS and load balancers
./scripts/infrastructure/update-dns-dr.sh
```

### 4. Data Center Loss
**Impact**: Complete data center unavailable

**Recovery Steps**:
```bash
# 1. Activate DR site
./scripts/dr/activate-dr-site.sh

# 2. Restore from offsite backups
./scripts/backup/disaster-recovery.sh data-center-loss --recovery-target dr-site

# 3. Verify DR site functionality
./scripts/dr/verify-dr-site.sh

# 4. Update routing to DR site
./scripts/dr/update-routing.sh
```

### 5. Human Error (Accidental Deletion)
**Impact**: Critical data accidentally deleted

**Recovery Steps**:
```bash
# 1. Identify deleted data
./scripts/backup/identify-missing-data.sh

# 2. Restore specific data
./scripts/backup/disaster-recovery.sh human-error --selective-restore

# 3. Verify restored data
./scripts/backup/verify-restoration.sh

# 4. Implement additional safeguards
./scripts/security/enable-deletion-protection.sh
```

## 3-2-1 Backup Strategy

Our backup strategy follows the industry-standard 3-2-1 rule:

### 3 Copies of Data
1. **Primary**: Live production data
2. **Local Backup**: On-premises backup server/NAS
3. **Offsite Backup**: Cloud storage (S3 + Azure/GCP)

### 2 Different Media Types
1. **Local Storage**: High-speed SSD array for recent backups
2. **Cloud Storage**: Object storage with lifecycle policies

### 1 Offsite Copy
- **Primary**: AWS S3 with cross-region replication
- **Secondary**: Azure Blob Storage or Google Cloud Storage
- **Tertiary**: Backblaze B2 for long-term archive

## Automated Backup System

### Components

#### 1. Database Backup
```yaml
Type: PostgreSQL Custom Format
Schedule: 
  - Full: Daily at 2:00 AM
  - Incremental: Every 6 hours
Retention:
  - Local: 7 days
  - S3 Standard: 30 days
  - S3 Glacier: 90 days
  - S3 Deep Archive: 365 days
```

#### 2. Application State Backup
```yaml
Components:
  - Agent states and beliefs
  - Coalition configurations
  - Active session data
  - System metrics
Schedule: Every 6 hours
Format: Compressed JSON archives
```

#### 3. Knowledge Graph Backup
```yaml
Tables:
  - knowledge_graph_nodes
  - knowledge_graph_edges
  - knowledge_graph_metadata
Method: Table-specific SQL dumps
Schedule: Daily with database backup
```

#### 4. Configuration Backup
```yaml
Files:
  - Environment variables (.env files)
  - Docker configurations
  - Nginx configurations
  - SSL certificates
  - System configurations
Schedule: On change + weekly full backup
Encryption: AES-256
```

#### 5. Redis Backup
```yaml
Method: RDB snapshots + AOF files
Schedule: Every hour
Retention: 48 hours local, 7 days offsite
```

### Backup Verification

#### Automated Verification
```bash
# Run automated verification
python scripts/backup/automated-backup-system.py --test-restore

# Verification includes:
# - Checksum validation
# - File integrity checks
# - Test database restore
# - Application state validation
```

#### Manual Verification
```bash
# Verify specific backup
./scripts/backup/verify-backup.sh --backup-id full_20240718_020000

# Comprehensive verification
./scripts/backup/comprehensive-verification.sh
```

### Monitoring and Alerts

#### Metrics Tracked
- Backup size and duration
- Success/failure rates
- Storage utilization
- Restore test results
- Offsite sync status

#### Alert Conditions
- Backup failure
- Backup size anomaly (>20% change)
- Storage space low (<10GB)
- Verification failure
- Offsite sync failure

#### Notification Channels
1. **Slack**: Real-time notifications
2. **Email**: Detailed reports
3. **PagerDuty**: Critical failures
4. **Grafana**: Visual dashboards

## Recovery Procedures

### Standard Recovery Process

#### 1. Assessment Phase
```bash
# Run disaster assessment
./scripts/backup/assess-disaster.sh

# Check system status
./scripts/health/system-status.sh

# Identify recovery point
./scripts/backup/list-available-backups.sh
```

#### 2. Preparation Phase
```bash
# Create recovery environment
./scripts/backup/prepare-recovery.sh

# Verify backup integrity
./scripts/backup/verify-backup-integrity.sh --backup-id <ID>

# Download offsite backups if needed
./scripts/backup/download-offsite-backup.sh --source s3 --backup-id <ID>
```

#### 3. Recovery Phase
```bash
# Stop affected services
docker-compose -f docker-compose.production.yml stop

# Restore database
./scripts/backup/restore-database.sh --backup-id <ID>

# Restore application state
./scripts/backup/restore-app-state.sh --backup-id <ID>

# Restore configurations
./scripts/backup/restore-config.sh --backup-id <ID>

# Restore Redis if needed
./scripts/backup/restore-redis.sh --backup-id <ID>
```

#### 4. Validation Phase
```bash
# Run integrity checks
./scripts/backup/post-restore-validation.sh

# Test application functionality
./scripts/test/smoke-tests.sh

# Verify data consistency
./scripts/backup/verify-data-consistency.sh
```

#### 5. Cutover Phase
```bash
# Update DNS if needed
./scripts/infrastructure/update-dns.sh

# Start services
docker-compose -f docker-compose.production.yml up -d

# Monitor startup
./scripts/monitoring/watch-startup.sh

# Run final validation
./scripts/test/production-validation.sh
```

### Quick Recovery Commands

#### Database Only
```bash
# Restore database from latest backup
./scripts/backup/quick-restore-db.sh
```

#### Redis Only
```bash
# Restore Redis from latest snapshot
./scripts/backup/quick-restore-redis.sh
```

#### Configuration Only
```bash
# Restore configuration from latest backup
./scripts/backup/quick-restore-config.sh
```

#### Full System
```bash
# Complete system restore
./scripts/backup/full-system-restore.sh --auto-select-latest
```

## Testing and Validation

### Weekly Automated Tests
```yaml
Schedule: Every Sunday at 5:00 AM
Tests:
  - Backup integrity verification
  - Test restore to staging environment
  - Recovery time measurement
  - Data consistency validation
  - Failover simulation
```

### Monthly Manual Tests
```yaml
Schedule: First Monday of each month
Tests:
  - Full disaster recovery drill
  - Cross-region failover
  - Team communication test
  - Documentation review
  - Tool functionality verification
```

### Quarterly Full DR Exercise
```yaml
Schedule: Quarterly
Scope:
  - Complete system failure simulation
  - Multi-team coordination
  - Customer communication procedures
  - Post-mortem and improvements
```

## Backup Storage Locations

### Local Storage
```
Path: /var/backups/freeagentics/
Structure:
  ├── daily/          # Daily full backups
  ├── incremental/    # Incremental backups
  ├── redis/          # Redis snapshots
  ├── config/         # Configuration backups
  ├── knowledge_graph/# Knowledge graph exports
  ├── app_state/      # Application state backups
  ├── logs/           # Backup logs
  └── metadata/       # Backup metadata
```

### Primary Offsite (AWS S3)
```
Bucket: freeagentics-backups-prod
Region: us-east-1
Replication: us-west-2
Structure: Same as local
```

### Secondary Offsite
```
Provider: Azure Blob Storage / Google Cloud Storage
Container: freeagentics-backups
Region: East US 2 / us-central1
Redundancy: Geo-redundant
```

### Archive Storage
```
Provider: AWS S3 Glacier / Backblaze B2
Transition: After 90 days
Retrieval Time: 1-12 hours
```

## Recovery Tools

### Automated Backup System
```bash
# Run full backup
python scripts/backup/automated-backup-system.py --run-now

# Run as daemon
python scripts/backup/automated-backup-system.py --daemon

# Test restore
python scripts/backup/automated-backup-system.py --test-restore

# Cleanup old backups
python scripts/backup/automated-backup-system.py --cleanup
```

### Disaster Recovery Script
```bash
# General usage
./scripts/backup/disaster-recovery.sh <scenario> [options]

# Examples
./scripts/backup/disaster-recovery.sh full-system --backup-source s3
./scripts/backup/disaster-recovery.sh database-corruption --point-in-time "2024-07-18 14:30:00"
./scripts/backup/disaster-recovery.sh ransomware --dry-run
```

### Verification Tools
```bash
# Verify backup integrity
./scripts/backup/verify-backup.sh --backup-id <ID>

# Comprehensive health check
./scripts/health/comprehensive-health-check.sh

# Data consistency check
./scripts/backup/verify-data-consistency.sh
```

## Emergency Contacts

### Primary Contacts
- **Operations Lead**: ops@freeagentics.io
- **CTO**: cto@freeagentics.io
- **24/7 Hotline**: +1-555-0100

### Escalation Path
1. On-call Engineer (PagerDuty)
2. Operations Lead
3. Engineering Manager
4. CTO

### External Support
- **AWS Support**: Premium Support Plan
- **Database Consultant**: db-support@consultant.com
- **Security Team**: security@freeagentics.io

## Appendix

### Configuration Template
See `/scripts/backup/backup-config.env.template` for complete configuration options.

### Backup Schedule Summary
```
Daily (2:00 AM):
  - Full database backup
  - Knowledge graph export
  - Configuration backup

Every 6 hours:
  - Incremental database backup
  - Application state snapshot

Hourly:
  - Redis RDB snapshot

Every 15 minutes:
  - Transaction log backup

Weekly:
  - Full system backup
  - Offsite sync verification
  - Automated restore test

Monthly:
  - Archive old backups
  - Storage optimization
  - DR drill
```

### Recovery Checklist
- [ ] Assess disaster impact
- [ ] Notify stakeholders
- [ ] Identify recovery point
- [ ] Prepare recovery environment
- [ ] Restore data
- [ ] Validate restoration
- [ ] Update DNS/routing
- [ ] Monitor services
- [ ] Document incident
- [ ] Post-mortem review