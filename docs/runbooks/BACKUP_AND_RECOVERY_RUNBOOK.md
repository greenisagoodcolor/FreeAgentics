# Backup and Recovery Runbook

## Overview

This runbook provides comprehensive procedures for backup and recovery operations for the FreeAgentics system, including database backups, application backups, disaster recovery, and business continuity procedures.

## Table of Contents

1. [Backup Strategy](#backup-strategy)
2. [Database Backup Procedures](#database-backup-procedures)
3. [Application Backup Procedures](#application-backup-procedures)
4. [System Recovery Procedures](#system-recovery-procedures)
5. [Disaster Recovery](#disaster-recovery)
6. [Business Continuity](#business-continuity)
7. [Backup Testing and Validation](#backup-testing-and-validation)
8. [Troubleshooting](#troubleshooting)

## Backup Strategy

### 1. Recovery Objectives

#### Recovery Time Objectives (RTO)
- **Database**: 30 minutes
- **Application Services**: 15 minutes
- **Full System**: 1 hour
- **Disaster Recovery**: 4 hours

#### Recovery Point Objectives (RPO)
- **Critical Data**: 15 minutes
- **Application Data**: 1 hour
- **System Configuration**: 4 hours
- **Log Data**: 24 hours

### 2. Backup Types and Schedule

#### Backup Types
- **Full Backup**: Complete system backup
- **Incremental Backup**: Changes since last backup
- **Differential Backup**: Changes since last full backup
- **Point-in-Time Backup**: Specific timestamp recovery
- **Continuous Backup**: Real-time data protection

#### Backup Schedule
```bash
# Daily full database backup
0 2 * * * /scripts/backup/daily-db-backup.sh

# Hourly incremental backup
0 * * * * /scripts/backup/hourly-incremental.sh

# Weekly full system backup
0 1 * * 0 /scripts/backup/weekly-full-backup.sh

# Monthly archive backup
0 0 1 * * /scripts/backup/monthly-archive.sh
```

### 3. Backup Storage

#### Storage Locations
- **Primary**: Local high-speed storage
- **Secondary**: Network-attached storage (NAS)
- **Tertiary**: Cloud storage (AWS S3/Azure Blob)
- **Offsite**: Geographic backup location

#### Retention Policy
```yaml
backup_retention:
  daily_backups: 30 days
  weekly_backups: 12 weeks
  monthly_backups: 12 months
  yearly_backups: 7 years
  compliance_backups: per_regulation
```

## Database Backup Procedures

### 1. PostgreSQL Database Backup

#### Full Database Backup
```bash
# Create full database backup
./scripts/backup/db/postgres-full-backup.sh

# Detailed backup procedure
export BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
export BACKUP_DIR="/backup/postgres/${BACKUP_DATE}"
export DB_NAME="freeagentics"
export DB_USER="postgres"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Perform backup
pg_dump -h localhost -U ${DB_USER} -d ${DB_NAME} \
  --format=custom \
  --compress=9 \
  --verbose \
  --file=${BACKUP_DIR}/freeagentics_${BACKUP_DATE}.dump

# Verify backup
pg_restore --list ${BACKUP_DIR}/freeagentics_${BACKUP_DATE}.dump > \
  ${BACKUP_DIR}/backup_contents.txt

# Calculate checksums
sha256sum ${BACKUP_DIR}/freeagentics_${BACKUP_DATE}.dump > \
  ${BACKUP_DIR}/checksums.txt
```

#### Incremental Database Backup
```bash
# WAL-based incremental backup
./scripts/backup/db/postgres-incremental-backup.sh

# Archive WAL files
export WAL_ARCHIVE_DIR="/backup/postgres/wal"
export BACKUP_DATE=$(date +%Y%m%d_%H%M%S)

# Archive WAL files
find ${POSTGRES_DATA_DIR}/pg_wal -name "*.backup" -type f \
  -exec cp {} ${WAL_ARCHIVE_DIR}/ \;

# Create backup label
echo "Incremental backup: ${BACKUP_DATE}" > \
  ${WAL_ARCHIVE_DIR}/backup_label_${BACKUP_DATE}

# Compress WAL files
gzip ${WAL_ARCHIVE_DIR}/*_${BACKUP_DATE}
```

#### Point-in-Time Recovery Backup
```bash
# Enable point-in-time recovery
./scripts/backup/db/postgres-pitr-setup.sh

# Configure PostgreSQL for PITR
cat >> postgresql.conf << EOF
archive_mode = on
archive_command = 'cp %p /backup/postgres/wal/%f'
wal_level = replica
max_wal_senders = 3
EOF

# Create base backup for PITR
pg_basebackup -h localhost -U ${DB_USER} \
  --format=tar \
  --gzip \
  --progress \
  --verbose \
  --checkpoint=fast \
  --wal-method=stream \
  --directory=/backup/postgres/basebackup/$(date +%Y%m%d_%H%M%S)
```

### 2. Redis Backup

#### Redis Data Backup
```bash
# Create Redis backup
./scripts/backup/redis/redis-backup.sh

# Perform Redis backup
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export BACKUP_DIR="/backup/redis/$(date +%Y%m%d_%H%M%S)"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Save Redis data
redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} BGSAVE

# Wait for background save to complete
while [ $(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} LASTSAVE) -eq \
       $(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} LASTSAVE) ]; do
  sleep 1
done

# Copy RDB file
cp /var/lib/redis/dump.rdb ${BACKUP_DIR}/

# Backup Redis configuration
cp /etc/redis/redis.conf ${BACKUP_DIR}/

# Create backup metadata
cat > ${BACKUP_DIR}/backup_info.txt << EOF
Backup Date: $(date)
Redis Version: $(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} INFO server | grep redis_version)
Database Size: $(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} DBSIZE)
EOF
```

### 3. Database Backup Verification

#### Backup Integrity Check
```bash
# Verify PostgreSQL backup integrity
./scripts/backup/verify/postgres-verify.sh --backup-file ${BACKUP_FILE}

# Restore test procedure
export TEST_DB="freeagentics_test_$(date +%Y%m%d_%H%M%S)"

# Create test database
createdb -h localhost -U ${DB_USER} ${TEST_DB}

# Restore backup to test database
pg_restore -h localhost -U ${DB_USER} -d ${TEST_DB} \
  --verbose \
  --clean \
  --if-exists \
  ${BACKUP_FILE}

# Verify data integrity
psql -h localhost -U ${DB_USER} -d ${TEST_DB} -c "
SELECT 
  schemaname,
  tablename,
  n_tup_ins,
  n_tup_upd,
  n_tup_del
FROM pg_stat_user_tables
ORDER BY schemaname, tablename;"

# Clean up test database
dropdb -h localhost -U ${DB_USER} ${TEST_DB}
```

## Application Backup Procedures

### 1. Application Code Backup

#### Source Code Backup
```bash
# Backup application source code
./scripts/backup/app/source-code-backup.sh

# Create code backup
export CODE_BACKUP_DIR="/backup/code/$(date +%Y%m%d_%H%M%S)"
export REPO_URL="https://github.com/company/freeagentics.git"

# Create backup directory
mkdir -p ${CODE_BACKUP_DIR}

# Clone repository
git clone --mirror ${REPO_URL} ${CODE_BACKUP_DIR}/freeagentics.git

# Create archive
tar -czf ${CODE_BACKUP_DIR}/source_code_$(date +%Y%m%d_%H%M%S).tar.gz \
  -C ${CODE_BACKUP_DIR} freeagentics.git

# Backup build artifacts
docker save freeagentics/api:latest > \
  ${CODE_BACKUP_DIR}/api_image_$(date +%Y%m%d_%H%M%S).tar

docker save freeagentics/web:latest > \
  ${CODE_BACKUP_DIR}/web_image_$(date +%Y%m%d_%H%M%S).tar
```

### 2. Configuration Backup

#### System Configuration Backup
```bash
# Backup system configuration
./scripts/backup/config/system-config-backup.sh

# Configuration backup locations
export CONFIG_BACKUP_DIR="/backup/config/$(date +%Y%m%d_%H%M%S)"

# Create backup directory
mkdir -p ${CONFIG_BACKUP_DIR}

# Backup Docker configuration
cp -r /opt/freeagentics/docker-compose.yml ${CONFIG_BACKUP_DIR}/
cp -r /opt/freeagentics/.env ${CONFIG_BACKUP_DIR}/

# Backup NGINX configuration
cp -r /etc/nginx ${CONFIG_BACKUP_DIR}/

# Backup SSL certificates
cp -r /etc/letsencrypt ${CONFIG_BACKUP_DIR}/

# Backup monitoring configuration
cp -r /etc/prometheus ${CONFIG_BACKUP_DIR}/
cp -r /etc/grafana ${CONFIG_BACKUP_DIR}/

# Backup secrets
./scripts/backup/config/backup-secrets.sh --destination ${CONFIG_BACKUP_DIR}
```

#### Application Configuration Backup
```bash
# Backup application configuration
./scripts/backup/config/app-config-backup.sh

# Application-specific configurations
export APP_CONFIG_DIR="/backup/app-config/$(date +%Y%m%d_%H%M%S)"

# Create backup directory
mkdir -p ${APP_CONFIG_DIR}

# Backup application settings
kubectl get configmaps -o yaml > ${APP_CONFIG_DIR}/configmaps.yaml
kubectl get secrets -o yaml > ${APP_CONFIG_DIR}/secrets.yaml

# Backup Kubernetes manifests
cp -r /opt/freeagentics/k8s ${APP_CONFIG_DIR}/

# Backup environment-specific configurations
cp -r /opt/freeagentics/config ${APP_CONFIG_DIR}/
```

### 3. User Data Backup

#### User-Generated Content Backup
```bash
# Backup user data
./scripts/backup/data/user-data-backup.sh

# User data backup procedure
export USER_DATA_DIR="/backup/user-data/$(date +%Y%m%d_%H%M%S)"

# Create backup directory
mkdir -p ${USER_DATA_DIR}

# Backup file uploads
rsync -av --progress /opt/freeagentics/uploads/ ${USER_DATA_DIR}/uploads/

# Backup user profiles
pg_dump -h localhost -U postgres -d freeagentics \
  --table=user_profiles \
  --table=user_settings \
  --format=custom \
  --file=${USER_DATA_DIR}/user_data.dump

# Backup session data
redis-cli --rdb ${USER_DATA_DIR}/sessions.rdb

# Create backup manifest
cat > ${USER_DATA_DIR}/manifest.txt << EOF
Backup Date: $(date)
Files Backed Up: $(find ${USER_DATA_DIR} -type f | wc -l)
Total Size: $(du -sh ${USER_DATA_DIR} | cut -f1)
EOF
```

## System Recovery Procedures

### 1. Database Recovery

#### PostgreSQL Recovery
```bash
# Restore PostgreSQL database
./scripts/recovery/db/postgres-restore.sh --backup-file ${BACKUP_FILE}

# Database recovery procedure
export BACKUP_FILE="/backup/postgres/20240115_020000/freeagentics_20240115_020000.dump"
export DB_NAME="freeagentics"
export DB_USER="postgres"

# Stop application services
docker-compose stop api workers

# Drop and recreate database
dropdb -h localhost -U ${DB_USER} ${DB_NAME}
createdb -h localhost -U ${DB_USER} ${DB_NAME}

# Restore database from backup
pg_restore -h localhost -U ${DB_USER} -d ${DB_NAME} \
  --verbose \
  --clean \
  --if-exists \
  --jobs=4 \
  ${BACKUP_FILE}

# Verify database integrity
psql -h localhost -U ${DB_USER} -d ${DB_NAME} -c "
SELECT pg_database_size('${DB_NAME}');
SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';
"

# Restart application services
docker-compose start api workers
```

#### Point-in-Time Recovery
```bash
# Perform point-in-time recovery
./scripts/recovery/db/postgres-pitr-restore.sh \
  --target-time "2024-01-15 14:30:00"

# PITR recovery procedure
export TARGET_TIME="2024-01-15 14:30:00"
export BASE_BACKUP_DIR="/backup/postgres/basebackup/20240115_020000"
export WAL_ARCHIVE_DIR="/backup/postgres/wal"

# Stop PostgreSQL
systemctl stop postgresql

# Clear data directory
rm -rf /var/lib/postgresql/data/*

# Restore base backup
tar -xzf ${BASE_BACKUP_DIR}/base.tar.gz -C /var/lib/postgresql/data/

# Create recovery configuration
cat > /var/lib/postgresql/data/recovery.conf << EOF
restore_command = 'cp ${WAL_ARCHIVE_DIR}/%f %p'
recovery_target_time = '${TARGET_TIME}'
recovery_target_action = 'promote'
EOF

# Start PostgreSQL
systemctl start postgresql

# Monitor recovery progress
tail -f /var/log/postgresql/postgresql.log
```

### 2. Application Recovery

#### Container Recovery
```bash
# Restore application containers
./scripts/recovery/app/container-restore.sh

# Container recovery procedure
export BACKUP_DATE="20240115_020000"
export IMAGE_BACKUP_DIR="/backup/code/${BACKUP_DATE}"

# Load container images
docker load < ${IMAGE_BACKUP_DIR}/api_image_${BACKUP_DATE}.tar
docker load < ${IMAGE_BACKUP_DIR}/web_image_${BACKUP_DATE}.tar

# Restore configuration
cp ${IMAGE_BACKUP_DIR}/docker-compose.yml /opt/freeagentics/
cp ${IMAGE_BACKUP_DIR}/.env /opt/freeagentics/

# Restart services
cd /opt/freeagentics
docker-compose down
docker-compose up -d

# Verify services
docker-compose ps
./scripts/recovery/verify/verify-services.sh
```

#### Configuration Recovery
```bash
# Restore system configuration
./scripts/recovery/config/config-restore.sh --backup-date ${BACKUP_DATE}

# Configuration recovery procedure
export CONFIG_BACKUP_DIR="/backup/config/${BACKUP_DATE}"

# Restore NGINX configuration
cp -r ${CONFIG_BACKUP_DIR}/nginx /etc/

# Restore SSL certificates
cp -r ${CONFIG_BACKUP_DIR}/letsencrypt /etc/

# Restore monitoring configuration
cp -r ${CONFIG_BACKUP_DIR}/prometheus /etc/
cp -r ${CONFIG_BACKUP_DIR}/grafana /etc/

# Reload services
systemctl reload nginx
systemctl restart prometheus
systemctl restart grafana-server
```

### 3. Data Recovery

#### User Data Recovery
```bash
# Restore user data
./scripts/recovery/data/user-data-restore.sh --backup-date ${BACKUP_DATE}

# User data recovery procedure
export USER_DATA_BACKUP_DIR="/backup/user-data/${BACKUP_DATE}"

# Restore file uploads
rsync -av --progress ${USER_DATA_BACKUP_DIR}/uploads/ /opt/freeagentics/uploads/

# Restore user database data
pg_restore -h localhost -U postgres -d freeagentics \
  --verbose \
  --clean \
  --if-exists \
  ${USER_DATA_BACKUP_DIR}/user_data.dump

# Restore session data
redis-cli --rdb ${USER_DATA_BACKUP_DIR}/sessions.rdb

# Verify data integrity
./scripts/recovery/verify/verify-user-data.sh
```

## Disaster Recovery

### 1. Disaster Recovery Planning

#### Disaster Scenarios
- **Data Center Failure**: Complete facility unavailability
- **Hardware Failure**: Server or storage system failure
- **Network Failure**: Connectivity loss
- **Cyber Attack**: Ransomware or data breach
- **Natural Disaster**: Fire, flood, earthquake
- **Human Error**: Accidental deletion or misconfiguration

#### DR Site Configuration
```yaml
disaster_recovery:
  primary_site:
    location: "Primary Data Center"
    capacity: "100%"
    services: ["api", "web", "database", "monitoring"]
    
  secondary_site:
    location: "Secondary Data Center"
    capacity: "80%"
    services: ["api", "web", "database"]
    
  cloud_backup:
    provider: "AWS"
    region: "us-east-1"
    capacity: "50%"
    services: ["api", "web", "database"]
```

### 2. Disaster Recovery Procedures

#### DR Site Activation
```bash
# Activate disaster recovery site
./scripts/disaster-recovery/activate-dr-site.sh

# DR activation procedure
export DR_SITE="secondary"
export PRIMARY_SITE="primary"

# Assess primary site status
./scripts/disaster-recovery/assess-primary-site.sh

# Activate DR infrastructure
./scripts/disaster-recovery/activate-infrastructure.sh --site ${DR_SITE}

# Restore data from backup
./scripts/disaster-recovery/restore-data.sh --site ${DR_SITE}

# Start services
./scripts/disaster-recovery/start-services.sh --site ${DR_SITE}

# Update DNS records
./scripts/disaster-recovery/update-dns.sh --target ${DR_SITE}

# Verify DR site functionality
./scripts/disaster-recovery/verify-dr-site.sh
```

#### Data Synchronization
```bash
# Synchronize data to DR site
./scripts/disaster-recovery/sync-data.sh --target ${DR_SITE}

# Database replication setup
./scripts/disaster-recovery/setup-db-replication.sh \
  --primary ${PRIMARY_SITE} \
  --secondary ${DR_SITE}

# File synchronization
rsync -av --progress --delete \
  /opt/freeagentics/uploads/ \
  ${DR_SITE}:/opt/freeagentics/uploads/

# Configuration synchronization
./scripts/disaster-recovery/sync-config.sh --target ${DR_SITE}
```

### 3. Business Continuity

#### Business Impact Assessment
```bash
# Assess business impact
./scripts/business-continuity/assess-impact.sh

# Generate business impact report
./scripts/business-continuity/generate-impact-report.sh

# Prioritize recovery activities
./scripts/business-continuity/prioritize-recovery.sh
```

#### Communication Plan
```bash
# Activate communication plan
./scripts/business-continuity/activate-communication.sh

# Notify stakeholders
./scripts/business-continuity/notify-stakeholders.sh --event disaster

# Update status page
./scripts/business-continuity/update-status-page.sh \
  --message "Disaster recovery in progress"

# Coordinate with external parties
./scripts/business-continuity/coordinate-external.sh
```

## Backup Testing and Validation

### 1. Regular Backup Testing

#### Automated Backup Testing
```bash
# Daily backup verification
./scripts/backup/testing/daily-backup-test.sh

# Weekly restore test
./scripts/backup/testing/weekly-restore-test.sh

# Monthly DR test
./scripts/backup/testing/monthly-dr-test.sh

# Quarterly full recovery test
./scripts/backup/testing/quarterly-full-test.sh
```

#### Backup Validation Procedures
```bash
# Validate backup integrity
./scripts/backup/validation/validate-backup.sh --backup-id ${BACKUP_ID}

# Test restore procedures
./scripts/backup/validation/test-restore.sh --backup-id ${BACKUP_ID}

# Verify data consistency
./scripts/backup/validation/verify-consistency.sh --backup-id ${BACKUP_ID}

# Generate validation report
./scripts/backup/validation/generate-report.sh --backup-id ${BACKUP_ID}
```

### 2. Recovery Testing

#### Disaster Recovery Drills
```bash
# Schedule DR drill
./scripts/backup/testing/schedule-dr-drill.sh --date "2024-02-15"

# Execute DR drill
./scripts/backup/testing/execute-dr-drill.sh

# Evaluate drill results
./scripts/backup/testing/evaluate-drill.sh

# Document lessons learned
./scripts/backup/testing/document-lessons.sh
```

### 3. Compliance Testing

#### Regulatory Compliance
```bash
# Test compliance requirements
./scripts/backup/compliance/test-compliance.sh --framework gdpr

# Generate compliance report
./scripts/backup/compliance/generate-report.sh --framework gdpr

# Audit backup procedures
./scripts/backup/compliance/audit-procedures.sh
```

## Troubleshooting

### 1. Common Backup Issues

#### Backup Failures
**Issue**: Backup fails with insufficient space
```bash
# Check available space
df -h /backup

# Clean old backups
./scripts/backup/maintenance/clean-old-backups.sh --days 30

# Increase backup storage
./scripts/backup/maintenance/increase-storage.sh
```

**Issue**: Database backup timeout
```bash
# Check database performance
./scripts/backup/troubleshooting/check-db-performance.sh

# Optimize backup parameters
./scripts/backup/troubleshooting/optimize-backup.sh

# Use parallel backup
pg_dump --jobs=4 --format=directory ...
```

### 2. Recovery Issues

#### Recovery Failures
**Issue**: Database restore fails
```bash
# Check backup integrity
./scripts/backup/troubleshooting/check-backup-integrity.sh

# Verify database connectivity
./scripts/backup/troubleshooting/check-db-connectivity.sh

# Restore with verbose logging
pg_restore --verbose --clean --if-exists ...
```

**Issue**: Application won't start after recovery
```bash
# Check configuration files
./scripts/backup/troubleshooting/check-config.sh

# Verify dependencies
./scripts/backup/troubleshooting/check-dependencies.sh

# Check logs
./scripts/backup/troubleshooting/check-logs.sh
```

### 3. Performance Issues

#### Backup Performance
```bash
# Monitor backup performance
./scripts/backup/monitoring/monitor-performance.sh

# Optimize backup schedules
./scripts/backup/optimization/optimize-schedules.sh

# Implement backup compression
./scripts/backup/optimization/implement-compression.sh
```

## Backup and Recovery Checklist

### Daily Checklist
- [ ] Verify automated backups completed
- [ ] Check backup logs for errors
- [ ] Validate backup integrity
- [ ] Monitor backup storage usage
- [ ] Test random backup restore

### Weekly Checklist
- [ ] Run full backup validation
- [ ] Test restore procedures
- [ ] Review backup policies
- [ ] Update backup documentation
- [ ] Coordinate with DR site

### Monthly Checklist
- [ ] Conduct DR drill
- [ ] Review backup retention
- [ ] Update recovery procedures
- [ ] Audit backup access
- [ ] Generate compliance report

### Quarterly Checklist
- [ ] Full system recovery test
- [ ] Review RTO/RPO objectives
- [ ] Update disaster recovery plan
- [ ] Train staff on procedures
- [ ] Evaluate backup technologies

---

**Document Information:**
- **Version**: 1.0
- **Last Updated**: January 2024
- **Next Review**: April 2024
- **Owner**: Infrastructure Team
- **Approved By**: Operations Manager

**Recovery Objectives:**
- **RTO**: 30 minutes (Database), 1 hour (Full System)
- **RPO**: 15 minutes (Critical Data), 1 hour (Application Data)