# FreeAgentics Automated Backup System

## Overview

The FreeAgentics Automated Backup System provides comprehensive data protection following the industry-standard 3-2-1 backup strategy:
- **3** copies of your data
- **2** different storage media types
- **1** offsite copy

## Features

### Backup Components
1. **PostgreSQL Database**: Full and incremental backups with point-in-time recovery
2. **Application State**: Agent beliefs, coalition states, and runtime data
3. **Knowledge Graph**: Complete graph structure and relationships
4. **Configuration**: All system and application configurations
5. **Redis Cache**: Session data and cache snapshots

### 3-2-1 Strategy Implementation
1. **Primary Copy**: Live production data
2. **Local Backup**: High-speed local storage with fast recovery
3. **Offsite Backups**:
   - Primary: AWS S3 with lifecycle policies
   - Secondary: Azure/GCP/Backblaze for redundancy

### Key Features
- **Automated Scheduling**: Configurable backup schedules
- **Encryption**: AES-256 encryption for all backups
- **Compression**: Optimized compression for storage efficiency
- **Verification**: Automated integrity checks and restore testing
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Notifications**: Slack, email, and PagerDuty alerts
- **Disaster Recovery**: Multiple recovery scenarios supported

## Installation

### Quick Start
```bash
# Run installation script as root
sudo ./scripts/backup/install-backup-system.sh

# Edit configuration
sudo vim /etc/freeagentics/backup.env

# Start backup service
sudo systemctl start freeagentics-backup.timer
```

### Manual Installation
1. Install system dependencies:
   ```bash
   apt-get install postgresql-client redis-tools python3-pip awscli
   ```

2. Install Python dependencies:
   ```bash
   pip3 install boto3 azure-storage-blob google-cloud-storage psycopg2-binary redis schedule
   ```

3. Create backup directories:
   ```bash
   mkdir -p /var/backups/freeagentics/{daily,redis,config,knowledge_graph,app_state,logs,metadata}
   ```

4. Copy configuration template:
   ```bash
   cp backup-config.env.template /etc/freeagentics/backup.env
   ```

5. Install systemd service:
   ```bash
   cp freeagentics-backup.service /etc/systemd/system/
   systemctl enable freeagentics-backup.timer
   ```

## Configuration

### Essential Settings
Edit `/etc/freeagentics/backup.env`:

```bash
# Database settings
POSTGRES_HOST="localhost"
POSTGRES_DB="freeagentics"
POSTGRES_USER="freeagentics"
POSTGRES_PASSWORD="your_password"

# S3 offsite backup
S3_BUCKET="freeagentics-backups-prod"
S3_REGION="us-east-1"

# Retention policies
LOCAL_RETENTION_DAYS="7"
OFFSITE_RETENTION_DAYS="30"
ARCHIVE_RETENTION_DAYS="365"

# Notifications
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
EMAIL_TO="ops@freeagentics.io"
```

### Cloud Storage Setup

#### AWS S3
```bash
# Configure AWS credentials
aws configure

# Create S3 bucket
aws s3 mb s3://freeagentics-backups-prod
```

#### Azure Blob Storage
```bash
# Login to Azure
az login

# Create storage account
az storage account create --name freeagenticsbackups --resource-group prod
```

#### Google Cloud Storage
```bash
# Authenticate with GCP
gcloud auth login

# Create bucket
gsutil mb gs://freeagentics-backups-prod
```

## Usage

### Command Line Interface

```bash
# Run full backup immediately
python3 automated-backup-system.py --run-now

# Run as daemon with scheduled backups
python3 automated-backup-system.py --daemon

# Test disaster recovery
python3 automated-backup-system.py --test-restore

# Clean up old backups
python3 automated-backup-system.py --cleanup
```

### Backup Scripts

```bash
# Full system backup
./full-backup.sh

# Database-only backup
./database-backup.sh

# Disaster recovery
./disaster-recovery.sh <scenario> [options]

# Verify backup integrity
./verify-backup-integrity.sh --latest --deep
```

### Verification

```bash
# Verify latest backup
./verify-backup-integrity.sh --latest

# Verify specific backup
./verify-backup-integrity.sh --backup-id full_20240718_020000

# Deep verification with restore test
./verify-backup-integrity.sh --latest --deep --restore-test

# Generate verification report
./verify-backup-integrity.sh --all --report
```

## Backup Schedule

| Backup Type | Schedule | Retention (Local) | Retention (Offsite) |
|------------|----------|-------------------|---------------------|
| Full Database | Daily at 2 AM | 7 days | 30 days → Glacier |
| Incremental DB | Every 6 hours | 2 days | 7 days |
| Redis Snapshot | Hourly | 48 hours | 7 days |
| Configuration | On change + Weekly | 30 days | 90 days |
| Application State | Every 6 hours | 7 days | 30 days |
| Knowledge Graph | Daily with DB | 7 days | 30 days |

## Monitoring

### Prometheus Metrics
- `freeagentics_backup_status`: Backup success/failure status
- `freeagentics_backup_size_bytes`: Backup size in bytes
- `freeagentics_backup_duration_seconds`: Backup duration
- `freeagentics_last_successful_backup_timestamp`: Last successful backup time
- `freeagentics_backup_storage_usage_percent`: Storage utilization

### Grafana Dashboard
Import the dashboard from `monitoring/dashboards/backup_monitoring_dashboard.json`

### Health Checks
```bash
# Check backup service status
systemctl status freeagentics-backup.timer

# View recent logs
journalctl -u freeagentics-backup -f

# Check backup metrics
curl localhost:9090/metrics | grep freeagentics_backup
```

## Disaster Recovery

### Recovery Scenarios

1. **Database Corruption**
   ```bash
   ./disaster-recovery.sh database-corruption --point-in-time "2024-07-18 14:00:00"
   ```

2. **Complete System Failure**
   ```bash
   ./disaster-recovery.sh full-system --backup-source s3
   ```

3. **Ransomware Attack**
   ```bash
   ./disaster-recovery.sh ransomware --backup-source s3 --force
   ```

4. **Accidental Deletion**
   ```bash
   ./disaster-recovery.sh human-error --selective-restore
   ```

### Quick Recovery Commands

```bash
# Restore database only
./quick-restore-db.sh

# Restore full system
./full-system-restore.sh --auto-select-latest

# Restore to different host
./disaster-recovery.sh full-system --recovery-target dr-site.example.com
```

## Testing

### Automated Tests
- Weekly restore tests (Sunday 5 AM)
- Daily checksum verification
- Continuous monitoring of backup health

### Manual Testing
```bash
# Test backup process
./test-backup.sh

# Test restore process
./test-restore.sh

# Full DR drill
./dr-drill.sh
```

## Troubleshooting

### Common Issues

1. **Backup Failures**
   ```bash
   # Check logs
   tail -f /var/log/freeagentics/backup.log
   journalctl -u freeagentics-backup -n 100

   # Verify credentials
   pg_isready -h localhost -U freeagentics
   redis-cli ping
   ```

2. **Storage Issues**
   ```bash
   # Check disk space
   df -h /var/backups/freeagentics

   # Clean up old backups
   ./cleanup-old-backups.sh
   ```

3. **Offsite Sync Failures**
   ```bash
   # Test S3 connectivity
   aws s3 ls s3://freeagentics-backups-prod

   # Check credentials
   aws sts get-caller-identity
   ```

### Debug Mode
```bash
# Run with debug logging
DEBUG=true ./automated-backup-system.py --run-now

# Test specific component
./test-component.sh --component database
```

## Security Considerations

1. **Encryption**
   - All backups encrypted with AES-256
   - Encryption keys stored separately
   - Keys rotated quarterly

2. **Access Control**
   - Dedicated backup user with minimal privileges
   - Separate credentials for each storage provider
   - MFA required for production access

3. **Network Security**
   - All transfers over TLS/SSL
   - VPN required for cross-region transfers
   - IP whitelisting for backup servers

## Best Practices

1. **Regular Testing**
   - Test restores weekly
   - Conduct quarterly DR drills
   - Verify backup integrity daily

2. **Documentation**
   - Keep runbooks updated
   - Document all procedures
   - Maintain contact lists

3. **Monitoring**
   - Set up comprehensive alerts
   - Review metrics regularly
   - Track storage growth

4. **Compliance**
   - Follow data retention policies
   - Ensure GDPR compliance
   - Maintain audit trails

## Support

- **Documentation**: `/docs/operations/DISASTER_RECOVERY_PROCEDURES.md`
- **Emergency Contact**: ops@freeagentics.io
- **24/7 Hotline**: +1-555-0100
- **Slack Channel**: #backup-alerts

## License

Copyright 2024 FreeAgentics. All rights reserved.
