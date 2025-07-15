# Disaster Recovery Runbook

## Emergency Information

**🚨 EMERGENCY CONTACTS**
- **Primary On-Call**: ops@freeagentics.io / +1-555-0100
- **DBA Team**: dba@freeagentics.io / +1-555-0101
- **Security Team**: security@freeagentics.io / +1-555-0102
- **Management Escalation**: cto@freeagentics.io / +1-555-0103

**📋 QUICK REFERENCE**
- Recovery Scripts: `/home/green/FreeAgentics/scripts/backup/`
- Backup Location: `/var/backups/freeagentics/`
- S3 Bucket: `s3://freeagentics-backups-prod`
- This Runbook: `/home/green/FreeAgentics/docs/operations/DISASTER_RECOVERY_RUNBOOK.md`

## Table of Contents

1. [Initial Response](#initial-response)
2. [Disaster Assessment](#disaster-assessment)
3. [Recovery Scenarios](#recovery-scenarios)
4. [Post-Recovery Procedures](#post-recovery-procedures)
5. [Communication Templates](#communication-templates)
6. [Testing and Validation](#testing-and-validation)

## Initial Response

### Step 1: Immediate Actions (0-5 minutes)

1. **Confirm the Incident**
   ```bash
   # Check system status
   systemctl status freeagentics
   docker ps
   pg_isready -h postgres -p 5432
   ```

2. **Assess Impact**
   - [ ] Database accessible?
   - [ ] Application responding?
   - [ ] Redis cache available?
   - [ ] External services reachable?

3. **Emergency Notification**
   ```bash
   # Send immediate alert
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"🚨 EMERGENCY: FreeAgentics system experiencing issues - investigating"}' \
     $SLACK_WEBHOOK
   ```

### Step 2: Incident Command Setup (5-10 minutes)

1. **Establish Command Center**
   - Primary responder takes incident command
   - Open emergency bridge/video call
   - Document all actions in shared document

2. **Gather Key Personnel**
   - [ ] Incident Commander
   - [ ] Technical Lead
   - [ ] Database Administrator
   - [ ] Security Representative (if applicable)

3. **Create War Room**
   - Slack channel: `#incident-YYYYMMDD-HHMMSS`
   - Shared document for timeline
   - Voice/video bridge for real-time communication

## Disaster Assessment

### Assessment Script

```bash
# Run disaster assessment
cd /home/green/FreeAgentics
./scripts/backup/disaster-recovery.sh assess

# Review assessment report
cat /tmp/freeagentics-recovery-*/assessment.txt
```

### Severity Classification

#### P0 - Critical (Complete Service Outage)
- All services down
- Database inaccessible
- Data corruption suspected
- **Target Recovery Time**: 30 minutes
- **Escalation**: Immediate management notification

#### P1 - High (Major Service Degradation)
- Some services down
- Database accessible but slow
- Application errors increasing
- **Target Recovery Time**: 1 hour
- **Escalation**: Team leads notified

#### P2 - Medium (Service Degradation)
- Services running but degraded
- Intermittent issues
- Performance problems
- **Target Recovery Time**: 2 hours
- **Escalation**: Standard team notification

### Data Loss Assessment

```bash
# Check last successful backup
ls -la /var/backups/freeagentics/daily/
aws s3 ls s3://freeagentics-backups-prod/daily/ --recursive | tail -10

# Check WAL log availability
ls -la /var/lib/postgresql/data/pg_wal/

# Estimate potential data loss
./scripts/backup/database-restore.sh verify
```

## Recovery Scenarios

### Scenario 1: Complete System Failure

**Symptoms**: All services down, infrastructure compromised

**Recovery Command**:
```bash
# Full system recovery
./scripts/backup/disaster-recovery.sh full-system

# With dry run first
./scripts/backup/disaster-recovery.sh full-system --dry-run
```

**Manual Steps**:
1. Provision new infrastructure if needed
2. Restore from backups
3. Update DNS/load balancers
4. Verify all services

**Estimated Recovery Time**: 2-4 hours

### Scenario 2: Database Corruption

**Symptoms**: Database errors, data inconsistencies, query failures

**Recovery Command**:
```bash
# Database corruption recovery
./scripts/backup/disaster-recovery.sh database-corruption
```

**Manual Steps**:
1. Stop application services
2. Assess corruption extent
3. Restore from clean backup
4. Verify data integrity
5. Restart services

**Estimated Recovery Time**: 30-60 minutes

### Scenario 3: Ransomware Attack

**Symptoms**: Encrypted files, ransom demands, suspicious activity

**Recovery Command**:
```bash
# Ransomware recovery
./scripts/backup/disaster-recovery.sh ransomware
```

**Manual Steps**:
1. **IMMEDIATELY** isolate affected systems
2. Do NOT pay ransom
3. Contact security team
4. Restore from clean backups
5. Implement additional security measures

**Estimated Recovery Time**: 1-2 hours

### Scenario 4: Data Center Loss

**Symptoms**: Complete data center outage, infrastructure unavailable

**Recovery Command**:
```bash
# Data center loss recovery
./scripts/backup/disaster-recovery.sh data-center-loss
```

**Manual Steps**:
1. Activate DR site
2. Restore from offsite backups
3. Update DNS records
4. Redirect traffic
5. Monitor performance

**Estimated Recovery Time**: 4-6 hours

### Scenario 5: Human Error (Accidental Deletion)

**Symptoms**: Missing data, accidental DROP commands

**Recovery Command**:
```bash
# Point-in-time recovery
./scripts/backup/disaster-recovery.sh human-error \
  --point-in-time "2024-01-15 14:30:00 UTC"
```

**Manual Steps**:
1. Identify exact time of error
2. Stop further changes
3. Restore to point before error
4. Verify data consistency
5. Implement preventive measures

**Estimated Recovery Time**: 15-30 minutes

## Detailed Recovery Procedures

### Database Recovery

#### Full Database Restore
```bash
# Stop application
docker-compose stop backend-prod web worker

# Restore from latest backup
./scripts/backup/database-restore.sh latest --force

# Verify restoration
./scripts/backup/database-restore.sh verify

# Start application
docker-compose up -d backend-prod web worker
```

#### Point-in-Time Recovery
```bash
# Restore to specific time
./scripts/backup/database-restore.sh pitr "2024-01-15 14:30:00 UTC"

# Verify recovery
psql -h postgres -U freeagentics -d freeagentics -c "SELECT NOW();"
```

### Application Recovery

#### Code Restoration
```bash
# Restore from Git
cd /home/green/FreeAgentics
git reset --hard HEAD
git clean -fd

# Or restore from backup
git checkout v1.2.3  # known good version
```

#### Configuration Restoration
```bash
# Restore configuration
./scripts/backup/config-backup.sh restore \
  /var/backups/freeagentics/config/config_backup_latest.tar.gz.enc

# Verify configuration
docker-compose config
```

### Service Recovery

#### Container Recovery
```bash
# Restart all services
docker-compose down
docker-compose up -d

# Check service health
docker-compose ps
./scripts/maintenance/system-health-check.sh
```

#### Load Balancer Recovery
```bash
# Update load balancer config
nginx -t
systemctl reload nginx

# Verify routing
curl -I http://api.freeagentics.io/health
```

## Post-Recovery Procedures

### Verification Checklist

- [ ] All services running and healthy
- [ ] Database queries working correctly
- [ ] Application responding to requests
- [ ] Cache functioning properly
- [ ] Monitoring systems operational
- [ ] SSL certificates valid
- [ ] Backups resuming normally

### Health Verification Commands

```bash
# System health check
./scripts/maintenance/system-health-check.sh --full

# Database integrity check
psql -h postgres -U freeagentics -d freeagentics -c "
  SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
  FROM pg_stat_user_tables 
  ORDER BY n_tup_ins + n_tup_upd + n_tup_del DESC LIMIT 10;"

# Application health check
curl -f http://localhost:8000/health
curl -f http://localhost:8000/api/v1/health

# Redis health check
redis-cli ping
redis-cli info replication
```

### Documentation Requirements

1. **Incident Report Template**
   - Timeline of events
   - Root cause analysis
   - Recovery actions taken
   - Lessons learned
   - Preventive measures

2. **Recovery Log**
   - All commands executed
   - Recovery time stamps
   - Verification results
   - Issues encountered

### Security Verification

```bash
# Check for unauthorized access
./scripts/security/audit-access.sh

# Verify certificates
./scripts/maintenance/check-certificates.sh

# Security scan
./scripts/security/run_security_tests.py
```

## Communication Templates

### Initial Incident Notification

```
Subject: [P0 INCIDENT] FreeAgentics Service Disruption

We are currently experiencing a service disruption affecting FreeAgentics.

Status: INVESTIGATING
Impact: [Service Impact Description]
ETA: [Estimated Recovery Time]

Updates will be provided every 15 minutes.

Incident Commander: [Name]
Next Update: [Time]
```

### Recovery Progress Update

```
Subject: [P0 INCIDENT] FreeAgentics Recovery Update

Recovery Progress Update:

Completed:
- [✓] System assessment
- [✓] Database restoration initiated

In Progress:
- [⏳] Database restoration (ETA: 30 min)

Next Steps:
- Service verification
- Performance validation

New ETA: [Time]
Next Update: [Time]
```

### Incident Resolution

```
Subject: [RESOLVED] FreeAgentics Service Restored

The FreeAgentics service has been fully restored.

Resolution Time: [Duration]
Root Cause: [Brief description]
Data Loss: [None/Minimal/Description]

All systems are now operational and monitoring shows normal performance.

A full post-incident review will be conducted within 24 hours.
```

### Customer Communication

```
Subject: Service Disruption Update

We experienced a technical issue that temporarily affected FreeAgentics service.

Impact: [Customer-facing description]
Duration: [Start time] - [End time]
Resolution: Issue has been resolved

We apologize for any inconvenience. Our team is conducting a thorough review 
to prevent similar issues in the future.

For questions: support@freeagentics.io
```

## Testing and Validation

### Monthly Recovery Tests

```bash
# Test database recovery
./scripts/testing/test-db-restore.sh

# Test application recovery
./scripts/testing/test-app-recovery.sh

# Test configuration recovery
./scripts/testing/test-config-restore.sh
```

### Quarterly Disaster Recovery Drills

1. **Scheduled Test**
   - Announce test to team
   - Execute recovery procedures
   - Document results
   - Identify improvements

2. **Surprise Test**
   - Unannounced test
   - Measure response time
   - Evaluate procedures
   - Team performance assessment

### Recovery Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|---------|
| Detection Time | < 5 min | - | - |
| Response Time | < 10 min | - | - |
| Database Recovery | < 30 min | - | - |
| Full Recovery | < 2 hours | - | - |
| Communication | < 15 min | - | - |

## Appendix

### A. Recovery Tools Inventory

| Tool | Location | Purpose |
|------|----------|---------|
| disaster-recovery.sh | /scripts/backup/ | Main recovery script |
| database-restore.sh | /scripts/backup/ | Database restoration |
| config-backup.sh | /scripts/backup/ | Configuration management |
| offsite-sync.sh | /scripts/backup/ | Offsite backup sync |

### B. Backup Locations

- **Local**: `/var/backups/freeagentics/`
- **S3**: `s3://freeagentics-backups-prod/`
- **Git**: `https://github.com/your-org/FreeAgentics.git`

### C. Recovery Time Objectives

| System Component | RTO | RPO |
|------------------|-----|-----|
| API Services | 30 min | 15 min |
| Database | 1 hour | 15 min |
| Web Interface | 30 min | 0 min |
| Cache Layer | 2 hours | 1 hour |
| Monitoring | 2 hours | 1 hour |

### D. Emergency Runbook Locations

- **Primary**: `/home/green/FreeAgentics/docs/operations/`
- **Backup**: S3 bucket documentation folder
- **Print Copy**: Emergency operations binder

---

**Last Updated**: January 2024  
**Next Review**: April 2024  
**Version**: 1.0

**Remember**: In an emergency, stay calm, follow procedures, and communicate clearly. When in doubt, escalate immediately.