# FreeAgentics Emergency Procedures Runbook

## Overview
This runbook provides step-by-step procedures for handling emergencies and critical incidents in the FreeAgentics production environment.

## Emergency Response Team

### Primary Contacts
- **Incident Commander**: admin@freeagentics.local
- **Technical Lead**: tech-lead@freeagentics.local
- **Security Lead**: security@freeagentics.local
- **On-Call Engineer**: +1-XXX-XXX-XXXX

### Escalation Matrix
| Severity | Response Time | Escalation |
|----------|--------------|------------|
| Critical | 15 minutes | Immediate |
| High | 1 hour | Within 2 hours |
| Medium | 4 hours | Next business day |
| Low | 1 business day | As needed |

## Incident Classification

### Critical (P0)
- Complete system outage
- Data corruption or loss
- Security breach
- Financial impact > $10k/hour

### High (P1)
- Major feature unavailable
- Performance degradation > 50%
- SSL certificate expired
- Database connectivity issues

### Medium (P2)
- Minor feature degraded
- Performance issues < 50%
- Monitoring alerts
- Non-critical service down

### Low (P3)
- Cosmetic issues
- Documentation problems
- Enhancement requests

## Emergency Procedures

### 1. System-Wide Outage

#### Immediate Response (0-15 minutes)
```bash
# 1. Verify the issue
curl -I https://your-domain.com/health
./scripts/validate-production-deployment.sh

# 2. Check service status
docker-compose ps
docker-compose logs --tail=100

# 3. Attempt quick restart
docker-compose restart

# 4. Notify stakeholders
# Send alert to #critical-alerts Slack channel
```

#### Investigation (15-60 minutes)
```bash
# 1. Check system resources
df -h
free -m
top

# 2. Check application logs
tail -f logs/freeagentics.json
docker-compose logs backend
docker-compose logs nginx

# 3. Check database status
docker exec postgres pg_isready -U freeagentics
docker exec postgres psql -U freeagentics -c "SELECT version();"

# 4. Check external dependencies
ping 8.8.8.8
curl -I https://api.openai.com
```

#### Resolution
```bash
# If quick restart didn't work:

# 1. Full system restart
docker-compose down
docker-compose up -d

# 2. Database recovery if needed
./scripts/database-backup.sh restore latest_backup.sql.gz

# 3. Verify recovery
./scripts/validate-production-deployment.sh

# 4. Update incident status
```

### 2. Database Emergency

#### PostgreSQL Down
```bash
# 1. Check container status
docker-compose ps postgres

# 2. Check logs
docker-compose logs postgres

# 3. Restart PostgreSQL
docker-compose restart postgres

# 4. Verify connectivity
docker exec postgres pg_isready -U freeagentics

# 5. Check data integrity
docker exec postgres psql -U freeagentics -c "SELECT count(*) FROM agents;"
```

#### Database Corruption
```bash
# 1. Stop all applications
docker-compose stop backend frontend

# 2. Backup current state (even if corrupted)
./scripts/database-backup.sh backup

# 3. Restore from latest good backup
./scripts/database-backup.sh restore latest_good_backup.sql.gz

# 4. Restart applications
docker-compose start backend frontend

# 5. Verify data integrity
# Run application health checks
```

#### High Connection Count
```bash
# 1. Check current connections
docker exec postgres psql -U freeagentics -c "
SELECT count(*) as connections,
       usename,
       application_name,
       state
FROM pg_stat_activity
GROUP BY usename, application_name, state
ORDER BY connections DESC;"

# 2. Kill idle connections
docker exec postgres psql -U freeagentics -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
AND query_start < now() - interval '5 minutes';"

# 3. Restart application if needed
docker-compose restart backend
```

### 3. Security Incident

#### Suspected Breach
```bash
# 1. IMMEDIATE: Isolate the system
# Block all external access
iptables -A INPUT -j DROP
iptables -A OUTPUT -j DROP

# 2. Preserve evidence
# DO NOT RESTART SERVICES YET
cp -r logs/ /secure/evidence/$(date +%Y%m%d_%H%M%S)/

# 3. Analyze logs for suspicious activity
grep -r "suspicious\|attack\|unauthorized" logs/
grep -r "failed.*login" logs/

# 4. Check for unauthorized access
docker exec postgres psql -U freeagentics -c "
SELECT * FROM auth_logs
WHERE success = false
AND created_at > now() - interval '24 hours'
ORDER BY created_at DESC;"

# 5. Contact security team immediately
# DO NOT proceed without security team approval
```

#### SSL Certificate Expired
```bash
# 1. Check certificate status
openssl x509 -in nginx/ssl/cert.pem -text -noout | grep "Not After"

# 2. Generate temporary self-signed certificate
./scripts/setup-ssl-production.sh self-signed

# 3. Restart nginx
docker-compose restart nginx

# 4. Obtain new certificate
./scripts/setup-ssl-production.sh letsencrypt

# 5. Verify SSL is working
curl -I https://your-domain.com
```

### 4. Performance Emergency

#### High CPU Usage (>90%)
```bash
# 1. Identify the process
docker stats
top -p $(docker inspect --format='{{.State.Pid}}' $(docker-compose ps -q backend))

# 2. Scale up if possible
docker-compose up -d --scale backend=2

# 3. Check for infinite loops or heavy operations
docker exec backend python -c "
import psutil
for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
    if proc.info['cpu_percent'] > 50:
        print(proc.info)
"

# 4. Restart problematic service
docker-compose restart backend
```

#### Memory Exhaustion
```bash
# 1. Check memory usage
free -m
docker stats

# 2. Identify memory hogs
docker exec backend python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
    if proc.info['memory_percent'] > 10:
        print(proc.info)
"

# 3. Clear caches if safe
docker exec redis redis-cli FLUSHALL
docker system prune -f

# 4. Restart services with memory limits
docker-compose restart
```

#### Disk Space Critical (<5%)
```bash
# 1. Check disk usage
df -h
du -sh /* | sort -hr | head -10

# 2. Clean up logs
find logs/ -name "*.log" -mtime +7 -delete
docker system prune -f --volumes

# 3. Clean old backups (keep last 7 days)
find /var/backups/freeagentics/ -name "*.sql.gz" -mtime +7 -delete

# 4. Restart services if needed
docker-compose restart
```

### 5. Network Emergency

#### DNS Issues
```bash
# 1. Check DNS resolution
nslookup your-domain.com
dig your-domain.com

# 2. Check DNS configuration
cat /etc/resolv.conf

# 3. Test with different DNS servers
nslookup your-domain.com 8.8.8.8
nslookup your-domain.com 1.1.1.1

# 4. Contact DNS provider if external issue
```

#### Load Balancer Issues
```bash
# 1. Check nginx status
docker-compose ps nginx
docker-compose logs nginx

# 2. Test backend connectivity
curl -I http://localhost:8000/health

# 3. Restart nginx
docker-compose restart nginx

# 4. Check configuration
docker exec nginx nginx -t
```

## Recovery Verification

### Health Check Checklist
```bash
# 1. Application health
curl -f https://your-domain.com/health

# 2. API endpoints
curl -f https://your-domain.com/api/v1/system/info

# 3. Database connectivity
docker exec postgres pg_isready -U freeagentics

# 4. Authentication system
curl -f https://your-domain.com/api/v1/auth/health

# 5. WebSocket functionality
# Manual test in browser console:
# ws = new WebSocket('wss://your-domain.com/ws/agents')

# 6. Monitoring systems
./scripts/validate-monitoring.sh
```

### Performance Verification
```bash
# 1. Response time check
curl -w "%{time_total}" -o /dev/null -s https://your-domain.com/

# 2. Load test (light)
for i in {1..10}; do
  curl -w "%{time_total}\n" -o /dev/null -s https://your-domain.com/health
done

# 3. Database performance
docker exec postgres psql -U freeagentics -c "
EXPLAIN ANALYZE SELECT count(*) FROM agents;
"
```

## Communication Templates

### Initial Alert
```
🚨 INCIDENT ALERT - [Severity Level]

Summary: [Brief description]
Impact: [User/system impact]
Started: [Timestamp]
Investigating: [Team member name]

Status page: https://status.your-domain.com
Updates will follow every 15 minutes.
```

### Progress Update
```
📊 INCIDENT UPDATE - [Severity Level]

Summary: [Brief description]
Progress: [What has been done]
Next steps: [What's happening next]
ETA: [Estimated resolution time]

Updated: [Timestamp]
```

### Resolution Notice
```
✅ INCIDENT RESOLVED - [Severity Level]

Summary: [Brief description]
Root cause: [What caused the issue]
Resolution: [How it was fixed]
Duration: [Total outage time]

Post-mortem: [Link to detailed analysis]
Resolved: [Timestamp]
```

## Post-Incident Procedures

### Immediate (0-24 hours)
1. **Verify full recovery** with comprehensive testing
2. **Document timeline** of events and actions taken
3. **Notify stakeholders** of resolution
4. **Schedule post-mortem** meeting within 48 hours

### Short-term (1-7 days)
1. **Conduct post-mortem** with all involved parties
2. **Identify root causes** and contributing factors
3. **Create action items** for prevention
4. **Update runbooks** based on lessons learned

### Long-term (1-4 weeks)
1. **Implement preventive measures** from action items
2. **Update monitoring** and alerting if needed
3. **Review and update** emergency procedures
4. **Share learnings** with broader team

## Contact Information

### Internal Teams
- **Operations**: ops@freeagentics.local
- **Development**: dev@freeagentics.local
- **Security**: security@freeagentics.local
- **Management**: mgmt@freeagentics.local

### External Vendors
- **DNS Provider**: [Contact info]
- **SSL Provider**: [Contact info]
- **Cloud Provider**: [Contact info]
- **Security Consultant**: [Contact info]

### Emergency Services
- **Local IT Support**: [Phone number]
- **Data Center**: [Contact info]
- **Legal**: [Contact info for security incidents]

---

**Remember**: Always prioritize safety and data integrity. When in doubt, contact the security team before taking action.

*Last Updated: [Date]*
