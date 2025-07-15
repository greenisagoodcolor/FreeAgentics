# FreeAgentics Incident Response Guide

## Overview

This document defines the incident response procedures for FreeAgentics, including severity levels, escalation paths, response team roles, and post-mortem processes.

## Incident Severity Levels

### SEV-1: Critical (System Down)
- **Definition**: Complete system outage or critical security breach
- **Examples**:
  - API completely unresponsive
  - Database corruption or data loss
  - Security breach with data exposure
  - Complete agent coordination failure
- **Response Time**: Immediate (< 15 minutes)
- **Notification**: Page on-call engineer, notify CTO/VP Engineering

### SEV-2: Major (Severe Degradation)
- **Definition**: Significant functionality impaired affecting multiple users
- **Examples**:
  - API response times > 5s for majority of requests
  - Agent coordination success rate < 50%
  - Memory usage > 90% sustained
  - Authentication service intermittent failures
- **Response Time**: < 30 minutes
- **Notification**: Page on-call engineer, notify team lead

### SEV-3: Minor (Limited Impact)
- **Definition**: Non-critical functionality affected or performance degradation
- **Examples**:
  - Individual agent failures (< 10%)
  - Slight performance degradation (< 2x normal)
  - Non-critical feature unavailable
- **Response Time**: < 2 hours
- **Notification**: Slack alert to engineering channel

### SEV-4: Low (Minimal Impact)
- **Definition**: Minor issues with workarounds available
- **Examples**:
  - UI cosmetic issues
  - Documentation errors
  - Non-blocking warnings in logs
- **Response Time**: Next business day
- **Notification**: Create ticket in issue tracker

## Escalation Procedures

### Initial Response Flow
```
1. Alert received (Prometheus/Grafana/User Report)
   ↓
2. On-call engineer acknowledges (< 5 min)
   ↓
3. Assess severity level
   ↓
4. If SEV-1/2: Create incident channel #incident-YYYY-MM-DD-description
   ↓
5. Begin investigation and mitigation
```

### Escalation Path
- **L1 (0-15 min)**: On-call engineer
- **L2 (15-30 min)**: Team lead + domain expert
- **L3 (30-60 min)**: VP Engineering/CTO
- **L4 (60+ min)**: CEO (for customer-facing SEV-1 only)

## Response Team Roles

### Incident Commander (IC)
- Overall incident coordination
- Makes final decisions on mitigation strategies
- Coordinates communication
- Typically: Senior engineer or team lead

### Technical Lead (TL)
- Leads technical investigation
- Implements fixes and mitigations
- Coordinates with other technical teams
- Documents technical timeline

### Communications Lead (CL)
- Updates status page
- Communicates with stakeholders
- Drafts customer communications
- Manages internal updates

### Scribe
- Documents all actions taken
- Records timeline of events
- Captures decisions and rationale
- Prepares post-mortem notes

## Communication Templates

### Internal Incident Alert
```
🚨 INCIDENT ALERT - [SEV-X]

**Issue**: [Brief description]
**Impact**: [User/system impact]
**Status**: Investigating / Mitigating / Resolved
**IC**: @[name]
**Channel**: #incident-[name]
**Next Update**: [time]
```

### Customer Communication (SEV-1/2)
```
Subject: [Service Name] - Service Disruption

We are currently experiencing issues with [affected service].

**Impact**: [What users may experience]
**Status**: Our team is actively working on resolution
**Workaround**: [If available]
**Next Update**: [Time, typically within 1 hour]

We apologize for any inconvenience and will update you shortly.
```

### Resolution Communication
```
Subject: [Service Name] - Issue Resolved

The issue affecting [service] has been resolved as of [time].

**Duration**: [start time] - [end time]
**Impact**: [Brief summary]
**Root Cause**: [High-level explanation]

A detailed post-mortem will follow. Thank you for your patience.
```

## Incident Response Checklist

### Immediate Actions (First 15 minutes)
- [ ] Acknowledge alert
- [ ] Assess severity and impact
- [ ] Create incident channel if SEV-1/2
- [ ] Assign IC role
- [ ] Begin investigation in monitoring tools
- [ ] Check recent deployments/changes
- [ ] Update status page if customer-facing

### Investigation Phase
- [ ] Review relevant dashboards:
  - [ ] Grafana system overview
  - [ ] API performance metrics
  - [ ] Agent coordination dashboard
  - [ ] Error rate trends
- [ ] Check logs:
  ```bash
  # Recent errors
  docker-compose logs --tail=1000 api | grep ERROR
  
  # Agent coordination issues
  docker-compose logs --tail=1000 api | grep -E "(coalition|coordination|agent)"
  
  # Database issues
  docker-compose logs --tail=500 postgres
  ```
- [ ] Verify external dependencies (database, Redis, external APIs)
- [ ] Check resource utilization

### Mitigation Actions
- [ ] Implement immediate mitigation (if available)
- [ ] Consider rollback if recent deployment
- [ ] Scale resources if needed
- [ ] Enable circuit breakers if applicable
- [ ] Communicate progress every 30 minutes

### Resolution Verification
- [ ] Confirm metrics returning to normal
- [ ] Verify through synthetic monitoring
- [ ] Test critical user journeys
- [ ] Monitor for 15 minutes post-fix
- [ ] Update status page

## Post-Mortem Process

### Timeline
- **Within 24 hours**: Initial incident report
- **Within 3 days**: Draft post-mortem
- **Within 5 days**: Post-mortem review meeting
- **Within 7 days**: Action items assigned with deadlines

### Post-Mortem Template
```markdown
# Incident Post-Mortem: [Incident Name]

**Date**: [YYYY-MM-DD]
**Duration**: [Start - End time]
**Severity**: SEV-X
**Author**: [Name]

## Executive Summary
[2-3 sentence summary of incident and impact]

## Impact
- **Customer Impact**: [Quantified where possible]
- **Revenue Impact**: [If applicable]
- **Data Impact**: [Any data loss/corruption]

## Timeline
[UTC times]
- HH:MM - Event/Action taken
- HH:MM - Event/Action taken

## Root Cause Analysis
[Detailed explanation of what caused the incident]

## Contributing Factors
- [Factor 1]
- [Factor 2]

## What Went Well
- [Positive aspect 1]
- [Positive aspect 2]

## What Could Be Improved
- [Improvement area 1]
- [Improvement area 2]

## Action Items
| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| [Action 1] | @owner | YYYY-MM-DD | High/Med/Low |

## Lessons Learned
[Key takeaways for the team]
```

### Blameless Culture
- Focus on systems and processes, not individuals
- Ask "what" and "how", not "who"
- Treat incidents as learning opportunities
- Share post-mortems organization-wide

## Monitoring and Alerting Integration

### Alert Response Mapping
| Alert | Runbook | Initial Action |
|-------|---------|----------------|
| High memory usage | [high_memory_usage.md](./high_memory_usage.md) | Check memory profiler |
| API latency > 1s | [api_performance_degradation.md](./api_performance_degradation.md) | Review slow queries |
| DB connection errors | [database_connection_issues.md](./database_connection_issues.md) | Check connection pool |
| Agent coordination failures | [agent_coordination_failures.md](./agent_coordination_failures.md) | Review agent logs |

### Key Dashboards
- **System Overview**: http://localhost:3001/d/freeagentics-overview
- **API Performance**: http://localhost:3001/d/freeagentics-api-perf
- **Agent Coordination**: http://localhost:3001/d/freeagentics-agents
- **Memory Analysis**: http://localhost:3001/d/freeagentics-memory

## Tools and Commands

### Quick Diagnosis Commands
```bash
# System health check
make health-check

# View recent errors
make logs-errors

# Check resource usage
docker stats

# Database connection count
docker exec freeagentics-postgres psql -U postgres -d freeagentics -c "SELECT count(*) FROM pg_stat_activity;"

# Redis memory usage
docker exec freeagentics-redis redis-cli INFO memory

# Agent status
curl -s http://localhost:8000/api/v1/monitoring/agents | jq
```

### Emergency Commands
```bash
# Restart API service
docker-compose restart api

# Clear Redis cache (USE WITH CAUTION)
docker exec freeagentics-redis redis-cli FLUSHDB

# Increase API replicas
docker-compose up -d --scale api=3

# Emergency database backup
make db-backup-emergency
```

## Training and Drills

### Quarterly Requirements
- Incident response drill (tabletop or live)
- Runbook review and updates
- Tool access verification
- Escalation path testing

### New Team Member Onboarding
- [ ] Review this document
- [ ] Shadow on-call rotation
- [ ] Participate in incident drill
- [ ] Complete runbook walkthrough

## Contact Information

### On-Call Rotation
- Schedule: [Link to PagerDuty/Opsgenie]
- Primary: [Phone/Slack]
- Secondary: [Phone/Slack]

### Key Contacts
- **Database Admin**: @dba-team
- **Security Team**: @security-team
- **Customer Success**: @cs-team
- **Legal**: legal@company.com (for data breaches)

---

*Last Updated: [Date]*
*Next Review: [Date + 3 months]*