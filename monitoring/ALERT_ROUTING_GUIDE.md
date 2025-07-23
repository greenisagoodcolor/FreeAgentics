# FreeAgentics AlertManager Configuration Guide

## Overview

This guide documents the AlertManager configuration for FreeAgentics production monitoring, including alert routing logic, escalation procedures, and operational guidelines.

## Configuration Files

- **AlertManager Config**: `/home/green/FreeAgentics/monitoring/alertmanager-intelligent.yml`
- **Prometheus Rules**: `/home/green/FreeAgentics/monitoring/rules/freeagentics-alerts.yml`
- **Validation Script**: `/home/green/FreeAgentics/test_alertmanager_config.py`

## Alert Severity Levels

### Critical (severity: critical)
- **Immediate Response Required**
- **Escalation**: PagerDuty + Slack + Email
- **Response Time**: < 5 minutes
- **Examples**: System down, coordination failures >50 agents, memory >34.5MB/agent

### High (severity: high)
- **Urgent Response Required**
- **Escalation**: Slack + Email after 15 minutes
- **Response Time**: < 15 minutes
- **Examples**: API response time >500ms p95, high error rates, database issues

### Medium (severity: medium)
- **Standard Response**
- **Escalation**: Slack notification
- **Response Time**: < 30 minutes
- **Examples**: Performance warnings, resource usage alerts

## Alert Routing Matrix

### Critical Alerts

| Alert Type | Primary Channel | Secondary Channel | Escalation Time |
|------------|----------------|-------------------|-----------------|
| System Down | PagerDuty | Slack (#incidents) | Immediate |
| Security Incidents | PagerDuty | Slack (#security-incidents) | Immediate |
| Agent Coordination | PagerDuty | Slack (#agents-alerts) | Immediate |
| Memory Critical | PagerDuty | Slack (#sre-alerts) | Immediate |

### High Severity Alerts

| Component | Primary Channel | Team | Escalation Time |
|-----------|----------------|------|-----------------|
| API Performance | Slack (#backend-high-alerts) | Backend Team | 15 minutes |
| Database Issues | Slack (#database-high-alerts) | Database Team | 15 minutes |
| Agent Errors | Slack (#agents-high-alerts) | Agents Team | 15 minutes |
| Security Violations | Slack (#security-high-alerts) | Security Team | 10 minutes |
| Infrastructure | Slack (#sre-high-alerts) | SRE Team | 15 minutes |

### Medium Severity Alerts

| Category | Primary Channel | Team | Escalation Time |
|----------|----------------|------|-----------------|
| Performance | Slack (#performance) | Performance Team | 30 minutes |
| Business Logic | Slack (#product) | Product Team | 1 hour |
| Coordination | Slack (#agents-medium-alerts) | Agents Team | 30 minutes |
| Knowledge Graph | Slack (#backend-medium-alerts) | Backend Team | 30 minutes |

## Team Responsibilities

### SRE Team
- **Critical**: System infrastructure, memory usage, container issues
- **High**: Infrastructure alerts, network latency, disk usage
- **Contacts**: `sre@freeagentics.com`, `sre-lead@freeagentics.com`

### Backend Team
- **Critical**: API system failures
- **High**: API performance issues, knowledge graph problems
- **Medium**: General backend performance warnings
- **Contacts**: `backend@freeagentics.com`, `backend-lead@freeagentics.com`

### Agents Team
- **Critical**: Agent coordination failures
- **High**: Agent error rates, coordination timeouts
- **Medium**: Agent memory warnings, belief system issues
- **Contacts**: `agents@freeagentics.com`, `agents-lead@freeagentics.com`

### Security Team
- **Critical**: Security anomalies, access violations
- **High**: Authentication failures, security warnings
- **Contacts**: `security@freeagentics.com`, `ciso@freeagentics.com`

### Database Team
- **High**: Database connection issues, query performance
- **Contacts**: `database@freeagentics.com`, `database-lead@freeagentics.com`

### Product Team
- **Medium**: User interaction rates, response quality
- **Contacts**: `product@freeagentics.com`

## Alert Inhibition Rules

### System-Level Inhibition
- **System Down** → Inhibits all other alerts
- **Critical Memory** → Inhibits memory warnings
- **High Error Rate** → Inhibits API response time alerts

### Component-Level Inhibition
- **Critical severity** → Inhibits high/medium for same component
- **High severity** → Inhibits medium for same component
- **Agent Coordination Failure** → Inhibits individual agent alerts

## Escalation Procedures

### Critical Alert Escalation
1. **Immediate**: PagerDuty notification
2. **Immediate**: Slack notification to relevant channel
3. **Immediate**: Email to critical escalation list
4. **5 minutes**: If unacknowledged, escalate to management
5. **15 minutes**: If unresolved, escalate to C-level

### High Alert Escalation
1. **Immediate**: Slack notification to team channel
2. **15 minutes**: Email to team lead
3. **1 hour**: If unresolved, escalate to manager
4. **2 hours**: If unresolved, escalate to critical level

### Medium Alert Escalation
1. **Immediate**: Slack notification
2. **30 minutes**: If unacknowledged, reminder notification
3. **2 hours**: If unresolved, escalate to high priority
4. **4 hours**: If unresolved, escalate to management

## Environment Variables

Configure these environment variables for proper alert routing:

```bash
# SMTP Configuration
ALERTMANAGER_SMTP_HOST=smtp.freeagentics.com:587
ALERTMANAGER_SMTP_FROM=alerts@freeagentics.com
ALERTMANAGER_SMTP_USER=alerts@freeagentics.com
ALERTMANAGER_SMTP_PASSWORD=<secure_password>

# Slack Configuration
ALERTMANAGER_SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# PagerDuty Configuration
PAGERDUTY_CRITICAL_KEY=<pagerduty_critical_integration_key>
PAGERDUTY_SYSTEM_DOWN_KEY=<pagerduty_system_down_integration_key>
PAGERDUTY_SECURITY_KEY=<pagerduty_security_integration_key>

# Email Lists
CRITICAL_EMAIL_LIST=oncall@freeagentics.com,sre@freeagentics.com
SYSTEM_DOWN_EMAIL_LIST=oncall@freeagentics.com,engineering@freeagentics.com,cto@freeagentics.com
SECURITY_EMAIL_LIST=security@freeagentics.com,ciso@freeagentics.com
HIGH_SEVERITY_EMAIL_LIST=alerts@freeagentics.com
BACKEND_HIGH_EMAIL_LIST=backend-lead@freeagentics.com
DATABASE_HIGH_EMAIL_LIST=database-lead@freeagentics.com
AGENTS_HIGH_EMAIL_LIST=agents-lead@freeagentics.com
SECURITY_HIGH_EMAIL_LIST=security-lead@freeagentics.com
SRE_HIGH_EMAIL_LIST=sre-lead@freeagentics.com
```

## Key Alert Definitions

### System Health Alerts
- **FreeAgenticsSystemDown**: Backend service unavailable
- **HighSystemErrorRate**: Error rate > 10% for 2 minutes
- **SystemMemoryUsageCritical**: Memory usage > 2GB for 5 minutes

### Agent Coordination Alerts
- **AgentCoordinationFailure**: Active agents > 50 for 1 minute
- **AgentCoordinationTimeout**: Timeout rate > 5% for 2 minutes
- **HighConcurrentSessions**: Concurrent sessions > 100 for 5 minutes

### Performance Alerts
- **HighAPIResponseTime**: P95 response time > 500ms for 3 minutes
- **SlowCoalitionFormation**: P90 coordination time > 2s for 5 minutes
- **AgentMemoryUsageWarning**: Agent memory > 30MB for 10 minutes

### Security Alerts
- **SecurityAnomalyDetected**: Security anomaly detected (immediate)
- **HighAuthenticationFailures**: Auth failure rate > 0.5/sec for 1 minute
- **AccessViolationDetected**: Access violation detected (immediate)

### Business Logic Alerts
- **LowUserInteractionRate**: User interaction rate < 0.01/hour for 30 minutes
- **ResponseQualityDegradation**: Response quality < 70% for 10 minutes
- **LowInferenceOperationRate**: Inference rate < 0.1 ops/sec for 10 minutes

## Maintenance and Silence Rules

### Scheduled Maintenance
- Use AlertManager silence rules for planned maintenance
- Notify teams 24 hours in advance
- Document maintenance windows in runbooks

### Emergency Silencing
- Critical alerts should rarely be silenced
- High/medium alerts can be silenced during incident response
- All silences must have expiration times

## Runbook URLs

Each alert includes runbook URLs for quick reference:
- **System alerts**: `https://docs.freeagentics.com/runbooks/system-*`
- **Agent alerts**: `https://docs.freeagentics.com/runbooks/agent-*`
- **Security alerts**: `https://docs.freeagentics.com/runbooks/security-*`
- **Performance alerts**: `https://docs.freeagentics.com/runbooks/performance-*`

## Monitoring the Monitoring

### AlertManager Health
- Monitor AlertManager service uptime
- Check notification delivery rates
- Verify alert routing effectiveness

### Alert Fatigue Prevention
- Regular review of alert frequency
- Consolidation of duplicate alerts
- Threshold tuning based on operational experience

### Metrics to Track
- Alert resolution time by severity
- False positive rate by alert type
- Team response time compliance
- Escalation frequency patterns

## Troubleshooting

### Common Issues
1. **Receiver not found**: Check receiver name matches exactly
2. **Alerts not routing**: Verify label matching in routes
3. **Notifications not sent**: Check environment variables
4. **Inhibition not working**: Verify label matching in inhibit rules

### Validation Commands
```bash
# Validate configuration
python test_alertmanager_config.py

# Check AlertManager status
curl http://localhost:9093/api/v1/status

# Test alert routing
curl -X POST http://localhost:9093/api/v1/alerts
```

## Best Practices

### Alert Design
- Use consistent labeling across all alerts
- Include meaningful descriptions and summaries
- Provide actionable runbook links
- Set appropriate thresholds to avoid noise

### Team Communication
- Maintain up-to-date contact information
- Regular testing of notification channels
- Clear escalation procedures for each team
- Documentation of response procedures

### Continuous Improvement
- Regular review of alert effectiveness
- Feedback collection from on-call teams
- Threshold adjustment based on operational data
- Regular testing of escalation procedures

---

**Last Updated**: 2024-07-15
**Configuration Version**: 1.0
**Contact**: sre@freeagentics.com
