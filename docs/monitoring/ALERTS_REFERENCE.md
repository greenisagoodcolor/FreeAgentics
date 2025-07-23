# FreeAgentics Alerts Reference Guide

## Overview

This document provides a comprehensive reference for all alerts configured in the FreeAgentics monitoring system. Each alert includes severity, thresholds, response procedures, and troubleshooting steps.

## Table of Contents

1. [Alert Severity Matrix](#alert-severity-matrix)
2. [Critical Alerts](#critical-alerts)
3. [High Severity Alerts](#high-severity-alerts)
4. [Medium Severity Alerts](#medium-severity-alerts)
5. [Security Alerts](#security-alerts)
6. [Alert Response Procedures](#alert-response-procedures)
7. [Alert Tuning Guidelines](#alert-tuning-guidelines)

## Alert Severity Matrix

| Severity | Response Time | Notification Method | Escalation | Page |
|----------|--------------|-------------------|------------|------|
| Critical | < 5 min | PagerDuty + Slack + Email | Immediate | Yes |
| High | < 15 min | Slack + Email | 15 min | No |
| Medium | < 30 min | Slack | 30 min | No |
| Info | < 2 hours | Email | None | No |

## Critical Alerts

### FreeAgenticsSystemDown

**Description**: The FreeAgentics backend service is completely unavailable
**Severity**: Critical
**Threshold**: `up{job="freeagentics-backend"} == 0` for 30 seconds
**Team**: SRE
**Runbook**: https://docs.freeagentics.com/runbooks/system-down

**Impact**:
- Complete service outage
- All agents offline
- No API responses

**Response Steps**:
1. Check infrastructure status
2. Verify network connectivity
3. Check recent deployments
4. Review system logs
5. Initiate rollback if needed

**Common Causes**:
- Infrastructure failure
- Network partition
- Failed deployment
- Resource exhaustion

### SystemMemoryUsageCritical

**Description**: System memory usage exceeds critical threshold
**Severity**: Critical
**Threshold**: Memory usage > 2GB for 5 minutes
**Team**: SRE
**Runbook**: https://docs.freeagentics.com/runbooks/memory-usage

**Impact**:
- System instability
- Potential OOM kills
- Performance degradation

**Response Steps**:
1. Check memory consumption by component
2. Identify memory leaks
3. Scale resources if needed
4. Restart affected services
5. Review recent code changes

**Common Causes**:
- Memory leaks
- Increased load
- Large data processing
- Configuration issues

### AgentCoordinationFailure

**Description**: Active agent count exceeds coordination limit
**Severity**: Critical
**Threshold**: Active agents > 50 for 1 minute
**Team**: Agents
**Runbook**: https://docs.freeagentics.com/runbooks/agent-coordination

**Impact**:
- Coordination failures
- Agent timeouts
- System overload
- Degraded performance

**Response Steps**:
1. Check active agent count
2. Identify stuck agents
3. Implement agent throttling
4. Review coordination queue
5. Scale coordination service

**Common Causes**:
- Agent proliferation
- Coordination bottleneck
- Network issues
- Bug in agent lifecycle

## High Severity Alerts

### HighAPIResponseTime

**Description**: API response time exceeds acceptable threshold
**Severity**: High
**Threshold**: P95 response time > 500ms for 3 minutes
**Team**: Backend
**Runbook**: https://docs.freeagentics.com/runbooks/api-performance

**Impact**:
- Poor user experience
- SLO violation risk
- Potential timeouts

**Response Steps**:
1. Identify slow endpoints
2. Check database performance
3. Review external dependencies
4. Analyze request patterns
5. Implement caching if needed

**Metrics to Check**:
```promql
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket[5m])
)
```

### HighAgentErrorRate

**Description**: Agent error rate exceeds threshold
**Severity**: High
**Threshold**: Error rate > 5% for 2 minutes
**Team**: Agents
**Runbook**: https://docs.freeagentics.com/runbooks/agent-errors

**Impact**:
- Agent failures
- Incomplete operations
- Data inconsistency

**Response Steps**:
1. Check error logs
2. Identify error patterns
3. Review recent deployments
4. Check agent configurations
5. Rollback if necessary

**Query for Investigation**:
```promql
rate(freeagentics_agent_errors_total[5m])
/ rate(freeagentics_agent_operations_total[5m])
```

### HighDatabaseConnections

**Description**: Database connection count approaching limit
**Severity**: High
**Threshold**: Connections > 80 for 5 minutes
**Team**: Database
**Runbook**: https://docs.freeagentics.com/runbooks/database-connections

**Impact**:
- Connection exhaustion
- Application errors
- Performance degradation

**Response Steps**:
1. Check connection pool status
2. Identify connection leaks
3. Review slow queries
4. Increase pool size if needed
5. Restart connection pools

### FreeEnergyAnomaly

**Description**: Belief system free energy outside normal range
**Severity**: High
**Threshold**: Free energy < 0.1 or > 10 for 5 minutes
**Team**: Agents
**Runbook**: https://docs.freeagentics.com/runbooks/belief-system

**Impact**:
- Belief system instability
- Poor predictions
- Agent confusion

**Response Steps**:
1. Check belief system metrics
2. Review input data quality
3. Verify model parameters
4. Check for data anomalies
5. Reset belief state if needed

## Medium Severity Alerts

### AgentMemoryUsageWarning

**Description**: Individual agent memory usage elevated
**Severity**: Medium
**Threshold**: Agent memory > 30MB for 10 minutes
**Team**: Agents
**Runbook**: https://docs.freeagentics.com/runbooks/agent-memory

**Impact**:
- Resource pressure
- Potential memory limit
- Performance impact

**Response Steps**:
1. Identify high-memory agents
2. Check agent workload
3. Review memory patterns
4. Optimize if needed
5. Consider agent restart

**Investigation Query**:
```promql
topk(10, freeagentics_agent_memory_usage_bytes / (1024*1024))
```

### SlowCoalitionFormation

**Description**: Coalition formation taking longer than expected
**Severity**: Medium
**Threshold**: P90 coordination time > 2s for 5 minutes
**Team**: Agents
**Runbook**: https://docs.freeagentics.com/runbooks/coalition-formation

**Impact**:
- Delayed operations
- Queue buildup
- User experience impact

**Response Steps**:
1. Check coordination metrics
2. Review agent availability
3. Analyze coordination patterns
4. Optimize algorithms
5. Scale if needed

### KnowledgeGraphGrowthRate

**Description**: Knowledge graph growing faster than expected
**Severity**: Medium
**Threshold**: Growth > 1000 nodes/hour for 15 minutes
**Team**: Backend
**Runbook**: https://docs.freeagentics.com/runbooks/knowledge-graph

**Impact**:
- Storage pressure
- Query performance
- Memory usage

**Response Steps**:
1. Check growth patterns
2. Review data sources
3. Implement pruning
4. Optimize storage
5. Scale if needed

### LowUserInteractionRate

**Description**: User interaction rate below expected threshold
**Severity**: Medium
**Threshold**: Interactions < 0.01/hour for 30 minutes
**Team**: Product
**Runbook**: https://docs.freeagentics.com/runbooks/user-interactions

**Impact**:
- Business metric decline
- Potential system issue
- User experience problem

**Response Steps**:
1. Check system availability
2. Review recent changes
3. Analyze user patterns
4. Check external factors
5. Notify product team

### ResponseQualityDegradation

**Description**: Response quality score below acceptable level
**Severity**: Medium
**Threshold**: Quality score < 70% for 10 minutes
**Team**: Product
**Runbook**: https://docs.freeagentics.com/runbooks/response-quality

**Impact**:
- Poor user experience
- Trust degradation
- Business impact

**Response Steps**:
1. Review quality metrics
2. Check model performance
3. Analyze failure patterns
4. Review training data
5. Consider rollback

## Security Alerts

### SecurityAnomalyDetected

**Description**: Security anomaly detected by monitoring system
**Severity**: Critical
**Threshold**: Any anomaly detection (immediate)
**Team**: Security
**Runbook**: https://docs.freeagentics.com/runbooks/security-anomaly

**Impact**:
- Potential security breach
- Data exposure risk
- Compliance violation

**Response Steps**:
1. **IMMEDIATE**: Isolate affected systems
2. Preserve evidence
3. Check access logs
4. Review anomaly details
5. Initiate incident response

### HighAuthenticationFailures

**Description**: Elevated authentication failure rate
**Severity**: High
**Threshold**: Failure rate > 0.5/sec for 1 minute
**Team**: Security
**Runbook**: https://docs.freeagentics.com/runbooks/authentication-failures

**Impact**:
- Potential brute force
- Account compromise
- Service disruption

**Response Steps**:
1. Check failure patterns
2. Identify source IPs
3. Review target accounts
4. Implement rate limiting
5. Block suspicious sources

### AccessViolationDetected

**Description**: Unauthorized access attempt detected
**Severity**: High
**Threshold**: Any access violation (immediate)
**Team**: Security
**Runbook**: https://docs.freeagentics.com/runbooks/access-violations

**Impact**:
- Security breach risk
- Data exposure
- Compliance issue

**Response Steps**:
1. Identify violation details
2. Check user permissions
3. Review access patterns
4. Revoke if necessary
5. Audit permissions

## Alert Response Procedures

### Initial Response

1. **Acknowledge Alert**
   - Respond within SLA time
   - Claim ownership in AlertManager
   - Update incident channel

2. **Assess Impact**
   - Check affected services
   - Estimate user impact
   - Determine severity

3. **Communicate Status**
   - Update status page
   - Notify stakeholders
   - Create incident ticket

### Investigation Process

1. **Gather Context**
   ```bash
   # Check recent deployments
   kubectl get deployments -n freeagentics --sort-by=.metadata.creationTimestamp

   # View recent logs
   kubectl logs -n freeagentics -l app=freeagentics --tail=100

   # Check metrics
   curl -s http://prometheus:9090/api/v1/query?query=up
   ```

2. **Use Dashboards**
   - Start with overview dashboard
   - Drill into specific component
   - Correlate with other metrics
   - Check historical data

3. **Analyze Logs**
   ```bash
   # Search error logs
   grep ERROR /var/log/freeagentics/*.log | tail -50

   # Check specific component
   grep -A 5 -B 5 "agent_id=agent-123" /var/log/freeagentics/agents.log
   ```

### Mitigation Actions

#### Quick Fixes
- Restart affected service
- Scale up resources
- Enable circuit breaker
- Redirect traffic

#### Rollback Procedures
```bash
# Rollback deployment
kubectl rollout undo deployment/freeagentics-backend -n freeagentics

# Verify rollback
kubectl rollout status deployment/freeagentics-backend -n freeagentics
```

#### Emergency Procedures
- Enable maintenance mode
- Drain traffic
- Isolate affected components
- Preserve state for debugging

### Post-Incident

1. **Document Timeline**
   - Alert time
   - Response time
   - Actions taken
   - Resolution time

2. **Root Cause Analysis**
   - What happened?
   - Why did it happen?
   - How was it detected?
   - How was it resolved?

3. **Action Items**
   - Fix root cause
   - Improve monitoring
   - Update runbooks
   - Prevent recurrence

## Alert Tuning Guidelines

### When to Tune Alerts

- High false positive rate (> 20%)
- Alerts firing too late
- Missing critical issues
- Team feedback

### Tuning Process

1. **Collect Data**
   ```promql
   # Alert firing frequency
   count_over_time(ALERTS{alertname="HighAPIResponseTime"}[7d])

   # Actual impact correlation
   count_over_time(http_requests_total{status=~"5.."}[7d])
   ```

2. **Analyze Patterns**
   - Time of day patterns
   - Day of week patterns
   - Correlation with events
   - Business impact

3. **Adjust Thresholds**
   ```yaml
   # Before
   expr: rate(errors_total[5m]) > 0.1

   # After (more specific)
   expr: rate(errors_total[5m]) > 0.1 AND rate(requests_total[5m]) > 10
   ```

4. **Test Changes**
   - Deploy to staging
   - Monitor for a week
   - Gather feedback
   - Deploy to production

### Alert Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| False Positive Rate | < 10% | False alerts / Total alerts |
| Detection Time | < 2 min | Time to alert after issue |
| Resolution Time | < 30 min | Time from alert to resolution |
| Coverage | > 95% | Incidents with alerts / Total incidents |

### Best Practices

1. **Alert Design**
   - Alert on symptoms, not causes
   - Include context in description
   - Provide clear runbook links
   - Use meaningful alert names

2. **Threshold Setting**
   - Base on historical data
   - Consider business hours
   - Account for seasonality
   - Leave room for growth

3. **Maintenance**
   - Review alerts monthly
   - Remove obsolete alerts
   - Update based on incidents
   - Train team on changes

## Appendix: Common Queries

### System Health
```promql
# Overall availability
avg_over_time(up{job="freeagentics-backend"}[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Agent Metrics
```promql
# Active agents
freeagentics_system_active_agents_total

# Agent memory usage
topk(10, freeagentics_agent_memory_usage_bytes / (1024*1024))

# Coordination success rate
rate(freeagentics_agent_coordination_requests_total{status="success"}[5m])
/ rate(freeagentics_agent_coordination_requests_total[5m])
```

### Business Metrics
```promql
# User interaction rate
rate(freeagentics_business_user_interactions_total[1h])

# Response quality
freeagentics_business_response_quality_score

# Inference operations
rate(freeagentics_business_inference_operations_total[5m])
```

---

**Last Updated**: 2025-01-15
**Version**: 1.0
**Contact**: sre@freeagentics.com
