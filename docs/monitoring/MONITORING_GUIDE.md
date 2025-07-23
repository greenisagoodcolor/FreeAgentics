# FreeAgentics Monitoring and Observability Guide

## Overview

This guide provides comprehensive documentation for monitoring the FreeAgentics multi-agent system in production. Our monitoring stack is designed to provide real-time visibility into system health, performance, and business metrics while ensuring rapid incident response and proactive capacity planning.

## Table of Contents

1. [Observability Stack Architecture](#observability-stack-architecture)
2. [Metrics Overview](#metrics-overview)
3. [Dashboard Descriptions](#dashboard-descriptions)
4. [Alert Configurations](#alert-configurations)
5. [SLI/SLO Definitions](#slislo-definitions)
6. [Log Aggregation and Analysis](#log-aggregation-and-analysis)
7. [Troubleshooting with Metrics](#troubleshooting-with-metrics)
8. [Performance Baselines](#performance-baselines)
9. [Operational Procedures](#operational-procedures)

## Observability Stack Architecture

### Components

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Applications  │────▶│   Prometheus    │────▶│     Grafana     │
│   (Exporters)   │     │   (Metrics)     │     │  (Dashboards)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         │                       ▼                        │
         │              ┌─────────────────┐               │
         │              │  AlertManager   │◀──────────────┘
         │              │    (Alerts)     │
         │              └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   Log Files     │────▶│ Log Aggregation │
│                 │     │   (Analysis)    │
└─────────────────┘     └─────────────────┘
```

### Data Flow

1. **Metrics Collection**: Applications expose metrics via Prometheus exporters
2. **Storage**: Prometheus scrapes and stores time-series metrics
3. **Visualization**: Grafana queries Prometheus for dashboard display
4. **Alerting**: AlertManager processes alerts from Prometheus rules
5. **Logs**: Log aggregation system collects and analyzes application logs

### Infrastructure Requirements

- **Prometheus**: 2 CPU cores, 4GB RAM, 100GB disk
- **Grafana**: 1 CPU core, 2GB RAM, 10GB disk
- **AlertManager**: 1 CPU core, 1GB RAM, 5GB disk
- **Log Storage**: 500GB disk (90-day retention)

## Metrics Overview

### System Metrics

#### Availability Metrics
- `up{job="freeagentics-backend"}` - System availability (1=up, 0=down)
- `http_requests_total` - Total HTTP requests by endpoint and status
- `http_request_duration_seconds` - Request latency histogram

#### Resource Metrics
- `freeagentics_system_memory_usage_bytes` - Total system memory usage
- `node_cpu_seconds_total` - CPU usage by mode
- `node_filesystem_avail_bytes` - Available disk space

### Agent Coordination Metrics

#### Coordination Performance
- `freeagentics_system_active_agents_total` - Currently active agents
- `freeagentics_agent_coordination_requests_total` - Coordination requests by status
- `freeagentics_agent_coordination_duration_seconds` - Coordination duration histogram
- `freeagentics_agent_coordination_concurrent_sessions` - Concurrent coordination sessions
- `freeagentics_agent_coordination_errors_total` - Coordination errors by type

#### Memory Usage
- `freeagentics_agent_memory_usage_bytes` - Per-agent memory consumption
- `freeagentics_agent_memory_growth_rate` - Memory growth rate per agent

### Belief System Metrics

- `freeagentics_belief_free_energy_current` - Current free energy value
- `freeagentics_belief_convergence_time_seconds` - Time to belief convergence
- `freeagentics_belief_accuracy_ratio` - Belief prediction accuracy
- `freeagentics_belief_prediction_errors_total` - Total prediction errors

### Business Metrics

- `freeagentics_business_user_interactions_total` - User interaction count
- `freeagentics_business_inference_operations_total` - Inference operations
- `freeagentics_business_response_quality_score` - Response quality (0-1)
- `freeagentics_system_knowledge_graph_nodes_total` - Knowledge graph size

### Security Metrics

- `freeagentics_security_authentication_attempts_total` - Auth attempts by outcome
- `freeagentics_security_anomaly_detections_total` - Security anomalies by type
- `freeagentics_security_access_violations_total` - Access violations by resource

## Dashboard Descriptions

### 1. System Overview Dashboard

**Purpose**: High-level system health monitoring
**URL**: `https://grafana.freeagentics.com/d/system-overview`
**Refresh Rate**: 30 seconds

**Key Panels**:
- System Status (Up/Down indicator)
- Active Agents Count with trend
- Memory Usage (System and Agents)
- Request Rate and Error Rate
- Knowledge Graph Growth
- Alert Summary

**Use Cases**:
- Executive dashboard for system health
- Incident response starting point
- Daily operations monitoring

### 2. Agent Coordination Dashboard

**Purpose**: Multi-agent system performance monitoring
**URL**: `https://grafana.freeagentics.com/d/agent-coordination`
**Refresh Rate**: 10 seconds

**Key Panels**:
- Active Agents by Type
- Coordination Success Rate
- Coordination Duration (P50, P90, P95)
- Concurrent Sessions
- Belief System Metrics
- Agent Error Rates

**Variables**:
- `agent_id` - Filter by specific agents
- `agent_type` - Filter by agent type

**Use Cases**:
- Agent team performance analysis
- Coordination bottleneck identification
- Belief system optimization

### 3. Memory Usage Heatmap

**Purpose**: Detailed memory usage analysis
**URL**: `https://grafana.freeagentics.com/d/memory-heatmap`
**Refresh Rate**: 30 seconds

**Key Panels**:
- Agent Memory Heatmap
- Top 10 Memory Consumers
- Memory Growth Trends
- Agents Over Limit (30MB threshold)
- Memory by Agent Type

**Use Cases**:
- Memory leak detection
- Resource allocation planning
- Performance optimization

### 4. API Performance Dashboard

**Purpose**: API latency and throughput monitoring
**URL**: `https://grafana.freeagentics.com/d/api-performance`
**Refresh Rate**: 10 seconds

**Key Panels**:
- Response Time Percentiles
- Request Rate by Endpoint
- Error Rate by Status Code
- Business Metrics
- Authentication Performance

**Variables**:
- `endpoint` - Filter by API endpoint
- `method` - Filter by HTTP method

**Use Cases**:
- SLA compliance monitoring
- Performance regression detection
- User experience optimization

### 5. Capacity Planning Dashboard

**Purpose**: Resource utilization and forecasting
**URL**: `https://grafana.freeagentics.com/d/capacity-planning`
**Refresh Rate**: 1 minute

**Key Panels**:
- Resource Usage Trends
- Agent Capacity Utilization
- Growth Forecasting
- Scale Trigger Indicators
- Performance Trend Analysis

**Variables**:
- `forecast_period` - 1h, 6h, 24h forecasting

**Use Cases**:
- Infrastructure scaling decisions
- Budget planning
- Performance capacity analysis

## Alert Configurations

### Alert Severity Levels

| Severity | Response Time | Notification Channels | Examples |
|----------|--------------|----------------------|----------|
| Critical | < 5 minutes | PagerDuty + Slack + Email | System down, Memory > 34.5MB/agent |
| High | < 15 minutes | Slack + Email | API P95 > 500ms, Error rate > 10% |
| Medium | < 30 minutes | Slack | Performance warnings, Resource usage |

### Critical Alerts

#### System Health
- **FreeAgenticsSystemDown**: Backend service unavailable for 30s
- **SystemMemoryUsageCritical**: System memory > 2GB for 5m
- **AgentCoordinationFailure**: Active agents > 50 (hard limit)

#### Security
- **SecurityAnomalyDetected**: Any security anomaly (immediate)
- **AccessViolationDetected**: Unauthorized access attempt (immediate)

### High Severity Alerts

#### Performance
- **HighAPIResponseTime**: P95 latency > 500ms for 3m
- **HighAgentErrorRate**: Agent errors > 5% for 2m
- **HighDatabaseConnections**: Connections > 80 for 5m

#### Availability
- **HighSystemErrorRate**: HTTP 5xx errors > 10% for 2m
- **AgentCoordinationTimeout**: Timeout rate > 5% for 2m

### Medium Severity Alerts

#### Resource Usage
- **AgentMemoryUsageWarning**: Agent memory > 30MB for 10m
- **SlowCoalitionFormation**: P90 coordination > 2s for 5m
- **KnowledgeGraphGrowthRate**: Growth > 1000 nodes/hour

#### Business Logic
- **LowUserInteractionRate**: Interactions < 0.01/hour for 30m
- **ResponseQualityDegradation**: Quality score < 70% for 10m

## SLI/SLO Definitions

### Service Level Indicators (SLIs)

#### 1. Availability SLI
- **Definition**: System uptime percentage
- **Measurement**: `avg_over_time(up{job="freeagentics-backend"}[5m])`
- **Good Events**: HTTP 200 responses
- **Total Events**: All HTTP responses

#### 2. Latency SLI
- **Definition**: Requests served within acceptable latency
- **Measurement**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- **Good Events**: Requests < 500ms
- **Total Events**: All requests

#### 3. Quality SLI
- **Definition**: Successful request percentage
- **Measurement**: Success rate excluding 5xx errors
- **Good Events**: HTTP 2xx, 3xx, 4xx responses
- **Total Events**: All HTTP responses

#### 4. Coordination SLI
- **Definition**: Agent coordination success rate
- **Measurement**: Successful coordination requests / Total requests
- **Good Events**: Successful coordinations
- **Total Events**: All coordination attempts

### Service Level Objectives (SLOs)

| SLO | Target | Error Budget | Measurement Window | Alert Threshold |
|-----|--------|--------------|-------------------|-----------------|
| Availability | 99.9% | 43.2 min/month | 30 days | < 99.5% |
| Latency | P95 < 500ms | 5% requests | 24 hours | P95 > 600ms |
| Quality | 99% success | 1% errors | 24 hours | < 90% |
| Coordination | 95% success | 5% failures | 1 hour | < 90% |
| Memory | < 30MB/agent | 10% agents | 1 hour | > 34.5MB |

### Error Budget Policies

#### Fast Burn (> 10x burn rate)
- Page on-call engineer immediately
- Halt non-critical deployments
- Escalate to engineering manager

#### Slow Burn (> 1x burn rate)
- Notify development team
- Review deployment frequency
- Prioritize reliability improvements

#### Budget Exhausted
- Halt all deployments
- Focus on reliability
- Executive notification

## Log Aggregation and Analysis

### Log Sources

1. **Application Logs**
   - Location: `/var/log/freeagentics/*.log`
   - Format: JSON structured logging
   - Retention: 90 days

2. **Agent Logs**
   - Location: `/var/log/freeagentics/agents/*.log`
   - Fields: agent_id, coordination_id, belief_state
   - Rotation: Daily, compressed after 7 days

3. **Security Logs**
   - Location: `/var/log/freeagentics/security/*.log`
   - Fields: user_id, action, resource, outcome
   - Retention: 365 days (compliance)

### Log Analysis Pipeline

```python
# Log processing pipeline
1. Collection: Filebeat/Fluentd agents
2. Parsing: JSON/Regex extractors
3. Enrichment: Add metadata, geo-location
4. Storage: Elasticsearch/S3
5. Analysis: Kibana/Custom dashboards
```

### Key Log Queries

#### Error Investigation
```
level:ERROR AND component:agent AND timestamp:[now-1h TO now]
| stats count by agent_id, error_type
```

#### Performance Analysis
```
component:api AND duration:>500
| timechart avg(duration) by endpoint
```

#### Security Audit
```
component:security AND action:authentication
| stats count by outcome, user_id
```

## Troubleshooting with Metrics

### Common Issues and Metrics

#### 1. High Memory Usage
**Symptoms**: Agent memory > 30MB
**Metrics to Check**:
- `freeagentics_agent_memory_usage_bytes`
- `freeagentics_agent_memory_growth_rate`
- `go_memstats_alloc_bytes`

**Investigation Steps**:
1. Check memory heatmap dashboard
2. Identify top memory consumers
3. Review recent deployments
4. Check for memory leaks in logs

#### 2. Coordination Timeouts
**Symptoms**: Timeout rate > 5%
**Metrics to Check**:
- `freeagentics_agent_coordination_errors_total{error_type="timeout"}`
- `freeagentics_agent_coordination_duration_seconds`
- `freeagentics_agent_coordination_concurrent_sessions`

**Investigation Steps**:
1. Check coordination dashboard
2. Review network latency
3. Analyze agent workload distribution
4. Check for resource contention

#### 3. API Performance Degradation
**Symptoms**: P95 latency > 500ms
**Metrics to Check**:
- `http_request_duration_seconds`
- `http_requests_total`
- Database query metrics

**Investigation Steps**:
1. Check API performance dashboard
2. Identify slow endpoints
3. Review database performance
4. Check external dependencies

#### 4. Belief System Anomalies
**Symptoms**: Free energy outside normal range
**Metrics to Check**:
- `freeagentics_belief_free_energy_current`
- `freeagentics_belief_accuracy_ratio`
- `freeagentics_belief_prediction_errors_total`

**Investigation Steps**:
1. Check belief system metrics
2. Review recent model updates
3. Analyze input data quality
4. Check for configuration changes

### Metric Correlation Patterns

#### Memory ↔ Performance
- High memory usage often correlates with increased latency
- Check both metrics when investigating performance issues

#### Coordination ↔ Agents
- More active agents increase coordination complexity
- Monitor both when scaling agent count

#### Errors ↔ Quality
- High error rates impact response quality scores
- Track both for user experience analysis

## Performance Baselines

### System-Level Baselines

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| Memory Usage | 1.5GB | 1.8GB | 2.0GB |
| CPU Usage | 40% | 70% | 90% |
| Disk Usage | 60% | 80% | 90% |
| Error Rate | <1% | 5% | 10% |

### Agent Coordination Baselines

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| Active Agents | 15 | 40 | 50 |
| Coordination Success | >95% | <95% | <90% |
| Coordination P95 | <1.5s | >1.8s | >2.0s |
| Timeout Rate | <2% | >3% | >5% |

### Memory Usage Baselines

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| Per-Agent Average | 20MB | 30MB | 34.5MB |
| Per-Agent Max | 30MB | 32MB | 34.5MB |
| Total Agent Memory | 300MB | 1.2GB | 1.5GB |

### API Performance Baselines

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| P95 Response Time | <300ms | >400ms | >500ms |
| Request Rate | 50 req/s | 300 req/s | 500 req/s |
| Error Rate | <1% | >5% | >10% |

## Operational Procedures

### Daily Operations

#### Morning Health Check (9:00 AM)
1. Review system overview dashboard
2. Check overnight alerts
3. Verify SLO compliance
4. Review error budgets

#### Shift Handover
1. Document any ongoing issues
2. Update incident tickets
3. Review scheduled maintenance
4. Brief incoming team

### Weekly Tasks

#### Performance Review (Mondays)
1. Analyze weekly performance trends
2. Review SLO compliance reports
3. Update baselines if needed
4. Plan capacity adjustments

#### Alert Review (Fridays)
1. Analyze alert frequency
2. Tune thresholds if needed
3. Update runbooks
4. Review false positives

### Monthly Tasks

#### Capacity Planning
1. Review growth trends
2. Update forecasts
3. Plan infrastructure changes
4. Budget adjustments

#### SLO Review
1. Analyze error budget consumption
2. Update SLO targets if needed
3. Review with stakeholders
4. Plan reliability improvements

### Incident Response

#### Severity Classification
- **Sev 1**: Complete outage, data loss risk
- **Sev 2**: Significant degradation, SLO violation
- **Sev 3**: Minor degradation, single component
- **Sev 4**: No user impact, proactive fix

#### Response Process
1. **Detect**: Alert fires or issue reported
2. **Triage**: Assess severity and impact
3. **Communicate**: Notify stakeholders
4. **Investigate**: Use dashboards and logs
5. **Mitigate**: Apply immediate fixes
6. **Resolve**: Implement permanent solution
7. **Review**: Post-incident analysis

### Maintenance Procedures

#### Planned Maintenance
1. Schedule during low-traffic windows
2. Notify users 24 hours in advance
3. Create maintenance silence in AlertManager
4. Execute runbook procedures
5. Verify system health post-maintenance

#### Emergency Maintenance
1. Assess risk vs impact
2. Get approval from on-call lead
3. Document in incident ticket
4. Execute with rollback plan ready
5. Comprehensive testing post-fix

## Best Practices

### Monitoring Hygiene
- Review and tune alerts monthly
- Keep dashboards focused and fast
- Document all custom queries
- Regular backup of configurations

### Team Collaboration
- Share dashboard links in incidents
- Document investigation steps
- Regular training on tools
- Rotate on-call responsibilities

### Continuous Improvement
- Track MTTR (Mean Time To Resolve)
- Analyze repeat incidents
- Automate common remediation
- Regular disaster recovery drills

## Contact Information

- **SRE Team**: sre@freeagentics.com
- **On-Call**: +1-555-ONCALL-1
- **Slack**: #sre-support
- **Documentation**: https://docs.freeagentics.com

---

**Last Updated**: 2025-01-15
**Version**: 1.0
**Next Review**: 2025-02-15
