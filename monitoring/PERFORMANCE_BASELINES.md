# FreeAgentics Performance Baselines & SLIs/SLOs

## Overview

This document establishes performance baselines for FreeAgentics production monitoring and defines Service Level Indicators (SLIs) and Service Level Objectives (SLOs) for operational excellence.

## Performance Baselines

### System-Level Baselines

#### System Availability
- **Metric**: `up{job="freeagentics-backend"}`
- **Baseline**: 99.9% uptime
- **Threshold**: < 99.5% triggers alert
- **Measurement Window**: 24 hours

#### Memory Usage
- **Metric**: `freeagentics_system_memory_usage_bytes`
- **Baseline**: 1.5GB average, 2.5GB peak
- **Critical Threshold**: > 2GB for 5 minutes
- **Warning Threshold**: > 1.8GB for 10 minutes

#### CPU Usage
- **Metric**: `100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`
- **Baseline**: 40% average, 80% peak
- **Critical Threshold**: > 90% for 5 minutes
- **Warning Threshold**: > 70% for 10 minutes

#### Disk Usage
- **Metric**: `(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100`
- **Baseline**: 60% average, 85% peak
- **Critical Threshold**: > 90%
- **Warning Threshold**: > 80%

### Agent Coordination Baselines

#### Active Agents
- **Metric**: `freeagentics_system_active_agents_total`
- **Baseline**: 15 agents average, 30 agents peak
- **Critical Threshold**: > 50 agents (coordination limit)
- **Warning Threshold**: > 40 agents

#### Coordination Success Rate
- **Metric**: `rate(freeagentics_agent_coordination_requests_total{status="success"}[5m]) / rate(freeagentics_agent_coordination_requests_total[5m])`
- **Baseline**: 95% success rate
- **Critical Threshold**: < 90% for 5 minutes
- **Warning Threshold**: < 95% for 10 minutes

#### Coordination Duration
- **Metric**: `histogram_quantile(0.95, rate(freeagentics_agent_coordination_duration_seconds_bucket[5m]))`
- **Baseline**: P95 < 1.5 seconds
- **Critical Threshold**: P95 > 2 seconds for 5 minutes
- **Warning Threshold**: P95 > 1.8 seconds for 10 minutes

#### Coordination Timeout Rate
- **Metric**: `rate(freeagentics_agent_coordination_errors_total{error_type="timeout"}[5m])`
- **Baseline**: < 2% timeout rate
- **Critical Threshold**: > 5% for 2 minutes
- **Warning Threshold**: > 3% for 5 minutes

### Memory Usage Baselines

#### Per-Agent Memory Usage
- **Metric**: `freeagentics_agent_memory_usage_bytes`
- **Baseline**: 20MB average per agent, 30MB peak
- **Critical Threshold**: > 34.5MB per agent
- **Warning Threshold**: > 30MB per agent

#### Total Agent Memory
- **Metric**: `sum(freeagentics_agent_memory_usage_bytes)`
- **Baseline**: 300MB average, 1GB peak
- **Critical Threshold**: > 1.5GB
- **Warning Threshold**: > 1.2GB

#### Memory Growth Rate
- **Metric**: `rate(freeagentics_agent_memory_usage_bytes[5m])`
- **Baseline**: Stable (< 1MB/min growth)
- **Critical Threshold**: > 5MB/min sustained growth
- **Warning Threshold**: > 2MB/min sustained growth

### API Performance Baselines

#### Response Time
- **Metric**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="freeagentics-backend"}[5m]))`
- **Baseline**: P95 < 300ms
- **Critical Threshold**: P95 > 500ms for 3 minutes
- **Warning Threshold**: P95 > 400ms for 5 minutes

#### Request Rate
- **Metric**: `rate(http_requests_total{job="freeagentics-backend"}[5m])`
- **Baseline**: 50 requests/second average, 200 requests/second peak
- **Critical Threshold**: > 500 requests/second
- **Warning Threshold**: > 300 requests/second

#### Error Rate
- **Metric**: `rate(http_requests_total{job="freeagentics-backend",status=~"5.."}[5m]) / rate(http_requests_total{job="freeagentics-backend"}[5m])`
- **Baseline**: < 1% error rate
- **Critical Threshold**: > 10% for 2 minutes
- **Warning Threshold**: > 5% for 5 minutes

### Belief System Baselines

#### Free Energy
- **Metric**: `freeagentics_belief_free_energy_current`
- **Baseline**: 0.5 - 5.0 (normal range)
- **Critical Threshold**: < 0.1 or > 10
- **Warning Threshold**: < 0.2 or > 8

#### Belief Accuracy
- **Metric**: `freeagentics_belief_accuracy_ratio`
- **Baseline**: > 80% accuracy
- **Critical Threshold**: < 70% for 10 minutes
- **Warning Threshold**: < 75% for 15 minutes

#### Convergence Time
- **Metric**: `histogram_quantile(0.90, rate(freeagentics_belief_convergence_time_seconds_bucket[5m]))`
- **Baseline**: P90 < 3 seconds
- **Critical Threshold**: P90 > 5 seconds for 5 minutes
- **Warning Threshold**: P90 > 4 seconds for 10 minutes

### Business Metrics Baselines

#### User Interactions
- **Metric**: `rate(freeagentics_business_user_interactions_total[1h])`
- **Baseline**: > 0.1 interactions/hour
- **Critical Threshold**: < 0.01 interactions/hour for 30 minutes
- **Warning Threshold**: < 0.05 interactions/hour for 60 minutes

#### Inference Operations
- **Metric**: `rate(freeagentics_business_inference_operations_total[5m])`
- **Baseline**: > 0.5 operations/second
- **Critical Threshold**: < 0.1 operations/second for 10 minutes
- **Warning Threshold**: < 0.3 operations/second for 15 minutes

#### Response Quality
- **Metric**: `freeagentics_business_response_quality_score`
- **Baseline**: > 75% quality score
- **Critical Threshold**: < 60% for 10 minutes
- **Warning Threshold**: < 70% for 15 minutes

## Service Level Indicators (SLIs)

### Availability SLI
- **Definition**: Percentage of time the system is available and responding
- **Measurement**: `avg_over_time(up{job="freeagentics-backend"}[24h])`
- **Good Events**: HTTP 200 responses
- **Total Events**: All HTTP responses

### Latency SLI
- **Definition**: Percentage of requests served within acceptable latency
- **Measurement**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- **Good Events**: Requests < 500ms
- **Total Events**: All requests

### Quality SLI
- **Definition**: Percentage of requests that are successful
- **Measurement**: `rate(http_requests_total{status!~"5.."}[5m]) / rate(http_requests_total[5m])`
- **Good Events**: HTTP 2xx, 3xx, 4xx responses
- **Total Events**: All HTTP responses

### Coordination SLI
- **Definition**: Percentage of agent coordination requests that succeed
- **Measurement**: `rate(freeagentics_agent_coordination_requests_total{status="success"}[5m]) / rate(freeagentics_agent_coordination_requests_total[5m])`
- **Good Events**: Successful coordination requests
- **Total Events**: All coordination requests

## Service Level Objectives (SLOs)

### Availability SLO
- **Target**: 99.9% uptime over 30 days
- **Error Budget**: 43.2 minutes of downtime per month
- **Measurement Window**: 30 days
- **Alerting**: Alert when availability drops below 99.5%

### Latency SLO
- **Target**: 95% of requests served within 500ms
- **Error Budget**: 5% of requests may exceed 500ms
- **Measurement Window**: 24 hours
- **Alerting**: Alert when P95 > 500ms for 3 minutes

### Quality SLO
- **Target**: 99% of requests are successful (non-5xx)
- **Error Budget**: 1% of requests may fail
- **Measurement Window**: 24 hours
- **Alerting**: Alert when error rate > 10% for 2 minutes

### Coordination SLO
- **Target**: 95% of coordination requests succeed
- **Error Budget**: 5% of coordination requests may fail
- **Measurement Window**: 1 hour
- **Alerting**: Alert when success rate < 90% for 5 minutes

### Memory SLO
- **Target**: Average agent memory usage < 30MB
- **Error Budget**: 10% of agents may exceed 30MB
- **Measurement Window**: 1 hour
- **Alerting**: Alert when any agent > 34.5MB

## Performance Test Scenarios

### Load Testing Scenarios

#### Scenario 1: Normal Load
- **Agents**: 15 active agents
- **Request Rate**: 50 requests/second
- **Duration**: 30 minutes
- **Expected Results**: All SLOs met

#### Scenario 2: Peak Load
- **Agents**: 30 active agents
- **Request Rate**: 200 requests/second
- **Duration**: 15 minutes
- **Expected Results**: All SLOs met with degraded performance

#### Scenario 3: Stress Test
- **Agents**: 45 active agents
- **Request Rate**: 400 requests/second
- **Duration**: 10 minutes
- **Expected Results**: Some SLO violations, system remains stable

#### Scenario 4: Coordination Limit Test
- **Agents**: 50+ active agents
- **Request Rate**: 100 requests/second
- **Duration**: 5 minutes
- **Expected Results**: Coordination alerts triggered, graceful degradation

### Regression Test Scenarios

#### API Performance Regression
- **Trigger**: P95 response time > 600ms for 2 minutes
- **Expected**: CI/CD pipeline fails
- **Action**: Block deployment until fixed

#### Memory Regression
- **Trigger**: Average agent memory > 35MB
- **Expected**: CI/CD pipeline fails
- **Action**: Block deployment until fixed

#### Coordination Regression
- **Trigger**: Coordination success rate < 85%
- **Expected**: CI/CD pipeline fails
- **Action**: Block deployment until fixed

## Monitoring and Alerting

### Alert Hierarchy

#### Critical Alerts (PagerDuty)
- System down
- Agent coordination > 50 limit
- Memory > 34.5MB per agent
- API P95 > 500ms

#### High Alerts (Slack + Email)
- Error rate > 10%
- Coordination success rate < 90%
- Memory > 30MB per agent
- API P95 > 400ms

#### Medium Alerts (Slack)
- Performance degradation trends
- Resource utilization warnings
- Business metric anomalies

### Baseline Monitoring

#### Daily Baseline Reports
- **Schedule**: Every day at 9:00 AM
- **Recipients**: Engineering, Product, Operations
- **Content**: SLO compliance, performance trends, anomalies

#### Weekly Baseline Review
- **Schedule**: Every Monday at 10:00 AM
- **Participants**: Engineering leads, SRE, Product
- **Content**: Baseline adjustments, SLO review, capacity planning

#### Monthly Baseline Analysis
- **Schedule**: First Monday of each month
- **Participants**: Engineering, Product, Leadership
- **Content**: Baseline evolution, SLO updates, capacity forecasting

## Capacity Planning

### Growth Projections

#### Agent Scaling
- **Current**: 15 agents average
- **3-month projection**: 25 agents average
- **6-month projection**: 40 agents average
- **12-month projection**: 60 agents average (requires architecture changes)

#### Memory Scaling
- **Current**: 300MB total agent memory
- **3-month projection**: 500MB total agent memory
- **6-month projection**: 1GB total agent memory
- **12-month projection**: 2GB total agent memory

#### Request Scaling
- **Current**: 50 requests/second
- **3-month projection**: 100 requests/second
- **6-month projection**: 200 requests/second
- **12-month projection**: 500 requests/second

### Capacity Thresholds

#### Scale-Up Triggers
- **Agent capacity**: > 80% of coordination limit (40 agents)
- **Memory usage**: > 80% of available memory
- **CPU usage**: > 70% sustained for 30 minutes
- **Disk usage**: > 80% of available disk

#### Scale-Down Triggers
- **Agent capacity**: < 30% of coordination limit (15 agents)
- **Memory usage**: < 40% of available memory
- **CPU usage**: < 30% sustained for 60 minutes

## Performance Regression Detection

### Automated Regression Detection

#### CI/CD Integration
- **Tool**: Jenkins/GitHub Actions
- **Trigger**: Every deployment
- **Tests**: Performance test suite
- **Failure Criteria**: Any baseline violation

#### Regression Test Suite
- **Location**: `tests/performance/`
- **Execution**: Post-deployment
- **Duration**: 15 minutes
- **Coverage**: All critical performance metrics

#### Regression Reporting
- **Format**: JSON report with baseline comparisons
- **Storage**: Performance history database
- **Alerting**: Slack notification on regression
- **Rollback**: Automatic rollback on critical regression

### Performance Trend Analysis

#### Weekly Trend Reports
- **Schedule**: Every Friday at 3:00 PM
- **Recipients**: Engineering team
- **Content**: Performance trends, regression analysis, recommendations

#### Monthly Performance Review
- **Schedule**: Last Friday of each month
- **Participants**: Engineering, Product, SRE
- **Content**: Performance evolution, baseline updates, capacity planning

## Troubleshooting

### Performance Degradation Playbook

#### Step 1: Identify Affected Metrics
- Check system overview dashboard
- Identify which baselines are violated
- Determine impact scope

#### Step 2: Investigate Root Cause
- Review recent deployments
- Check resource utilization
- Analyze error logs

#### Step 3: Immediate Actions
- Scale resources if needed
- Rollback recent changes if applicable
- Implement temporary fixes

#### Step 4: Long-term Resolution
- Address root cause
- Update baselines if needed
- Improve monitoring if gaps identified

### Common Performance Issues

#### High Memory Usage
- **Symptoms**: Agent memory > 30MB
- **Causes**: Memory leaks, inefficient algorithms, large datasets
- **Solutions**: Code optimization, garbage collection tuning, data structure improvements

#### Coordination Timeouts
- **Symptoms**: Timeout rate > 5%
- **Causes**: Network latency, resource contention, algorithm inefficiency
- **Solutions**: Timeout tuning, load balancing, algorithm optimization

#### API Latency
- **Symptoms**: P95 > 500ms
- **Causes**: Database queries, external API calls, CPU bottlenecks
- **Solutions**: Query optimization, caching, resource scaling

---

**Last Updated**: 2024-07-15  
**Version**: 1.0  
**Contact**: sre@freeagentics.com