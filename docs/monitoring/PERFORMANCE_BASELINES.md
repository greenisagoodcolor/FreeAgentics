# FreeAgentics Performance Baselines Documentation

## Overview

This document defines performance baselines, thresholds, and capacity limits for the FreeAgentics multi-agent system. These baselines are used for monitoring, alerting, capacity planning, and performance regression detection.

## Table of Contents

1. [Baseline Methodology](#baseline-methodology)
1. [System Performance Baselines](#system-performance-baselines)
1. [Agent Coordination Baselines](#agent-coordination-baselines)
1. [Memory Usage Baselines](#memory-usage-baselines)
1. [API Performance Baselines](#api-performance-baselines)
1. [Belief System Baselines](#belief-system-baselines)
1. [Business Metrics Baselines](#business-metrics-baselines)
1. [Capacity Planning Thresholds](#capacity-planning-thresholds)
1. [Performance Testing Scenarios](#performance-testing-scenarios)
1. [Baseline Maintenance](#baseline-maintenance)

## Baseline Methodology

### Data Collection

Baselines are established through:

- 30-day rolling average of production metrics
- Load testing under controlled conditions
- Statistical analysis (P50, P90, P95, P99)
- Peak vs. off-peak analysis

### Update Frequency

- **Weekly Review**: Trend analysis and minor adjustments
- **Monthly Update**: Comprehensive baseline recalculation
- **Quarterly Review**: Capacity planning and architecture review

### Statistical Methods

```python
# Baseline calculation example
import numpy as np

def calculate_baseline(data, method='percentile'):
    if method == 'percentile':
        return {
            'p50': np.percentile(data, 50),
            'p90': np.percentile(data, 90),
            'p95': np.percentile(data, 95),
            'p99': np.percentile(data, 99)
        }
    elif method == 'stddev':
        mean = np.mean(data)
        std = np.std(data)
        return {
            'normal': mean,
            'warning': mean + 2 * std,
            'critical': mean + 3 * std
        }
```

## System Performance Baselines

### System Availability

| Metric | Baseline | Warning | Critical | SLO Target |
|--------|----------|---------|----------|------------|
| Uptime | 99.95% | < 99.9% | < 99.5% | 99.9% |
| Health Check Success | 99.99% | < 99.9% | < 99.0% | 99.9% |
| Service Ready Time | < 30s | > 45s | > 60s | < 45s |

**Measurement Query**:

```promql
# Availability over 24h
avg_over_time(up{job="freeagentics-backend"}[24h]) * 100
```

### System Resource Usage

| Resource | Normal (P50) | Elevated (P90) | Warning (P95) | Critical |
|----------|--------------|----------------|---------------|----------|
| CPU Usage | 40% | 60% | 70% | 90% |
| Memory Usage | 1.5 GB | 1.8 GB | 2.0 GB | 2.5 GB |
| Disk I/O | 50 MB/s | 100 MB/s | 150 MB/s | 200 MB/s |
| Network I/O | 10 MB/s | 25 MB/s | 40 MB/s | 50 MB/s |

**Key Metrics**:

```promql
# CPU Usage
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory Usage
freeagentics_system_memory_usage_bytes / (1024*1024*1024)
```

### System Throughput

| Metric | Baseline | Peak Load | Maximum | Degradation Point |
|--------|----------|-----------|---------|-------------------|
| Requests/sec | 50 | 200 | 500 | 400 |
| Concurrent Users | 100 | 500 | 1000 | 800 |
| Active Connections | 200 | 800 | 2000 | 1500 |

## Agent Coordination Baselines

### Active Agent Limits

| Metric | Normal | Warning | Critical | Hard Limit |
|--------|--------|---------|----------|------------|
| Active Agents | 15 | 40 | 45 | 50 |
| Agents per Type | 5 | 15 | 18 | 20 |
| Coordination Queue | 10 | 50 | 80 | 100 |

**Monitoring Query**:

```promql
# Active agents by type
sum by (agent_type) (freeagentics_agent_active{state="running"})
```

### Coordination Performance

| Operation | P50 | P90 | P95 | P99 | Timeout |
|-----------|-----|-----|-----|-----|---------|
| Coalition Formation | 500ms | 1.5s | 1.8s | 2.0s | 5s |
| Agent Communication | 50ms | 100ms | 150ms | 200ms | 1s |
| State Synchronization | 100ms | 300ms | 500ms | 800ms | 2s |
| Belief Convergence | 1s | 3s | 4s | 5s | 10s |

**Performance Tracking**:

```promql
# Coordination duration percentiles
histogram_quantile(0.95, 
  rate(freeagentics_agent_coordination_duration_seconds_bucket[5m])
)
```

### Coordination Success Rates

| Metric | Baseline | Acceptable | Warning | Critical |
|--------|----------|------------|---------|----------|
| Success Rate | 98% | 95% | 90% | 85% |
| Timeout Rate | 1% | 2% | 3% | 5% |
| Retry Rate | 2% | 5% | 8% | 10% |
| Failure Rate | 1% | 3% | 5% | 10% |

## Memory Usage Baselines

### Per-Agent Memory

| Agent Type | Normal (P50) | Expected (P90) | Warning | Critical | Kill Threshold |
|------------|--------------|----------------|---------|----------|----------------|
| Coordinator | 20 MB | 28 MB | 30 MB | 34 MB | 35 MB |
| Worker | 15 MB | 22 MB | 25 MB | 30 MB | 32 MB |
| Monitor | 10 MB | 15 MB | 18 MB | 20 MB | 22 MB |
| Analyzer | 25 MB | 32 MB | 35 MB | 38 MB | 40 MB |

**Memory Monitoring**:

```promql
# Top memory consumers
topk(10, freeagentics_agent_memory_usage_bytes / (1024*1024))
  by (agent_id, agent_type)
```

### Memory Growth Patterns

| Timeframe | Normal Growth | Acceptable | Warning | Critical |
|-----------|---------------|------------|---------|----------|
| Per Hour | 0.5 MB | 1 MB | 2 MB | 5 MB |
| Per Day | 10 MB | 20 MB | 50 MB | 100 MB |
| Per Week | 50 MB | 100 MB | 200 MB | 500 MB |

**Growth Detection**:

```promql
# Memory growth rate over 1 hour
rate(freeagentics_agent_memory_usage_bytes[1h]) * 3600 / (1024*1024)
```

### Total System Memory

| Component | Baseline | Peak | Warning | Critical |
|-----------|----------|------|---------|----------|
| All Agents | 300 MB | 500 MB | 1 GB | 1.5 GB |
| API Server | 500 MB | 800 MB | 1 GB | 1.2 GB |
| Database Connections | 200 MB | 400 MB | 600 MB | 800 MB |
| Cache | 1 GB | 1.5 GB | 2 GB | 2.5 GB |

## API Performance Baselines

### Response Time Percentiles

| Endpoint | P50 | P90 | P95 | P99 | SLO Target |
|----------|-----|-----|-----|-----|------------|
| GET /health | 5ms | 10ms | 15ms | 25ms | < 50ms |
| GET /api/v1/agents | 50ms | 150ms | 200ms | 300ms | < 500ms |
| POST /api/v1/agents | 100ms | 300ms | 400ms | 500ms | < 1s |
| GET /api/v1/beliefs | 80ms | 250ms | 350ms | 450ms | < 500ms |
| POST /api/v1/coordinate | 200ms | 500ms | 700ms | 900ms | < 1s |

**Latency Monitoring**:

```promql
# P95 latency by endpoint
histogram_quantile(0.95,
  sum by (handler, le) (
    rate(http_request_duration_seconds_bucket[5m])
  )
) * 1000
```

### Throughput Metrics

| Endpoint | Normal RPS | Peak RPS | Max RPS | Degradation |
|----------|------------|----------|---------|-------------|
| /health | 100 | 500 | 1000 | 800 |
| /api/v1/agents | 20 | 100 | 200 | 150 |
| /api/v1/coordinate | 10 | 50 | 100 | 80 |
| /api/v1/knowledge | 30 | 150 | 300 | 250 |

### Error Rates

| Error Type | Baseline | Acceptable | Warning | Critical |
|------------|----------|------------|---------|----------|
| 4xx Errors | 2% | 5% | 8% | 10% |
| 5xx Errors | 0.1% | 1% | 5% | 10% |
| Timeouts | 0.5% | 2% | 5% | 10% |
| Total Errors | 2.5% | 5% | 10% | 15% |

## Belief System Baselines

### Free Energy Metrics

| State | Minimum | Normal Range | Maximum | Anomaly Threshold |
|-------|---------|--------------|---------|-------------------|
| Stable | 0.5 | 1.0 - 3.0 | 5.0 | > 10.0 |
| Learning | 2.0 | 3.0 - 6.0 | 8.0 | > 15.0 |
| Adapting | 3.0 | 5.0 - 8.0 | 10.0 | > 20.0 |

**Free Energy Monitoring**:

```promql
# Free energy distribution
histogram_quantile(0.95,
  rate(freeagentics_belief_free_energy_bucket[5m])
)
```

### Belief Accuracy

| Metric | Baseline | Acceptable | Warning | Critical |
|--------|----------|------------|---------|----------|
| Prediction Accuracy | 85% | 80% | 75% | 70% |
| Convergence Rate | 95% | 90% | 85% | 80% |
| Update Success | 98% | 95% | 90% | 85% |

### Belief Processing Performance

| Operation | P50 | P90 | P95 | P99 |
|-----------|-----|-----|-----|-----|
| Belief Update | 10ms | 50ms | 100ms | 200ms |
| Prediction Generation | 20ms | 100ms | 200ms | 500ms |
| State Convergence | 500ms | 2s | 3s | 5s |

## Business Metrics Baselines

### User Engagement

| Metric | Baseline | Target | Warning | Critical |
|--------|----------|--------|---------|----------|
| Active Users/Hour | 100 | 150 | < 50 | < 10 |
| Interactions/User | 5 | 8 | < 3 | < 1 |
| Session Duration | 10 min | 15 min | < 5 min | < 2 min |
| Return Rate | 60% | 70% | < 40% | < 20% |

### Inference Operations

| Metric | Normal | Peak | Warning | Critical |
|--------|--------|------|---------|----------|
| Inferences/Second | 0.5 | 2.0 | < 0.1 | < 0.01 |
| Inference Quality | 80% | 85% | < 70% | < 60% |
| Cache Hit Rate | 70% | 80% | < 50% | < 30% |

### Knowledge Graph Growth

| Metric | Normal/Day | Peak/Day | Warning | Critical |
|--------|------------|----------|---------|----------|
| New Nodes | 1,000 | 5,000 | > 10,000 | > 20,000 |
| New Edges | 5,000 | 20,000 | > 50,000 | > 100,000 |
| Graph Size | +100 MB | +500 MB | > 1 GB | > 2 GB |

## Capacity Planning Thresholds

### Scale-Up Triggers

| Resource | Utilization | Duration | Action |
|----------|-------------|----------|--------|
| CPU | > 70% | 10 min | Add compute nodes |
| Memory | > 80% | 5 min | Increase memory |
| Agents | > 80% limit | 2 min | Scale coordinator |
| Queue | > 100 items | 5 min | Add workers |

### Scale-Down Triggers

| Resource | Utilization | Duration | Action |
|----------|-------------|----------|--------|
| CPU | < 30% | 30 min | Remove nodes |
| Memory | < 40% | 60 min | Reduce memory |
| Agents | < 30% limit | 30 min | Reduce coordinators |
| Queue | < 10 items | 30 min | Remove workers |

### Growth Projections

| Timeframe | Agents | Memory | Storage | Compute |
|-----------|--------|--------|---------|---------|
| Current | 15 | 2 GB | 50 GB | 4 cores |
| 3 Months | 25 | 4 GB | 100 GB | 8 cores |
| 6 Months | 40 | 8 GB | 200 GB | 16 cores |
| 12 Months | 60 | 16 GB | 500 GB | 32 cores |

## Performance Testing Scenarios

### Load Test Profiles

#### 1. Normal Load Test

```yaml
scenario: normal_load
duration: 30m
configuration:
  agents: 15
  rps: 50
  users: 100
expected_results:
  error_rate: < 1%
  p95_latency: < 300ms
  success_rate: > 99%
```

#### 2. Peak Load Test

```yaml
scenario: peak_load
duration: 15m
configuration:
  agents: 30
  rps: 200
  users: 500
expected_results:
  error_rate: < 5%
  p95_latency: < 500ms
  success_rate: > 95%
```

#### 3. Stress Test

```yaml
scenario: stress_test
duration: 10m
configuration:
  agents: 45
  rps: 400
  users: 1000
expected_results:
  error_rate: < 10%
  p95_latency: < 1s
  no_crashes: true
```

#### 4. Endurance Test

```yaml
scenario: endurance_test
duration: 24h
configuration:
  agents: 20
  rps: 100
  users: 200
expected_results:
  memory_growth: < 100MB
  error_rate: < 1%
  stable_performance: true
```

### Performance Test Execution

```bash
#!/bin/bash
# run-performance-test.sh

SCENARIO=$1
RESULTS_DIR="./performance-results/$(date +%Y%m%d_%H%M%S)"

# Run load test
k6 run \
  --out json=${RESULTS_DIR}/results.json \
  --out influxdb=http://influxdb:8086/k6 \
  scenarios/${SCENARIO}.js

# Analyze results
python analyze_performance.py \
  --input ${RESULTS_DIR}/results.json \
  --baseline baseline.yaml \
  --output ${RESULTS_DIR}/analysis.html

# Check for regressions
if [ $? -ne 0 ]; then
  echo "Performance regression detected!"
  exit 1
fi
```

## Baseline Maintenance

### Review Process

#### Weekly Review Checklist

- [ ] Compare current metrics to baselines
- [ ] Identify trending deviations
- [ ] Review alert frequency
- [ ] Update minor thresholds

#### Monthly Update Process

1. Export 30-day metrics
1. Calculate new percentiles
1. Compare to existing baselines
1. Update configuration
1. Test alert thresholds
1. Document changes

#### Quarterly Capacity Review

1. Analyze growth trends
1. Project future capacity
1. Review architecture limits
1. Plan infrastructure changes
1. Update SLOs if needed

### Baseline Configuration Management

```yaml
# baseline-config.yaml
version: "1.2"
updated: "2025-01-15"
author: "sre-team"

baselines:
  system:
    cpu:
      p50: 40
      p90: 60
      p95: 70
      critical: 90
    memory:
      p50: 1500  # MB
      p90: 1800
      p95: 2000
      critical: 2500
  
  agents:
    active_count:
      normal: 15
      warning: 40
      critical: 50
    memory_per_agent:
      p50: 20  # MB
      p90: 28
      p95: 30
      critical: 34.5
  
  api:
    latency:
      p50: 100  # ms
      p90: 300
      p95: 500
      p99: 800
    error_rate:
      normal: 0.001  # 0.1%
      warning: 0.05  # 5%
      critical: 0.10  # 10%
```

### Automation Scripts

```python
# update_baselines.py
import pandas as pd
from prometheus_api_client import PrometheusConnect
import yaml

def update_baselines():
    prom = PrometheusConnect(url="http://prometheus:9090")
    
    # Fetch 30-day metrics
    metrics = {
        'cpu': 'avg(rate(node_cpu_seconds_total[5m]))',
        'memory': 'freeagentics_system_memory_usage_bytes',
        'latency': 'http_request_duration_seconds'
    }
    
    baselines = {}
    for name, query in metrics.items():
        data = prom.custom_query_range(
            query=query,
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now(),
            step='5m'
        )
        
        values = [float(v[1]) for v in data[0]['values']]
        baselines[name] = {
            'p50': np.percentile(values, 50),
            'p90': np.percentile(values, 90),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
    
    # Save updated baselines
    with open('baseline-config.yaml', 'w') as f:
        yaml.dump(baselines, f)
    
    return baselines
```

### Change Tracking

All baseline changes must be:

1. Documented in change log
1. Reviewed by SRE team
1. Tested in staging
1. Announced to engineering
1. Monitored for impact

```markdown
# Baseline Change Log

## 2025-01-15
- Updated CPU baseline from 35% to 40% (normal operations)
- Increased agent memory warning from 28MB to 30MB
- Adjusted API latency P95 from 450ms to 500ms
- Reason: System growth and optimization improvements

## 2024-12-15
- Initial baseline establishment
- Based on 30-day production data
- Validated through load testing
```

______________________________________________________________________

**Last Updated**: 2025-01-15\
**Version**: 1.2\
**Next Review**: 2025-02-15\
**Contact**: sre@freeagentics.com
