# Monitoring and Alerting Runbook

## Overview

This runbook provides comprehensive procedures for monitoring and alerting operations for the FreeAgentics system, including metrics collection, alert management, dashboard configuration, and troubleshooting procedures.

## Table of Contents

1. [Monitoring Architecture](#monitoring-architecture)
2. [Metrics Collection](#metrics-collection)
3. [Alert Management](#alert-management)
4. [Dashboard Configuration](#dashboard-configuration)
5. [Log Management](#log-management)
6. [Performance Monitoring](#performance-monitoring)
7. [Capacity Planning](#capacity-planning)
8. [Troubleshooting](#troubleshooting)

## Monitoring Architecture

### 1. Monitoring Stack Components

#### Core Monitoring Components
- **Prometheus**: Metrics collection and time-series database
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and management
- **Node Exporter**: System metrics collection
- **Blackbox Exporter**: Endpoint monitoring
- **Custom Exporters**: Application-specific metrics

#### Observability Stack
```yaml
monitoring_stack:
  metrics:
    - prometheus
    - node_exporter
    - blackbox_exporter
    - custom_exporters
    
  visualization:
    - grafana
    - custom_dashboards
    
  logging:
    - elasticsearch
    - logstash
    - kibana
    - fluentd
    
  tracing:
    - jaeger
    - zipkin
    
  alerting:
    - alertmanager
    - pagerduty
    - slack
```

### 2. Monitoring Infrastructure

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
      
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
      
  - job_name: 'freeagentics-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'freeagentics-workers'
    static_configs:
      - targets: ['workers:8001']
    metrics_path: '/metrics'
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

#### Grafana Configuration
```yaml
# grafana.yml
server:
  http_port: 3000
  domain: monitoring.freeagentics.io

database:
  type: postgres
  host: postgres:5432
  name: grafana
  user: grafana
  password: ${GRAFANA_DB_PASSWORD}

security:
  admin_user: admin
  admin_password: ${GRAFANA_ADMIN_PASSWORD}

auth:
  oauth_auto_login: true
  disable_login_form: false

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

## Metrics Collection

### 1. System Metrics

#### Infrastructure Metrics
```bash
# Collect system metrics
./scripts/monitoring/collect-system-metrics.sh

# CPU metrics
node_cpu_seconds_total{mode="idle"}
node_cpu_seconds_total{mode="system"}
node_cpu_seconds_total{mode="user"}

# Memory metrics
node_memory_MemTotal_bytes
node_memory_MemFree_bytes
node_memory_MemAvailable_bytes
node_memory_Buffers_bytes
node_memory_Cached_bytes

# Disk metrics
node_disk_read_bytes_total
node_disk_written_bytes_total
node_filesystem_size_bytes
node_filesystem_free_bytes

# Network metrics
node_network_receive_bytes_total
node_network_transmit_bytes_total
```

#### Container Metrics
```bash
# Docker container metrics
./scripts/monitoring/collect-container-metrics.sh

# Container CPU usage
container_cpu_usage_seconds_total
container_cpu_system_seconds_total
container_cpu_user_seconds_total

# Container memory usage
container_memory_usage_bytes
container_memory_max_usage_bytes
container_memory_cache

# Container network I/O
container_network_receive_bytes_total
container_network_transmit_bytes_total
```

### 2. Application Metrics

#### API Metrics
```python
# Custom API metrics collection
from prometheus_client import Counter, Histogram, Gauge

# Request counters
REQUEST_COUNT = Counter(
    'freeagentics_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Response time histogram
REQUEST_DURATION = Histogram(
    'freeagentics_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Active connections gauge
ACTIVE_CONNECTIONS = Gauge(
    'freeagentics_active_connections',
    'Number of active connections'
)

# Agent coordination metrics
AGENT_COORDINATION_SUCCESS = Counter(
    'freeagentics_agent_coordination_success_total',
    'Successful agent coordination attempts'
)

AGENT_COORDINATION_FAILURES = Counter(
    'freeagentics_agent_coordination_failures_total',
    'Failed agent coordination attempts'
)
```

#### Database Metrics
```bash
# PostgreSQL metrics
./scripts/monitoring/collect-db-metrics.sh

# Database connections
pg_stat_database_numbackends
pg_stat_database_xact_commit
pg_stat_database_xact_rollback

# Query performance
pg_stat_statements_total_time
pg_stat_statements_calls
pg_stat_statements_mean_time

# Table statistics
pg_stat_user_tables_n_tup_ins
pg_stat_user_tables_n_tup_upd
pg_stat_user_tables_n_tup_del
```

### 3. Business Metrics

#### User Metrics
```bash
# User activity metrics
./scripts/monitoring/collect-user-metrics.sh

# Active users
freeagentics_active_users_total
freeagentics_new_users_total
freeagentics_user_sessions_total

# User engagement
freeagentics_user_actions_total
freeagentics_feature_usage_total
freeagentics_session_duration_seconds
```

#### Performance Metrics
```bash
# Performance metrics
./scripts/monitoring/collect-performance-metrics.sh

# Response times
freeagentics_api_response_time_seconds
freeagentics_database_query_time_seconds
freeagentics_agent_coordination_time_seconds

# Throughput
freeagentics_requests_per_second
freeagentics_transactions_per_second
freeagentics_agent_actions_per_second
```

## Alert Management

### 1. Alert Configuration

#### Alert Rules
```yaml
# /etc/prometheus/rules/freeagentics.yml
groups:
  - name: freeagentics.rules
    rules:
      - alert: HighCPUUsage
        expr: (100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes"

      - alert: DiskSpaceLow
        expr: (node_filesystem_free_bytes / node_filesystem_size_bytes) * 100 < 15
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Disk space is below 15%"

      - alert: APIHighErrorRate
        expr: rate(freeagentics_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate"
          description: "API error rate is above 10%"

      - alert: AgentCoordinationFailures
        expr: rate(freeagentics_agent_coordination_failures_total[5m]) > 0.05
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Agent coordination failures"
          description: "Agent coordination failure rate is above 5%"
```

#### Alertmanager Configuration
```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@freeagentics.io'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
    - match:
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://webhook:5000/alert'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        description: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'slack'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts'
        title: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

### 2. Alert Response Procedures

#### Alert Triage
```bash
# Alert triage procedures
./scripts/monitoring/alert-triage.sh --alert-id ${ALERT_ID}

# Assess alert severity
./scripts/monitoring/assess-severity.sh --alert-id ${ALERT_ID}

# Assign alert to team member
./scripts/monitoring/assign-alert.sh --alert-id ${ALERT_ID} --assignee ${ASSIGNEE}

# Escalate alert if needed
./scripts/monitoring/escalate-alert.sh --alert-id ${ALERT_ID} --level ${LEVEL}
```

#### Alert Investigation
```bash
# Investigate alert
./scripts/monitoring/investigate-alert.sh --alert-id ${ALERT_ID}

# Gather relevant metrics
./scripts/monitoring/gather-metrics.sh --alert-id ${ALERT_ID} --timeframe 1h

# Check related systems
./scripts/monitoring/check-related-systems.sh --alert-id ${ALERT_ID}

# Generate investigation report
./scripts/monitoring/generate-investigation-report.sh --alert-id ${ALERT_ID}
```

### 3. Alert Automation

#### Automated Alert Responses
```bash
# Configure automated responses
./scripts/monitoring/configure-auto-response.sh

# Auto-scaling response
./scripts/monitoring/auto-scale-response.sh --trigger cpu_high --action scale_up

# Circuit breaker response
./scripts/monitoring/circuit-breaker-response.sh --trigger error_rate_high --action enable_circuit_breaker

# Notification automation
./scripts/monitoring/notification-automation.sh --trigger alert_created --action notify_team
```

## Dashboard Configuration

### 1. System Overview Dashboard

#### Main Dashboard Configuration
```json
{
  "dashboard": {
    "title": "FreeAgentics System Overview",
    "panels": [
      {
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"freeagentics-api\"}",
            "legendFormat": "API Service"
          },
          {
            "expr": "up{job=\"node\"}",
            "legendFormat": "System"
          }
        ]
      },
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "Memory Usage %"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(freeagentics_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### 2. Application Performance Dashboard

#### API Performance Metrics
```bash
# Create API performance dashboard
./scripts/monitoring/create-api-dashboard.sh

# API response times
histogram_quantile(0.50, rate(freeagentics_request_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(freeagentics_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(freeagentics_request_duration_seconds_bucket[5m]))

# API request rates
rate(freeagentics_requests_total[5m])

# API error rates
rate(freeagentics_requests_total{status=~"5.."}[5m]) / rate(freeagentics_requests_total[5m])
```

### 3. Custom Business Dashboards

#### Agent Coordination Dashboard
```bash
# Create agent coordination dashboard
./scripts/monitoring/create-agent-dashboard.sh

# Agent coordination success rate
rate(freeagentics_agent_coordination_success_total[5m]) / 
(rate(freeagentics_agent_coordination_success_total[5m]) + 
 rate(freeagentics_agent_coordination_failures_total[5m]))

# Agent coordination latency
histogram_quantile(0.95, rate(freeagentics_agent_coordination_time_seconds_bucket[5m]))

# Active agents
freeagentics_active_agents_total
```

## Log Management

### 1. Log Collection

#### Centralized Logging Setup
```bash
# Configure centralized logging
./scripts/monitoring/setup-logging.sh

# Fluentd configuration
./scripts/monitoring/configure-fluentd.sh

# Elasticsearch setup
./scripts/monitoring/setup-elasticsearch.sh

# Kibana configuration
./scripts/monitoring/configure-kibana.sh
```

#### Log Forwarding Configuration
```yaml
# fluentd.conf
<source>
  @type tail
  path /var/log/freeagentics/*.log
  pos_file /var/log/fluentd/freeagentics.log.pos
  tag freeagentics.*
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%L%z
</source>

<match freeagentics.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name freeagentics
  type_name _doc
  logstash_format true
  logstash_prefix freeagentics
</match>
```

### 2. Log Analysis

#### Log Search and Analysis
```bash
# Search logs
./scripts/monitoring/search-logs.sh --query "error" --timeframe 1h

# Analyze error patterns
./scripts/monitoring/analyze-errors.sh --period 24h

# Generate log report
./scripts/monitoring/generate-log-report.sh --period 24h

# Create log alerts
./scripts/monitoring/create-log-alerts.sh
```

#### Log Retention and Archival
```bash
# Configure log retention
./scripts/monitoring/configure-retention.sh --days 30

# Archive old logs
./scripts/monitoring/archive-logs.sh --older-than 30d

# Cleanup archived logs
./scripts/monitoring/cleanup-logs.sh --older-than 1y
```

## Performance Monitoring

### 1. Application Performance Monitoring (APM)

#### APM Setup
```bash
# Configure APM
./scripts/monitoring/setup-apm.sh

# Install APM agents
./scripts/monitoring/install-apm-agents.sh

# Configure distributed tracing
./scripts/monitoring/configure-tracing.sh
```

#### Performance Metrics
```bash
# Collect performance metrics
./scripts/monitoring/collect-performance-metrics.sh

# Application response times
./scripts/monitoring/measure-response-times.sh

# Database query performance
./scripts/monitoring/analyze-db-performance.sh

# Memory usage patterns
./scripts/monitoring/analyze-memory-usage.sh
```

### 2. Real User Monitoring (RUM)

#### RUM Implementation
```javascript
// Client-side RUM implementation
import { init } from '@freeagentics/rum-sdk';

init({
  apiKey: 'your-api-key',
  serviceName: 'freeagentics-web',
  environment: 'production'
});

// Track page loads
window.addEventListener('load', () => {
  const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
  rum.track('page_load', { duration: loadTime });
});

// Track user interactions
document.addEventListener('click', (event) => {
  rum.track('user_interaction', { target: event.target.tagName });
});
```

### 3. Synthetic Monitoring

#### Synthetic Tests
```bash
# Configure synthetic monitoring
./scripts/monitoring/setup-synthetic-monitoring.sh

# Create API health checks
./scripts/monitoring/create-api-health-checks.sh

# Setup user journey tests
./scripts/monitoring/setup-user-journey-tests.sh

# Configure availability monitoring
./scripts/monitoring/configure-availability-monitoring.sh
```

## Capacity Planning

### 1. Resource Utilization Analysis

#### Resource Trends
```bash
# Analyze resource trends
./scripts/monitoring/analyze-resource-trends.sh --period 30d

# Generate capacity report
./scripts/monitoring/generate-capacity-report.sh

# Predict resource needs
./scripts/monitoring/predict-capacity-needs.sh --horizon 3m

# Recommend scaling actions
./scripts/monitoring/recommend-scaling.sh
```

### 2. Performance Baselines

#### Baseline Management
```bash
# Create performance baselines
./scripts/monitoring/create-baselines.sh

# Update baselines
./scripts/monitoring/update-baselines.sh --period 7d

# Compare against baselines
./scripts/monitoring/compare-baselines.sh

# Alert on baseline deviations
./scripts/monitoring/alert-baseline-deviations.sh
```

### 3. Capacity Forecasting

#### Forecasting Models
```python
# Capacity forecasting
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

def forecast_capacity(metric_data, horizon_days=30):
    """
    Forecast capacity requirements based on historical data
    """
    # Prepare data
    df = pd.DataFrame(metric_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['days_from_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
    
    # Train model
    model = LinearRegression()
    model.fit(df[['days_from_start']], df['value'])
    
    # Forecast
    future_days = range(df['days_from_start'].max() + 1, 
                       df['days_from_start'].max() + horizon_days + 1)
    forecast = model.predict([[day] for day in future_days])
    
    return forecast
```

## Troubleshooting

### 1. Common Monitoring Issues

#### Prometheus Issues
```bash
# Check Prometheus status
./scripts/monitoring/check-prometheus.sh

# Restart Prometheus
./scripts/monitoring/restart-prometheus.sh

# Verify scrape targets
./scripts/monitoring/verify-scrape-targets.sh

# Check Prometheus configuration
./scripts/monitoring/validate-prometheus-config.sh
```

#### Grafana Issues
```bash
# Check Grafana status
./scripts/monitoring/check-grafana.sh

# Restart Grafana
./scripts/monitoring/restart-grafana.sh

# Verify datasource connections
./scripts/monitoring/verify-datasources.sh

# Import/export dashboards
./scripts/monitoring/manage-dashboards.sh
```

### 2. Alert Troubleshooting

#### Alert Debugging
```bash
# Debug alert rules
./scripts/monitoring/debug-alerts.sh --alert-name ${ALERT_NAME}

# Test alert conditions
./scripts/monitoring/test-alert-conditions.sh --alert-name ${ALERT_NAME}

# Verify alert routing
./scripts/monitoring/verify-alert-routing.sh --alert-name ${ALERT_NAME}

# Check alert history
./scripts/monitoring/check-alert-history.sh --alert-name ${ALERT_NAME}
```

### 3. Performance Troubleshooting

#### Performance Issues
```bash
# Identify performance bottlenecks
./scripts/monitoring/identify-bottlenecks.sh

# Analyze slow queries
./scripts/monitoring/analyze-slow-queries.sh

# Check resource constraints
./scripts/monitoring/check-resource-constraints.sh

# Generate performance report
./scripts/monitoring/generate-performance-report.sh
```

## Monitoring Best Practices

### 1. Metrics Best Practices

#### Metric Naming
- Use consistent naming conventions
- Include units in metric names
- Group related metrics with prefixes
- Use labels for dimensions

#### Metric Collection
- Collect metrics at appropriate intervals
- Avoid high-cardinality metrics
- Use histograms for latency metrics
- Monitor metric collection performance

### 2. Alerting Best Practices

#### Alert Design
- Alert on symptoms, not causes
- Avoid alert fatigue
- Set appropriate thresholds
- Use multiple conditions for complex alerts

#### Alert Management
- Implement proper alert escalation
- Document alert response procedures
- Regular review and tuning of alerts
- Test alert systems regularly

### 3. Dashboard Best Practices

#### Dashboard Design
- Focus on key metrics
- Use appropriate visualization types
- Implement drill-down capabilities
- Keep dashboards simple and focused

#### Dashboard Management
- Regular dashboard reviews
- Version control for dashboards
- Document dashboard purpose
- Share dashboards with relevant teams

## Monitoring Checklist

### Daily Monitoring Tasks
- [ ] Check system health dashboards
- [ ] Review overnight alerts
- [ ] Verify monitoring system status
- [ ] Check metric collection gaps
- [ ] Review performance trends

### Weekly Monitoring Tasks
- [ ] Review alert effectiveness
- [ ] Update performance baselines
- [ ] Check capacity trends
- [ ] Review dashboard usage
- [ ] Update monitoring documentation

### Monthly Monitoring Tasks
- [ ] Conduct monitoring system audit
- [ ] Review and tune alert thresholds
- [ ] Update capacity forecasts
- [ ] Review monitoring costs
- [ ] Train team on new tools

### Quarterly Monitoring Tasks
- [ ] Comprehensive monitoring review
- [ ] Evaluate new monitoring tools
- [ ] Update monitoring strategy
- [ ] Review monitoring SLAs
- [ ] Conduct monitoring disaster recovery test

---

**Document Information:**
- **Version**: 1.0
- **Last Updated**: January 2024
- **Next Review**: April 2024
- **Owner**: Monitoring Team
- **Approved By**: Operations Manager

**Monitoring Objectives:**
- **Availability**: 99.9% uptime
- **Performance**: < 500ms API response time
- **Alerting**: < 5 minute detection time