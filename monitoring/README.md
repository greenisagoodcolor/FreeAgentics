# FreeAgentics Production Monitoring & Alerting

This directory contains comprehensive monitoring, alerting, and incident response configurations for FreeAgentics production deployment.

## Overview

The monitoring system provides:
- **Real-time health monitoring** with custom dashboards
- **Proactive alerting** with intelligent thresholds
- **Automated incident response** with predefined playbooks
- **SLO/SLA tracking** with error budget monitoring
- **Security monitoring** for threat detection

## Components

### 1. Configuration Files

#### `/config/monitoring_config.yaml`
Main monitoring configuration including:
- Alert thresholds for all system components
- Alert routing rules
- Automated response configurations
- Health check endpoints
- Dashboard configurations
- Retention policies

#### `/config/incident_response.yaml`
Automated incident response playbooks for:
- High memory usage
- API performance degradation
- Database connection exhaustion
- DDoS attacks
- Agent coordination failures

#### `/config/sli_slo_config.yaml`
Service Level Indicators and Objectives:
- Availability targets (99.9%)
- Latency targets (P95 < 500ms)
- Error budget policies
- Performance baselines

### 2. Dashboards

#### System Overview Dashboard
- System uptime and availability
- Error rates and request latency
- Active agents and resource usage
- Database and cache performance

#### Agent Performance Dashboard
- Belief state evolution and accuracy
- Inference time distribution
- Memory and CPU usage per agent
- Coordination performance metrics

### 3. Alert Rules

#### `/rules/comprehensive_alerts.yml`
Prometheus alert rules covering:
- **Performance**: Response time, throughput
- **Agents**: Memory, inference time, belief accuracy
- **Resources**: Database, Redis, disk, network
- **Security**: Authentication, DDoS, SSL certificates
- **Business**: User activity, task completion
- **SLOs**: Availability, latency, error budgets

### 4. Health Check Endpoints

#### Basic Health Checks
- `/health` - Simple availability check
- `/health/ready` - Kubernetes readiness probe
- `/health/live` - Kubernetes liveness probe
- `/health/startup` - Kubernetes startup probe

#### Extended Health Checks
- `/health/detailed` - Comprehensive system diagnostics
- `/health/dependencies` - External dependency health
- Database connectivity and performance
- Redis cache health and hit rates
- Agent system status
- System resource utilization

### 5. Monitoring Scripts

#### `health_monitor.py`
Continuous health monitoring script that:
- Performs regular health checks
- Monitors synthetic user journeys
- Triggers alerts on threshold violations
- Sends notifications to Slack/PagerDuty

#### `incident_responder.py`
Automated incident response system that:
- Listens for alerts via Redis pub/sub
- Executes predefined playbooks
- Performs remediation actions
- Sends notifications to teams

## Setup Instructions

### 1. Environment Variables

Set the following environment variables:
```bash
export GRAFANA_API_KEY="your-grafana-api-key"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export PAGERDUTY_INTEGRATION_KEY="your-pagerduty-key"
export PAGERDUTY_ROUTING_KEY="your-routing-key"
```

### 2. Deploy Dashboards

Deploy Grafana dashboards:
```bash
./scripts/deploy_dashboards.sh
```

### 3. Configure Prometheus

Update `prometheus.yml` to include FreeAgentics job:
```yaml
scrape_configs:
  - job_name: "freeagentics-app"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: "/metrics"
    scrape_interval: 5s
```

### 4. Start Monitoring

Start the health monitoring service:
```bash
python monitoring/scripts/health_monitor.py
```

Start the incident responder:
```bash
python monitoring/scripts/incident_responder.py
```

## Alert Thresholds

### API Performance
- **P50 Response Time**: Warning at 100ms, Critical at 200ms
- **P95 Response Time**: Warning at 500ms, Critical at 1s
- **Error Rate**: Warning at 1%, Critical at 5%

### Agent Performance
- **Inference Time**: Warning at 100ms, Critical at 200ms
- **Memory per Agent**: Warning at 25MB, Critical at 30MB
- **Belief Accuracy**: Warning at 75%, Critical at 60%
- **Free Energy**: Warning at 5.0, Critical at 8.0

### System Resources
- **Database Connections**: Warning at 150, Critical at 180
- **Redis Memory**: Warning at 80%, Critical at 90%
- **Disk Usage**: Warning at 70%, Critical at 85%

### Security
- **Failed Auth**: Warning at 10/min, Critical at 50/min
- **Rate Limiting**: Warning at 100/min, Critical at 500/min
- **SSL Expiry**: Warning at 30 days, Critical at 7 days

## Automated Responses

### Memory Pressure
1. Collect memory profile diagnostics
2. Trigger aggressive garbage collection
3. Kill agents exceeding 35MB
4. Clear non-essential caches
5. Scale out agent pool if needed

### API Degradation
1. Enable circuit breaker for slow endpoints
2. Enable aggressive caching
3. Scale API servers
4. Shed non-critical load

### Database Issues
1. Kill idle connections
2. Reset connection pool
3. Route reads to replica
4. Reduce connection limits

### DDoS Attack
1. Enable Cloudflare protection
2. Set aggressive rate limits
3. Block suspicious IPs
4. Scale edge nodes
5. Page security team

## Integration Points

### Prometheus
- Metrics exposed at `/metrics`
- Custom FreeAgentics metrics with `freeagentics_` prefix
- Agent-specific and system-wide metrics

### Grafana
- Dashboards in `freeagentics` folder
- Alerts configured with notification channels
- Variables for filtering by agent/environment

### Alertmanager
- Routes alerts to appropriate teams
- Escalation policies for critical alerts
- Deduplication and grouping

### PagerDuty
- Critical alerts create incidents
- Integration with on-call schedules
- Automatic escalation

### Slack
- All alerts posted to team channels
- Formatted with severity and runbook links
- Interactive incident updates

## Troubleshooting

### Common Issues

**Metrics not appearing in Prometheus**
- Check `/metrics` endpoint is accessible
- Verify Prometheus scrape configuration
- Check for firewall/network issues

**Dashboards not loading**
- Verify Grafana API key is valid
- Check datasource configuration
- Ensure Prometheus is accessible

**Alerts not firing**
- Check alert rule syntax
- Verify metrics are being collected
- Review threshold values

**Automated responses failing**
- Check script permissions
- Verify API endpoints are accessible
- Review incident response logs

## Maintenance

### Regular Tasks
- Review and adjust alert thresholds monthly
- Update dashboards based on new metrics
- Test incident response playbooks quarterly
- Archive old alert history

### Monitoring the Monitors
- Health check on monitoring infrastructure
- Alert on monitoring failures
- Backup of configurations
- Regular testing of notification channels

## Contact

For monitoring issues or questions:
- Slack: #platform-monitoring
- Email: platform@freeagentics.com
- On-call: PagerDuty