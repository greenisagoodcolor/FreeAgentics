# FreeAgentics Grafana Dashboards Setup Guide

## Overview

This guide provides step-by-step instructions for setting up and configuring Grafana dashboards for the FreeAgentics monitoring system. It covers installation, configuration, dashboard deployment, and customization.

## Prerequisites

- Grafana 9.0+ installed
- Prometheus configured and running
- Network access between Grafana and Prometheus
- Admin access to Grafana
- `curl` and `jq` installed for automation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/freeagentics/monitoring.git
cd monitoring

# Set environment variables
export GRAFANA_URL=http://localhost:3000
export GRAFANA_USER=admin
export GRAFANA_PASSWORD=admin

# Deploy everything
./monitoring/deploy-dashboards.sh all
```

## Manual Setup

### 1. Configure Prometheus Data Source

#### Via UI

1. Navigate to Configuration → Data Sources
2. Click "Add data source"
3. Select "Prometheus"
4. Configure:
   - Name: `FreeAgentics-Prometheus`
   - URL: `http://prometheus:9090`
   - Access: Server (default)
   - HTTP Method: POST
5. Click "Save & test"

#### Via API

```bash
curl -X POST ${GRAFANA_URL}/api/datasources \
  -H "Content-Type: application/json" \
  -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  -d '{
    "name": "FreeAgentics-Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy",
    "basicAuth": false,
    "isDefault": true,
    "jsonData": {
      "httpMethod": "POST",
      "keepCookies": []
    }
  }'
```

### 2. Create Dashboard Folder

```bash
curl -X POST ${GRAFANA_URL}/api/folders \
  -H "Content-Type: application/json" \
  -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  -d '{
    "title": "FreeAgentics",
    "uid": "freeagentics"
  }'
```

### 3. Import Dashboards

#### System Overview Dashboard

```bash
curl -X POST ${GRAFANA_URL}/api/dashboards/db \
  -H "Content-Type: application/json" \
  -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  -d @monitoring/grafana/dashboards/freeagentics-system-overview.json
```

#### Agent Coordination Dashboard

```bash
curl -X POST ${GRAFANA_URL}/api/dashboards/db \
  -H "Content-Type: application/json" \
  -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  -d @monitoring/grafana/dashboards/freeagentics-agent-coordination.json
```

#### Memory Heatmap Dashboard

```bash
curl -X POST ${GRAFANA_URL}/api/dashboards/db \
  -H "Content-Type: application/json" \
  -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  -d @monitoring/grafana/dashboards/freeagentics-memory-heatmap.json
```

#### API Performance Dashboard

```bash
curl -X POST ${GRAFANA_URL}/api/dashboards/db \
  -H "Content-Type: application/json" \
  -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  -d @monitoring/grafana/dashboards/freeagentics-api-performance.json
```

#### Capacity Planning Dashboard

```bash
curl -X POST ${GRAFANA_URL}/api/dashboards/db \
  -H "Content-Type: application/json" \
  -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  -d @monitoring/grafana/dashboards/freeagentics-capacity-planning.json
```

## Dashboard Provisioning

### Automatic Provisioning

Create provisioning configuration files:

#### Dashboard Provider

`/etc/grafana/provisioning/dashboards/freeagentics.yaml`:

```yaml
apiVersion: 1

providers:
  - name: 'FreeAgentics Dashboards'
    orgId: 1
    folder: 'FreeAgentics'
    folderUid: 'freeagentics'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards/freeagentics
      foldersFromFilesStructure: false
```

#### Data Source Provider

`/etc/grafana/provisioning/datasources/prometheus.yaml`:

```yaml
apiVersion: 1

datasources:
  - name: FreeAgentics-Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      httpMethod: POST
      timeInterval: 15s
      exemplarTraceIdDestinations:
        - name: trace_id
          url: http://jaeger:16686
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  grafana:
    image: grafana/grafana:9.5.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards/freeagentics
    networks:
      - monitoring

volumes:
  grafana-storage:

networks:
  monitoring:
```

## Dashboard Customization

### Variables Configuration

Each dashboard includes template variables for filtering:

#### Agent Variables

```json
{
  "name": "agent_id",
  "type": "query",
  "query": "label_values(freeagentics_agent_memory_usage_bytes, agent_id)",
  "refresh": 2,
  "multi": true,
  "includeAll": true,
  "allValue": ".*"
}
```

#### Endpoint Variables

```json
{
  "name": "endpoint",
  "type": "query",
  "query": "label_values(http_requests_total, handler)",
  "refresh": 2,
  "multi": true,
  "includeAll": true,
  "regex": "/api/v1/(.*)"
}
```

### Panel Customization

#### Modify Thresholds

```json
{
  "fieldConfig": {
    "defaults": {
      "thresholds": {
        "mode": "absolute",
        "steps": [
          { "color": "green", "value": null },
          { "color": "yellow", "value": 30 },
          { "color": "red", "value": 34.5 }
        ]
      }
    }
  }
}
```

#### Add Custom Panels

```json
{
  "title": "Custom Metric",
  "type": "timeseries",
  "targets": [
    {
      "expr": "rate(custom_metric_total[5m])",
      "legendFormat": "{{label_name}}"
    }
  ]
}
```

## Alert Configuration

### Create Alert Rules

```json
{
  "alert": {
    "name": "High Memory Usage",
    "conditions": [
      {
        "evaluator": {
          "params": [30],
          "type": "gt"
        },
        "query": {
          "params": ["A", "5m", "now"]
        },
        "reducer": {
          "type": "avg"
        },
        "type": "query"
      }
    ],
    "noDataState": "no_data",
    "executionErrorState": "alerting",
    "frequency": "60s",
    "for": "5m"
  }
}
```

### Configure Notification Channels

#### Slack Integration

```bash
curl -X POST ${GRAFANA_URL}/api/alert-notifications \
  -H "Content-Type: application/json" \
  -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  -d '{
    "name": "FreeAgentics Slack",
    "type": "slack",
    "isDefault": true,
    "settings": {
      "url": "${SLACK_WEBHOOK_URL}",
      "channel": "#alerts",
      "username": "Grafana"
    }
  }'
```

#### Email Integration

```bash
curl -X POST ${GRAFANA_URL}/api/alert-notifications \
  -H "Content-Type: application/json" \
  -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  -d '{
    "name": "FreeAgentics Email",
    "type": "email",
    "settings": {
      "addresses": "alerts@freeagentics.com;oncall@freeagentics.com"
    }
  }'
```

## Performance Optimization

### Query Optimization

1. **Use Recording Rules**
   ```yaml
   # prometheus.rules.yml
   groups:
     - name: freeagentics_recordings
       interval: 30s
       rules:
         - record: freeagentics:api_request_rate_5m
           expr: rate(http_requests_total[5m])
         - record: freeagentics:api_error_rate_5m
           expr: rate(http_requests_total{status=~"5.."}[5m])
   ```

2. **Optimize Time Ranges**
   ```json
   {
     "targets": [{
       "expr": "increase(metric_total[$__interval])",
       "interval": "1m"
     }]
   }
   ```

3. **Use Appropriate Functions**
   - Use `rate()` for counters
   - Use `delta()` for gauges
   - Use `increase()` for totals over time

### Dashboard Performance

1. **Limit Concurrent Queries**
   ```json
   {
     "options": {
       "maxDataPoints": 300,
       "queryTimeout": "30s"
     }
   }
   ```

2. **Set Appropriate Refresh Rates**
   - Overview: 30s
   - Detailed: 10s
   - Historical: 1m

3. **Use Caching**
   ```ini
   # grafana.ini
   [caching]
   enabled = true
   ttl = 60
   ```

## Backup and Recovery

### Backup Dashboards

```bash
#!/bin/bash
# backup-dashboards.sh

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p ${BACKUP_DIR}

# Export all dashboards
for dashboard in $(curl -s -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  ${GRAFANA_URL}/api/search?folderIds=1 | jq -r '.[] | .uid'); do

  curl -s -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
    ${GRAFANA_URL}/api/dashboards/uid/${dashboard} \
    | jq '.dashboard' > ${BACKUP_DIR}/${dashboard}.json
done

# Export data sources
curl -s -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
  ${GRAFANA_URL}/api/datasources \
  > ${BACKUP_DIR}/datasources.json

echo "Backup completed: ${BACKUP_DIR}"
```

### Restore Dashboards

```bash
#!/bin/bash
# restore-dashboards.sh

BACKUP_DIR=$1

# Restore data sources
for ds in $(cat ${BACKUP_DIR}/datasources.json | jq -r '.[] | @base64'); do
  echo ${ds} | base64 -d | curl -X POST ${GRAFANA_URL}/api/datasources \
    -H "Content-Type: application/json" \
    -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
    -d @-
done

# Restore dashboards
for dashboard in ${BACKUP_DIR}/*.json; do
  if [[ $(basename $dashboard) != "datasources.json" ]]; then
    curl -X POST ${GRAFANA_URL}/api/dashboards/db \
      -H "Content-Type: application/json" \
      -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
      -d "{\"dashboard\": $(cat $dashboard), \"overwrite\": true}"
  fi
done
```

## Troubleshooting

### Common Issues

#### No Data in Dashboards

1. **Check Data Source**
   ```bash
   curl -u ${GRAFANA_USER}:${GRAFANA_PASSWORD} \
     ${GRAFANA_URL}/api/datasources/proxy/1/api/v1/query?query=up
   ```

2. **Verify Metrics**
   ```bash
   curl http://prometheus:9090/api/v1/label/__name__/values | grep freeagentics
   ```

3. **Check Time Range**
   - Ensure data exists in selected time range
   - Check timezone settings

#### Slow Dashboard Loading

1. **Check Query Performance**
   - Use Grafana query inspector
   - Look for expensive queries
   - Add recording rules

2. **Optimize Panels**
   - Reduce number of panels
   - Increase query intervals
   - Use query caching

#### Variables Not Working

1. **Check Query Syntax**
   ```promql
   label_values(metric_name, label_name)
   ```

2. **Verify Label Existence**
   ```bash
   curl http://prometheus:9090/api/v1/labels
   ```

### Debug Mode

Enable debug logging:

```ini
# grafana.ini
[log]
level = debug

[log.console]
level = debug
```

View logs:
```bash
docker logs grafana 2>&1 | grep -i error
journalctl -u grafana -f
```

## Best Practices

### Dashboard Design

1. **Consistent Layout**
   - Use 24-column grid
   - Standard panel heights
   - Logical grouping

2. **Clear Naming**
   - Descriptive titles
   - Include units
   - Add descriptions

3. **Color Consistency**
   - Green: Good/Healthy
   - Yellow: Warning
   - Red: Critical/Error

### Query Best Practices

1. **Use Labels Effectively**
   ```promql
   sum by (endpoint) (rate(http_requests_total[5m]))
   ```

2. **Avoid Expensive Queries**
   ```promql
   # Bad: unbounded query
   http_requests_total

   # Good: time-bounded rate
   rate(http_requests_total[5m])
   ```

3. **Use Recording Rules**
   - For frequently used queries
   - For complex calculations
   - For dashboard performance

### Maintenance

1. **Regular Reviews**
   - Monthly dashboard review
   - Remove unused panels
   - Update thresholds

2. **Version Control**
   - Store dashboards in Git
   - Use meaningful commit messages
   - Tag releases

3. **Documentation**
   - Document custom panels
   - Explain complex queries
   - Maintain runbooks

## Integration with CI/CD

### Automated Deployment

```yaml
# .gitlab-ci.yml
deploy-dashboards:
  stage: deploy
  script:
    - ./monitoring/deploy-dashboards.sh all
  only:
    changes:
      - monitoring/grafana/dashboards/*
```

### Validation

```bash
#!/bin/bash
# validate-dashboards.sh

for dashboard in monitoring/grafana/dashboards/*.json; do
  # Check JSON syntax
  jq . ${dashboard} > /dev/null || exit 1

  # Check required fields
  jq -e '.title' ${dashboard} > /dev/null || exit 1
  jq -e '.panels' ${dashboard} > /dev/null || exit 1
done
```

---

**Last Updated**: 2025-01-15
**Version**: 1.0
**Next Review**: 2025-02-15
