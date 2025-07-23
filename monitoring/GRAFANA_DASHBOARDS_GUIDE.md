# FreeAgentics Grafana Dashboards Guide

## Overview

This guide documents the comprehensive Grafana dashboard suite for FreeAgentics production monitoring. The dashboards provide real-time visibility into system performance, agent coordination, memory usage, API performance, and capacity planning.

## Dashboard Architecture

### Dashboard Files

- **System Overview**: `freeagentics-system-overview.json`
- **Agent Coordination**: `freeagentics-agent-coordination.json`
- **Memory Heatmap**: `freeagentics-memory-heatmap.json`
- **API Performance**: `freeagentics-api-performance.json`
- **Capacity Planning**: `freeagentics-capacity-planning.json`

### Provisioning Configuration

- **Dashboard Provisioning**: `provisioning/dashboards/freeagentics-dashboards.yaml`
- **Datasource Provisioning**: `provisioning/datasources/prometheus-datasource.yaml`
- **Deployment Script**: `deploy-dashboards.sh`

## Dashboard Descriptions

### 1. FreeAgentics System Overview

**Purpose**: High-level system health and performance overview
**Refresh Rate**: 30 seconds
**Key Metrics**:
- System status (up/down)
- Active agents count
- Memory usage
- Request rate
- Error rate
- Knowledge graph growth

**Panels**:
- System Status (stat)
- Active Agents (timeseries)
- Memory Usage (timeseries)
- Request Rate (timeseries)
- Error Rate (timeseries)
- Knowledge Graph Growth (timeseries)

**Use Cases**:
- Operations team dashboard
- Executive summary view
- System health monitoring
- Incident response overview

### 2. FreeAgentics Agent Coordination

**Purpose**: Real-time multi-agent coordination metrics and efficiency tracking
**Refresh Rate**: 10 seconds
**Key Metrics**:
- Active agents count
- Concurrent coordination sessions
- Coordination timeout rate
- Coordination success rate
- Duration percentiles (P50, P90, P95)
- Belief system metrics

**Panels**:
- Active Agents (stat)
- Concurrent Sessions (stat)
- Coordination Timeout Rate (stat)
- Coordination Success Rate (stat)
- Coordination Duration Percentiles (timeseries)
- Request Rate by Type (timeseries)
- Belief Convergence Time (timeseries)
- Belief System Free Energy (timeseries)
- Coordination Types Distribution (pie chart)
- Belief Accuracy (timeseries)

**Variables**:
- Agent ID (multi-select)

**Use Cases**:
- Agent team monitoring
- Coordination performance analysis
- Belief system health tracking
- Multi-agent system optimization

### 3. FreeAgentics Memory Usage Heatmap

**Purpose**: Per-agent memory usage visualization and analysis
**Refresh Rate**: 30 seconds
**Key Metrics**:
- Total agent memory usage
- Average memory usage per agent
- Peak memory usage
- Agents over memory limit (30MB threshold)
- Memory usage by agent type
- Memory usage distribution

**Panels**:
- Total Agent Memory Usage (stat)
- Average Agent Memory Usage (stat)
- Peak Agent Memory Usage (stat)
- Agents Over Memory Limit (stat)
- Agent Memory Usage Heatmap (heatmap)
- Top 10 Memory Consumers (timeseries)
- Memory Usage Rate of Change (timeseries)
- Memory Usage by Agent Type (pie chart)
- Memory Usage Distribution (timeseries)

**Variables**:
- Agent Type (multi-select)
- Agent ID (multi-select, filtered by type)

**Use Cases**:
- Memory optimization
- Resource allocation planning
- Agent performance analysis
- Memory leak detection

### 4. FreeAgentics API Performance

**Purpose**: API latency percentiles, error rates, and throughput metrics
**Refresh Rate**: 10 seconds
**Key Metrics**:
- P95 response time
- Error rate
- Request rate
- Success rate
- Response time percentiles
- Request distribution by endpoint
- Business metrics

**Panels**:
- P95 Response Time (stat)
- Error Rate (stat)
- Request Rate (stat)
- Success Rate (stat)
- Response Time Percentiles (timeseries)
- Request Rate by Endpoint (timeseries)
- Request Rate by Status Code (timeseries)
- P95 Response Time by Endpoint (timeseries)
- Traffic Distribution by Endpoint (pie chart)
- Business Metrics (timeseries)
- Response Quality Score (timeseries)
- Authentication Attempts (timeseries)

**Variables**:
- Endpoint (multi-select)
- Method (multi-select)

**Use Cases**:
- API performance monitoring
- SLA compliance tracking
- Performance optimization
- User experience monitoring

### 5. FreeAgentics Capacity Planning

**Purpose**: Resource utilization trends and capacity forecasting
**Refresh Rate**: 1 minute
**Key Metrics**:
- System resource usage (CPU, memory, disk)
- Agent capacity utilization
- Growth trends
- Performance trends
- Capacity forecasting

**Panels**:
- System Memory Usage (stat)
- System CPU Usage (stat)
- System Disk Usage (stat)
- Agent Capacity Utilization (stat)
- System Resource Usage Trends (timeseries)
- Agent Capacity Trends (timeseries)
- Memory Usage Trends (timeseries)
- Knowledge Graph Growth (timeseries)
- Request Rate Trends (timeseries)
- Performance Trends (timeseries)
- Capacity Forecasting (1 Hour) (timeseries)
- Growth Forecasting (timeseries)

**Variables**:
- Forecast Period (1h, 6h, 24h)

**Use Cases**:
- Capacity planning
- Resource allocation decisions
- Growth forecasting
- Infrastructure scaling

## Deployment

### Prerequisites

- Grafana instance running
- Prometheus configured with FreeAgentics metrics
- `curl` and `jq` installed on deployment system

### Quick Deployment

```bash
# Deploy all dashboards
./deploy-dashboards.sh

# Deploy specific components
./deploy-dashboards.sh datasource  # Create Prometheus datasource
./deploy-dashboards.sh dashboards  # Deploy dashboards only
./deploy-dashboards.sh alerts      # Setup alerts only
./deploy-dashboards.sh verify      # Verify deployment
```

### Manual Deployment

1. **Create Prometheus Datasource**:
   - URL: `http://prometheus:9090`
   - Access: Server (default)
   - HTTP Method: POST

2. **Import Dashboards**:
   - Copy dashboard JSON files to Grafana
   - Import via Grafana UI or API
   - Organize in "FreeAgentics" folder

3. **Configure Variables**:
   - Verify datasource variables point to correct Prometheus instance
   - Test query variables work correctly

### Environment Variables

```bash
# Grafana Configuration
GRAFANA_URL=http://localhost:3000
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin

# Prometheus Configuration
PROMETHEUS_URL=http://prometheus:9090

# Slack Integration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

## Dashboard Navigation

### Navigation Flow

1. **System Overview** → General health check
2. **Agent Coordination** → Multi-agent specific metrics
3. **Memory Heatmap** → Resource usage analysis
4. **API Performance** → User-facing performance
5. **Capacity Planning** → Future resource needs

### Drill-Down Patterns

- **Overview → Detail**: Start with system overview, drill into specific components
- **Time-based Analysis**: Use different time ranges for different analysis types
- **Filter by Component**: Use variables to focus on specific agents/endpoints
- **Correlation Analysis**: Use multiple dashboards to correlate issues

## Alert Integration

### Grafana Alerts

The dashboards include alert rules for:
- High response times (P95 > 500ms)
- High error rates (> 10%)
- Memory usage warnings (> 30MB per agent)
- Agent coordination failures
- System resource exhaustion

### Alert Channels

- **Slack**: Real-time notifications
- **Email**: Escalation notifications
- **PagerDuty**: Critical alerts
- **Webhook**: Custom integrations

## Troubleshooting

### Common Issues

1. **No Data Visible**:
   - Check Prometheus datasource configuration
   - Verify metrics are being scraped
   - Check time range selection

2. **Queries Timing Out**:
   - Reduce query complexity
   - Increase timeout settings
   - Check Prometheus performance

3. **Variables Not Working**:
   - Verify label values exist in metrics
   - Check regex patterns
   - Test queries manually

### Performance Optimization

1. **Query Optimization**:
   - Use recording rules for expensive queries
   - Implement appropriate caching
   - Optimize time ranges

2. **Dashboard Performance**:
   - Limit concurrent queries
   - Use appropriate refresh rates
   - Implement query result caching

## Metrics Reference

### System Metrics

- `up{job="freeagentics-backend"}` - System availability
- `http_requests_total` - HTTP request counts
- `http_request_duration_seconds` - Request latency
- `freeagentics_system_memory_usage_bytes` - System memory usage
- `freeagentics_system_active_agents_total` - Active agent count

### Agent Metrics

- `freeagentics_agent_memory_usage_bytes` - Per-agent memory usage
- `freeagentics_agent_coordination_requests_total` - Coordination requests
- `freeagentics_agent_coordination_duration_seconds` - Coordination duration
- `freeagentics_agent_coordination_errors_total` - Coordination errors
- `freeagentics_agent_coordination_concurrent_sessions` - Concurrent sessions

### Belief System Metrics

- `freeagentics_belief_free_energy_current` - Current free energy
- `freeagentics_belief_convergence_time_seconds` - Belief convergence time
- `freeagentics_belief_accuracy_ratio` - Belief accuracy ratio
- `freeagentics_belief_prediction_errors_total` - Prediction errors

### Business Metrics

- `freeagentics_business_user_interactions_total` - User interactions
- `freeagentics_business_inference_operations_total` - Inference operations
- `freeagentics_business_response_quality_score` - Response quality
- `freeagentics_system_knowledge_graph_nodes_total` - Knowledge graph size

### Security Metrics

- `freeagentics_security_authentication_attempts_total` - Auth attempts
- `freeagentics_security_anomaly_detections_total` - Security anomalies
- `freeagentics_security_access_violations_total` - Access violations

## Best Practices

### Dashboard Design

1. **Consistent Layout**:
   - Use standard panel sizes
   - Maintain consistent color schemes
   - Follow logical information flow

2. **Meaningful Metrics**:
   - Focus on actionable metrics
   - Use appropriate units and scales
   - Include relevant thresholds

3. **User Experience**:
   - Optimize for common use cases
   - Provide clear navigation
   - Include helpful descriptions

### Operations

1. **Regular Maintenance**:
   - Review dashboard relevance
   - Update thresholds based on experience
   - Remove obsolete metrics

2. **Performance Monitoring**:
   - Monitor dashboard load times
   - Optimize expensive queries
   - Use appropriate refresh rates

3. **Team Collaboration**:
   - Share dashboard URLs
   - Document common workflows
   - Train team on dashboard usage

## Advanced Features

### Templating

- **Multi-select Variables**: Filter by multiple agents/endpoints
- **Chained Variables**: Agent type → Agent ID filtering
- **Custom Variables**: Environment-specific configurations

### Annotations

- **Deployment Events**: Mark release deployments
- **Incident Events**: Track system incidents
- **Maintenance Windows**: Schedule maintenance periods

### Linking

- **Cross-dashboard Links**: Navigate between related dashboards
- **External Links**: Link to runbooks and documentation
- **Drill-down Links**: Deep-dive into specific metrics

## Integration Points

### Prometheus Integration

- **Recording Rules**: Pre-calculated metrics for performance
- **Alert Rules**: Automated alerting based on thresholds
- **Service Discovery**: Automatic target discovery

### AlertManager Integration

- **Alert Routing**: Route alerts to appropriate teams
- **Alert Grouping**: Group related alerts
- **Alert Inhibition**: Prevent alert storms

### External Tools

- **Jaeger**: Distributed tracing integration
- **Loki**: Log aggregation integration
- **PagerDuty**: Incident management integration

---

**Last Updated**: 2024-07-15
**Dashboard Version**: 1.0
**Contact**: sre@freeagentics.com
