# Prometheus Metrics Implementation

## Overview

This document describes the Prometheus metrics implementation for FreeAgentics, following Charity Majors' observability principles: "Observability is not optional" and "You can't protect what you can't see."

## Implementation Details

### Endpoint

The Prometheus metrics are exposed at `/metrics` endpoint in the standard Prometheus exposition format.

### Required Counters

As per the requirements, the following counters have been implemented:

1. **agent_spawn_total** - Tracks the total number of agents spawned
   - Labels: `agent_type` (e.g., "active_inference", "llm")
   - Incremented when agents are created via the API

2. **kg_node_total** - Tracks the total number of knowledge graph nodes created
   - Labels: `node_type` (e.g., "entity", "relation", "attribute")
   - Will be incremented when KG nodes are created (API not fully implemented yet)

### HTTP Request Metrics

3. **http_requests_total** - Total number of HTTP requests
   - Labels: `method`, `endpoint`, `status`
   - Automatically collected by MetricsMiddleware

4. **http_request_duration_seconds** - HTTP request duration histogram
   - Labels: `method`, `endpoint`
   - Buckets: 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0 seconds

### System Health Metrics

5. **system_cpu_usage_percent** - System CPU usage percentage
6. **system_memory_usage_bytes** - System memory usage in bytes

Additional FreeAgentics-specific metrics are also available with the `freeagentics_` prefix.

## Architecture

### Components

1. **observability/prometheus_metrics.py**
   - Defines all metrics using prometheus_client
   - Provides helper functions for recording metrics
   - Implements PrometheusMetricsCollector class

2. **api/middleware/metrics.py**
   - MetricsMiddleware automatically tracks HTTP requests
   - Records request duration and status codes
   - Never breaks the application if metrics collection fails

3. **main.py**
   - Exposes `/metrics` endpoint
   - Returns metrics in Prometheus format
   - Falls back gracefully if prometheus_client is not available

### Integration Points

- **Agent Creation**: The `create_agent` endpoint in `api/v1/agents.py` calls `record_agent_spawn()`
- **HTTP Requests**: MetricsMiddleware automatically tracks all HTTP requests
- **System Metrics**: Collected periodically by the PrometheusMetricsCollector

## Usage

### Scraping Metrics

Configure Prometheus to scrape the `/metrics` endpoint:

```yaml
scrape_configs:
  - job_name: 'freeagentics'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
```

### Example Metrics Output

```
# HELP agent_spawn_total Total number of agents spawned
# TYPE agent_spawn_total counter
agent_spawn_total{agent_type="active_inference"} 1.0
agent_spawn_total{agent_type="llm"} 2.0

# HELP kg_node_total Total number of knowledge graph nodes created
# TYPE kg_node_total counter
kg_node_total{node_type="entity"} 5.0
kg_node_total{node_type="relation"} 3.0

# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health",status="200"} 42.0
http_requests_total{method="POST",endpoint="/api/v1/agents",status="201"} 3.0
```

## Security Considerations

- Metrics endpoint is publicly accessible (no authentication required per Prometheus standards)
- No sensitive information is exposed in metrics
- Metrics collection never interferes with request processing
- Failed metrics recording is logged but doesn't break the application

## Performance Impact

- Minimal overhead for counter increments
- HTTP request timing adds ~0.1ms per request
- Metrics are stored in memory (no database writes)
- Prometheus scraping is pull-based (no push overhead)

## Future Enhancements

1. Add authentication for metrics endpoint if required
2. Implement kg_node_total integration when KG API is complete
3. Add custom business metrics as needed
4. Consider adding metric aggregation for distributed deployments

## Testing

Tests are provided in:
- `tests/integration/test_prometheus_metrics.py` - Integration tests
- `tests/unit/test_prometheus_metrics.py` - Unit tests

Run tests with:
```bash
pytest tests/integration/test_prometheus_metrics.py -v
pytest tests/unit/test_prometheus_metrics.py -v
```

## Dependencies

- prometheus-client==0.20.0 (added to requirements.txt)

## References

- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Charity Majors on Observability](https://charity.wtf/tag/observability/)
- [OpenTelemetry Metrics](https://opentelemetry.io/docs/concepts/signals/metrics/)