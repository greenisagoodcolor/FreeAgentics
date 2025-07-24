# FreeAgentics Performance Monitoring & Analysis Guide

## Overview

The FreeAgentics performance monitoring system provides comprehensive observability across all system components with real-time metrics collection, anomaly detection, performance profiling, regression analysis, and automated reporting.

## Quick Start

### 1. Start the Integrated Monitoring System

```bash
# Start with default configuration
python -m tests.performance.integrated_monitoring_system start

# Start with custom configuration
python -m tests.performance.integrated_monitoring_system start --config monitoring_config.yaml

# Start with specific options
python -m tests.performance.integrated_monitoring_system start \
  --dashboard \
  --port 8090 \
  --profiling \
  --regression
```

### 2. Access the Dashboard

Open your browser to: `http://localhost:8090`

The dashboard provides:

- Real-time performance metrics
- Live charts and visualizations
- Alert notifications
- System health status

### 3. Generate Reports

```bash
# Generate a performance report
python -m tests.performance.integrated_monitoring_system report \
  --format markdown \
  --window 3600

# Establish performance baselines
python -m tests.performance.integrated_monitoring_system baseline \
  --duration 300
```

## Architecture

### Components

1. **Unified Metrics Collector** (`unified_metrics_collector.py`)

   - Centralized metrics aggregation
   - Time-series data storage
   - Statistical calculations
   - Alert rule evaluation

2. **Monitoring Dashboard** (`monitoring_dashboard.py`)

   - Web-based real-time visualization
   - WebSocket live updates
   - Interactive charts
   - Alert display

3. **Performance Profiler** (`performance_profiler.py`)

   - CPU and memory profiling
   - Component-level analysis
   - Bottleneck identification
   - Optimization recommendations

4. **Regression Analyzer** (`regression_analyzer.py`)

   - Automated regression detection
   - Baseline comparison
   - Trend analysis
   - Architectural limit validation

5. **Anomaly Detector** (`anomaly_detector.py`)

   - Statistical anomaly detection
   - Machine learning models
   - Pattern recognition
   - Correlated anomaly analysis

6. **Report Generator** (`performance_report_generator.py`)

   - Comprehensive report generation
   - Multiple output formats
   - Visualization creation
   - Historical analysis

7. **Integrated System** (`integrated_monitoring_system.py`)
   - Unified interface
   - Component orchestration
   - Background task management
   - CLI interface

## Metrics Collected

### Database Metrics

- `query_latency_ms` - Database query execution time
- `connection_pool_size` - Active database connections
- `transaction_rate` - Transactions per second
- `query_count` - Total queries executed
- `error_count` - Database errors

### WebSocket Metrics

- `connections_per_second` - New connection rate
- `messages_per_second` - Message throughput
- `current_latency_ms` - Message latency
- `error_rate` - Connection/message errors
- `active_connections` - Current connections

### Agent Metrics

- `inference_time_ms` - Agent inference latency
- `active_agents` - Number of active agents
- `agent_throughput` - Operations per second
- `belief_updates_per_sec` - Belief update rate
- `free_energy_avg` - Average free energy

### System Metrics

- `cpu_usage_percent` - CPU utilization
- `memory_usage_percent` - Memory utilization
- `memory_available_mb` - Available memory
- `disk_read_mb_per_sec` - Disk read rate
- `disk_write_mb_per_sec` - Disk write rate

## Alert Configuration

Alerts are configured in `monitoring_config.yaml`:

```yaml
alert_rules:
  - name: "High Database Query Latency"
    metric: "query_latency_ms"
    source: "database"
    condition: "p95 > 100" # 95th percentile > 100ms
    threshold: 100
    severity: "warning"
```

### Alert Conditions

- `avg` - Average value
- `min` - Minimum value
- `max` - Maximum value
- `p50` - 50th percentile (median)
- `p95` - 95th percentile
- `p99` - 99th percentile
- `rate` - Rate of change

### Severity Levels

- `info` - Informational
- `warning` - Requires attention
- `critical` - Immediate action required

## Profiling

### Profile a Specific Operation

```python
from tests.performance.performance_profiler import profile_operation

@profile_operation("database", "complex_query")
async def complex_database_operation():
    # Your code here
    pass
```

### Manual Profiling

```python
from tests.performance.performance_profiler import component_profiler

async with component_profiler.profile_component_async("agent", "inference"):
    # Code to profile
    result = await agent.infer()
```

### View Profiling Results

```python
# Get component summary
summary = component_profiler.get_component_summary("database")

# Export profiles
component_profiler.export_profiles(
    Path("profiles.json"),
    component="database",
    format="json"
)
```

## Anomaly Detection

The system automatically detects anomalies using:

1. **Statistical Methods**

   - Z-score analysis
   - Interquartile range (IQR)
   - Trend detection

2. **Machine Learning**

   - Isolation Forest
   - Pattern recognition
   - Multivariate analysis

3. **Threshold Detection**
   - Configurable thresholds
   - Dynamic baselines

### Anomaly Types

- `spike` - Sudden increase
- `drop` - Sudden decrease
- `trend` - Gradual change
- `pattern` - Pattern deviation
- `multivariate` - Multi-metric anomaly
- `threshold` - Threshold breach

## Regression Analysis

### Establish Baselines

```bash
python -m tests.performance.integrated_monitoring_system baseline --duration 300
```

### Automatic Regression Detection

The system continuously monitors for:

- Performance degradation > 10%
- Architectural limit violations
- Efficiency reductions
- Resource usage increases

### Manual Analysis

```python
from tests.performance.regression_analyzer import regression_analyzer

# Analyze current performance
regressions = await regression_analyzer.analyze_regressions()

# Generate regression report
report = regression_analyzer.generate_regression_report(
    regressions,
    include_recommendations=True
)
```

## Load Testing Integration

Enable load testing in configuration:

```yaml
enable_load_testing: true
load_test_scenarios:
  - database
  - websocket
  - agent
  - coordination
```

The system will automatically:

1. Run load tests periodically
2. Collect performance metrics
3. Detect regressions
4. Generate reports

## Reports

### Report Types

1. **HTML Report** - Interactive web report
2. **JSON Report** - Machine-readable data
3. **Markdown Report** - Documentation format

### Report Contents

- Performance summary
- Metric statistics
- Regression analysis
- Anomaly detection
- Profiling results
- Recommendations

### Automated Reporting

Configure in `monitoring_config.yaml`:

```yaml
enable_automated_reports: true
report_interval_hours: 24
report_formats:
  - html
  - json
  - markdown
```

## Best Practices

### 1. Baseline Establishment

- Run baseline collection during normal operation
- Update baselines after major changes
- Use at least 5 minutes of data

### 2. Alert Tuning

- Start with conservative thresholds
- Adjust based on false positive rate
- Use percentiles for variable metrics

### 3. Dashboard Usage

- Monitor during load tests
- Watch for correlated anomalies
- Use for troubleshooting

### 4. Performance Optimization

- Profile before optimizing
- Focus on identified bottlenecks
- Validate improvements with metrics

## Troubleshooting

### High Memory Usage

1. Check `memory_usage_percent` metric
2. Review profiling data for memory leaks
3. Analyze `memory_allocated_mb` trends

### Slow Response Times

1. Monitor `*_latency_ms` metrics
2. Check CPU usage patterns
3. Review database query performance

### Connection Issues

1. Monitor `error_rate` metrics
2. Check connection pool usage
3. Review WebSocket statistics

## API Reference

### Metrics Collection

```python
from tests.performance.unified_metrics_collector import (
    record_metric, MetricSource, MetricType
)

# Record a metric
record_metric(
    name="custom_metric",
    value=42.5,
    source=MetricSource.AGENT,
    type=MetricType.GAUGE,
    tags={"component": "belief_update"}
)
```

### Custom Alerts

```python
from tests.performance.unified_metrics_collector import unified_collector

# Add custom alert rule
unified_collector.add_alert_rule(
    name="Custom Alert",
    metric_name="custom_metric",
    source=MetricSource.AGENT,
    condition="avg > 100",
    threshold=100,
    severity="warning"
)
```

### Anomaly Detection

```python
from tests.performance.anomaly_detector import detect_anomalies, MetricPoint

# Detect anomalies for a metric
point = MetricPoint(
    timestamp=datetime.now(),
    source=MetricSource.AGENT,
    name="inference_time_ms",
    value=150.0,
    type=MetricType.HISTOGRAM
)

anomalies = await detect_anomalies(point)
```

## Configuration Reference

See `monitoring_config.yaml` for all configuration options:

- Metrics collection settings
- Dashboard configuration
- Profiling options
- Regression analysis parameters
- Alert rules
- Anomaly detection settings
- Report generation options

## CLI Commands

```bash
# Start monitoring
python -m tests.performance.integrated_monitoring_system start

# Generate report
python -m tests.performance.integrated_monitoring_system report

# Establish baselines
python -m tests.performance.integrated_monitoring_system baseline

# Show help
python -m tests.performance.integrated_monitoring_system --help
```

## Integration Examples

### With Load Tests

```python
from tests.performance.integrated_monitoring_system import IntegratedMonitoringSystem

# Start monitoring
monitoring = IntegratedMonitoringSystem()
await monitoring.start()

# Run load test
await run_load_test()

# Generate report
reports = await monitoring.generate_comprehensive_report()
```

### With Unit Tests

```python
import pytest
from tests.performance.performance_profiler import profile_operation

@pytest.mark.performance
@profile_operation("test", "integration")
async def test_performance():
    # Test code
    pass
```

## Performance Targets

Based on architectural analysis:

- **Agent Inference**: < 10ms target
- **Database Queries**: < 50ms p95
- **WebSocket Latency**: < 100ms p95
- **Agent Scalability**: 50 agents @ 28.4% efficiency
- **Memory per Agent**: < 34.5MB

---

For more information, see the individual component documentation or run `python -m tests.performance.integrated_monitoring_system --help`.
