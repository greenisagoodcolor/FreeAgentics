# PyMDP Performance Benchmark Suite Design

## Overview

This document outlines the design for comprehensive performance benchmarking of PyMDP operations in FreeAgentics. Based on the NEMESIS audit findings, we need to validate the actual ~9x performance improvement achieved through optimizations.

## Benchmark Categories

### 1. Core Inference Operations

#### 1.1 Belief State Updates

- **Metric**: Time per belief update (ms)
- **Parameters**:
  - State space size: [10, 50, 100, 500, 1000]
  - Number of modalities: [1, 2, 3, 5]
  - Observation precision levels
- **Test scenarios**:
  - Single observation update
  - Batch observation updates
  - Sparse vs dense observations

#### 1.2 Expected Free Energy (EFE) Calculations

- **Metric**: Computation time (ms), memory usage (MB)
- **Parameters**:
  - Policy depth: [1, 3, 5, 10]
  - Number of policies: [10, 50, 100, 500]
  - State/action space size
- **Test scenarios**:
  - Full EFE computation
  - Selective policy evaluation
  - Cached vs uncached calculations

#### 1.3 Policy Selection

- **Metric**: Selection time (ms), accuracy
- **Parameters**:
  - Number of policies
  - Temperature parameter
  - Prior preferences strength
- **Test scenarios**:
  - Greedy selection
  - Softmax selection
  - Multi-objective optimization

### 2. Optimization Benchmarks

#### 2.1 Matrix Caching Performance

- **Metric**: Cache hit rate, memory overhead, speedup factor
- **Test scenarios**:
  - Cold start (empty cache)
  - Warm cache (pre-populated)
  - Cache invalidation patterns
  - Memory pressure scenarios

#### 2.2 Selective Update Mechanisms

- **Metric**: Computation savings (%), accuracy maintained
- **Test scenarios**:
  - Partial belief updates
  - Sparse observation handling
  - Hierarchical state updates
  - Change detection thresholds

### 3. Scalability Tests

#### 3.1 Agent Scaling

- **Metric**: Performance degradation per agent
- **Parameters**:
  - Number of agents: [1, 5, 10, 25, 50, 100]
  - Inter-agent communication frequency
  - Shared belief state size
- **Test scenarios**:
  - Independent agents
  - Communicating agents
  - Shared world model

#### 3.2 State Space Scaling

- **Metric**: Computation time growth rate
- **Parameters**:
  - State dimensions: exponential growth
  - Factorized vs full state representation
- **Test scenarios**:
  - Linear state growth
  - Exponential state growth
  - Hierarchical decomposition

### 4. Memory Usage Patterns

#### 4.1 Belief State Memory

- **Metric**: Memory per agent (MB)
- **Test scenarios**:
  - Baseline memory usage
  - Memory growth over time
  - Memory after optimization

#### 4.2 Matrix Storage

- **Metric**: Memory for transition/observation matrices
- **Test scenarios**:
  - Dense matrix storage
  - Sparse matrix optimization
  - Shared matrix structures

## Performance Metrics

### Primary Metrics

1. **Execution Time**: Wall-clock time for operations
2. **Memory Usage**: Peak and average memory consumption
3. **Throughput**: Operations per second
4. **Latency**: Response time distribution

### Secondary Metrics

1. **CPU Utilization**: Core usage patterns
2. **Cache Efficiency**: L1/L2/L3 cache hit rates
3. **GC Pressure**: Garbage collection frequency/duration
4. **Energy Efficiency**: Operations per watt (if measurable)

## Benchmark Implementation Structure

```python
class PyMDPBenchmark:
    """Base class for PyMDP benchmarks."""

    def setup(self):
        """Initialize test environment."""
        pass

    def teardown(self):
        """Cleanup after benchmark."""
        pass

    def run(self, iterations=100):
        """Execute benchmark with timing."""
        pass

    def report(self):
        """Generate performance report."""
        pass
```

## Validation Criteria

### Performance Targets (based on NEMESIS audit)

- Belief updates: < 10ms for 100-state system
- EFE calculation: < 50ms for 50 policies
- Memory per agent: < 35MB (current baseline)
- Cache hit rate: > 80% in steady state
- Optimization speedup: ~9x vs naive implementation

### Regression Detection

- Performance degradation > 10% triggers alert
- Memory usage increase > 20% requires justification
- New features must include benchmark impact analysis

## Reporting Format

### Summary Report

```json
{
  "benchmark": "belief_update",
  "configuration": {
    "state_size": 100,
    "modalities": 2
  },
  "results": {
    "mean_time_ms": 8.5,
    "std_dev_ms": 1.2,
    "percentiles": {
      "p50": 8.3,
      "p95": 10.1,
      "p99": 12.5
    },
    "memory_mb": 12.3,
    "cache_hit_rate": 0.85
  },
  "comparison": {
    "vs_baseline": -15.2,
    "vs_previous": -2.1
  }
}
```

### Detailed Analysis

- Performance profiles
- Bottleneck identification
- Optimization recommendations
- Scaling projections

## Integration with CI/CD

### Automated Benchmarking

- Run on every PR to main branch
- Nightly comprehensive benchmark suite
- Performance regression gates

### Benchmark Storage

- Historical performance data in database
- Trend analysis and visualization
- Automated performance reports

## Next Steps

1. Implement base benchmark framework
2. Create specific benchmarks for each category
3. Establish baseline measurements
4. Set up CI integration
5. Create visualization dashboard
