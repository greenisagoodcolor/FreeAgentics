# PyMDP Performance Benchmark Methodology

## Overview

This document describes the methodology, tools, and practices used for performance benchmarking
in the FreeAgentics project. All benchmarks focus on real PyMDP operations rather than mocked
or simulated performance data.

## Benchmark Categories

### 1. Matrix Caching Benchmarks

**Purpose**: Evaluate the effectiveness of caching transition matrices, observation likelihoods,
and intermediate computation results.

**Metrics Measured**:

- Cache hit rates and miss rates
- Memory overhead of caching
- Computation speedup factors
- Time savings from cache usage

**Test Scenarios**:

- Small models (20-25 state dimensions)
- Medium models (30-40 state dimensions)
- Large models (50+ state dimensions)
- Different cache sizes and eviction policies

### 2. Selective Update Optimizations

**Purpose**: Measure the performance impact of selective updates that avoid redundant computations.

**Metrics Measured**:

- Computation time reduction
- Percentage of operations skipped
- Accuracy maintained vs. full updates
- Memory usage optimization

**Test Scenarios**:

- Sparse observation updates (10-50% sparsity)
- Partial policy updates (20-80% changes)
- Hierarchical model propagation
- Incremental free energy calculations

### 3. Inference Benchmarking

**Purpose**: Profile core PyMDP inference algorithms across different model configurations.

**Metrics Measured**:

- Variational free energy convergence time
- Belief propagation message passing efficiency
- Policy computation latency
- Action selection performance

**Test Scenarios**:

- Different state space sizes
- Varying observation modalities
- Multiple inference iterations
- Complex factor graph structures

## Benchmark Infrastructure

### Core Components

1. **PyMDPBenchmark Base Class**: Provides standardized interface for all benchmarks
2. **BenchmarkSuite**: Manages benchmark execution and result collection
3. **BenchmarkResult**: Standardized data structure for performance metrics
4. **MemoryMonitor**: Tracks memory usage during benchmark execution
5. **PerformanceReportGenerator**: Creates analysis reports and visualizations

### Data Collection

- **Timing**: High-precision timing using `time.perf_counter()`
- **Memory**: Process memory tracking with `psutil`
- **Iterations**: Minimum 30 iterations with warmup runs
- **Statistics**: Mean, standard deviation, percentiles (50th, 90th, 95th, 99th)

### Quality Assurance

- **No Mock Data**: All benchmarks use real PyMDP operations
- **Dependency Validation**: Hard failure when PyMDP unavailable
- **Consistent Environment**: Virtual environment with pinned dependencies
- **Outlier Detection**: Statistical filtering of anomalous results

## Benchmark Execution

### Prerequisites

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Verify PyMDP installation
python -c "import pymdp; print('PyMDP available')"
```

### Running Individual Benchmarks

```bash
# Matrix caching benchmarks
python tests/performance/matrix_caching_benchmarks.py

# Selective update benchmarks
python tests/performance/selective_update_benchmarks.py

# Inference benchmarks
python tests/performance/inference_benchmarks.py
```

### Running Full Benchmark Suite

```bash
# Execute all benchmarks and generate reports
python tests/performance/performance_report_generator.py --run-all
```

## Result Interpretation

### Performance Metrics

- **Speedup Factor**: Ratio of uncached/unoptimized time to optimized time
- **Hit Rate**: Percentage of cache hits vs. total cache accesses
- **Memory Overhead**: Additional memory used by optimization techniques
- **Computation Savings**: Percentage reduction in computational operations

### Regression Detection

- **Threshold**: 10% performance degradation triggers alert
- **Severity Levels**:
  - Minor: 10-15% regression
  - Moderate: 15-25% regression
  - Severe: >25% regression

### Statistical Significance

- **Minimum Iterations**: 30 runs per benchmark
- **Confidence Level**: 95%
- **Outlier Filtering**: Remove values beyond 2 standard deviations
- **Warmup**: 10 warmup iterations before measurement

## Optimization Guidelines

### Matrix Caching

✅ **Use When**:

- Repeated access to same matrices
- Limited memory available for cache
- Computation cost > cache lookup cost

❌ **Avoid When**:

- Matrices change frequently
- Memory severely constrained
- Cache miss rate > 70%

### Selective Updates

✅ **Use When**:

- Sparse or partial state changes
- Hierarchical model structures
- Computational budget constraints

❌ **Avoid When**:

- Dense state changes (>80% modified)
- Simple linear models
- Change detection overhead > savings

## Continuous Integration

### Automated Benchmarking

- **Frequency**: Weekly automated runs
- **Regression Alerts**: Automatic notifications for >15% degradation
- **Historical Tracking**: Long-term performance trend analysis
- **Comparison Reports**: Version-to-version performance changes

### Performance Gates

- **Minimum Requirements**: No benchmark should regress >25%
- **Cache Effectiveness**: Hit rates should exceed 40%
- **Memory Efficiency**: <50MB per agent target
- **Speedup Validation**: Optimizations should show >1.2x improvement

---

_This methodology ensures reliable, reproducible, and meaningful performance measurements
for the FreeAgentics multi-agent system._
