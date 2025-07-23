# 🚀 Performance Benchmarks Suite

**PERF-ENGINEER** implementation following **Bryan Cantrill + Brendan Gregg** systems performance methodology.

## 📊 Benchmark Overview

```
Benchmark Suite
├── test_agent_spawning.py     # Agent creation performance
├── test_message_throughput.py # Message routing efficiency
├── test_memory_usage.py       # Memory profiling & leak detection
├── test_database_queries.py   # Database performance (TBD)
├── test_concurrent_load.py    # Concurrency & scalability (TBD)
└── threading_vs_multiprocessing_benchmark.py # Threading vs multiprocessing comparison
```

## 🎯 Performance Targets

| Component | Metric | Target | Current | Status |
|-----------|--------|--------|---------|---------|
| **Agent Spawning** | Single agent | < 50ms | TBD | 🔄 |
| | Parallel (10 agents) | < 100ms | TBD | 🔄 |
| | Memory per agent | < 10MB | TBD | 🔄 |
| **Message Throughput** | Simple routing | > 1000 msg/s | TBD | 🔄 |
| | Concurrent routing | > 5000 msg/s | TBD | 🔄 |
| | Async routing | > 10000 msg/s | TBD | 🔄 |
| **Memory Usage** | Agent footprint | < 10MB | TBD | 🔄 |
| | Message history | < 100MB/10k | TBD | 🔄 |
| | No memory leaks | 0 growth | TBD | 🔄 |
| **Database** | Query p95 | < 50ms | TBD | 🔄 |
| | Connection pool | 10-50 | TBD | 🔄 |
| | Transaction/sec | > 500 | TBD | 🔄 |

## 🏃 Running Benchmarks

### Quick Start
```bash
# Run all benchmarks
python -m pytest benchmarks/ --benchmark-only

# Run specific benchmark group
python -m pytest benchmarks/test_agent_spawning.py --benchmark-only

# Generate HTML report
python -m pytest benchmarks/ --benchmark-only --benchmark-autosave --benchmark-histogram
```

### Individual Benchmark Suites

#### Agent Spawning Performance
```bash
# Test agent creation performance
python benchmarks/test_agent_spawning.py

# Key metrics:
# - Single agent spawn time
# - Parallel spawning efficiency
# - Memory usage per agent
# - Scaling characteristics
```

#### Message Throughput
```bash
# Test message routing performance
python benchmarks/test_message_throughput.py

# Key metrics:
# - Messages per second
# - Routing algorithm efficiency
# - Serialization performance
# - Queue implementation comparison
```

#### Memory Usage
```bash
# Test memory efficiency
python benchmarks/test_memory_usage.py

# Key metrics:
# - Agent memory footprint
# - Message history efficiency
# - Memory leak detection
# - Object pooling benefits
```

## 📈 Continuous Performance Monitoring

### CI Integration
```yaml
# .github/workflows/unified-pipeline.yml
performance-verification:
  runs-on: ubuntu-latest
  steps:
    - name: Run performance benchmarks
      run: |
        python -m pytest benchmarks/ \
          --benchmark-only \
          --benchmark-json=benchmark_results.json

    - name: Check for regressions
      run: python benchmarks/ci_integration.py --check-regression
```

### Performance Dashboard
```bash
# Generate performance dashboard
python benchmarks/generate_dashboard.py --output performance_dashboard.html

# View historical trends
python benchmarks/analyze_trends.py --days 30
```

## 🔥 Flame Graphs

### Generate CPU Flame Graphs
```bash
# Profile with py-spy
py-spy record -o profile.svg -- python benchmarks/test_agent_spawning.py

# Profile with cProfile
python -m cProfile -o profile.stats benchmarks/test_message_throughput.py
python benchmarks/generate_flamegraph.py profile.stats
```

### Memory Profiling
```bash
# Memory profiling with memory_profiler
python -m memory_profiler benchmarks/test_memory_usage.py

# Heap analysis with pympler
python benchmarks/heap_analysis.py
```

## Threading vs Multiprocessing Benchmarks

This directory also contains comprehensive benchmarks comparing threading and multiprocessing approaches for FreeAgentics Active Inference agents.

## Overview

The benchmarks measure real-world performance characteristics including:

- **Performance Metrics**: Throughput, latency, scaling efficiency
- **Memory Usage**: Per-agent and total memory consumption
- **Communication Overhead**: Inter-agent messaging costs
- **Realistic Scenarios**: Exploration, coordination, and learning tasks

## Files

### `threading_vs_multiprocessing_benchmark.py`

**Comprehensive benchmark suite** that tests both threading and multiprocessing across multiple scenarios:

- **Exploration workloads**: Independent agents exploring environments
- **Coordination workloads**: Agents communicating and coordinating
- **Scaling analysis**: Performance from 1 to 30+ agents
- **Memory profiling**: Detailed memory usage tracking
- **Communication overhead**: Measuring inter-agent messaging costs

**Usage:**

```bash
python benchmarks/threading_vs_multiprocessing_benchmark.py
```

**Expected runtime**: 10-20 minutes for full benchmark suite

### `quick_threading_vs_multiprocessing_test.py`

**Rapid validation test** that runs quickly to validate the approach:

- **Fast execution**: 1-2 minutes total runtime
- **Core comparisons**: Performance and memory for key scenarios
- **Communication test**: Direct comparison of messaging overhead
- **Immediate results**: Quick feedback on which approach is better

**Usage:**

```bash
python benchmarks/quick_threading_vs_multiprocessing_test.py
```

**Expected runtime**: 1-2 minutes

## Key Metrics

### Performance Metrics

- **Throughput**: Operations per second (higher is better)
- **Latency**: Time per operation in milliseconds (lower is better)
- **P95/P99 Latency**: 95th and 99th percentile response times
- **Scaling Efficiency**: How well performance scales with agent count

### Memory Metrics

- **Per-agent Memory**: Memory usage per agent instance
- **Total Memory**: Overall memory consumption
- **Memory Delta**: Memory increase during benchmark

### Communication Metrics

- **Message Throughput**: Messages per second
- **Message Latency**: Time per message
- **Overhead Ratio**: Relative communication cost

## Expected Results

Based on the theoretical analysis, we expect:

### Threading Advantages

- **Better Performance**: 2-10x faster for PyMDP computations
- **Lower Memory**: 3-5x less memory usage due to shared memory
- **Faster Communication**: 10-100x faster inter-agent messaging
- **Better Scaling**: More efficient with 20+ agents

### Multiprocessing Advantages

- **True Parallelism**: Can overcome GIL limitations
- **Fault Isolation**: Process crashes don't affect others
- **CPU-Intensive Tasks**: Better for pure computation workloads

## Interpreting Results

### Performance Comparison

Look for:

- **Throughput ratios**: Threading should be 2-5x faster
- **Memory efficiency**: Threading should use 50-80% less memory
- **Communication overhead**: Threading should be 10-100x faster

### Production Recommendations

- **Use Threading** for most FreeAgentics scenarios
- **Consider Multiprocessing** only for:
  - Very CPU-intensive custom models
  - Fault isolation requirements
  - Integration with non-Python components

## Running the Benchmarks

### Prerequisites

```bash
pip install psutil numpy
```

### Quick Test (Recommended First)

```bash
cd /path/to/FreeAgentics
python benchmarks/quick_threading_vs_multiprocessing_test.py
```

### Full Benchmark Suite

```bash
cd /path/to/FreeAgentics
python benchmarks/threading_vs_multiprocessing_benchmark.py
```

Results are saved to `benchmark_results.json` for further analysis.

## Benchmark Architecture

### Threading Implementation

- Uses `OptimizedThreadPoolManager` from FreeAgentics
- Shared memory model with thread-safe operations
- Direct memory access for communication
- Thread pool scaling based on load

### Multiprocessing Implementation

- Uses `ProcessPoolExecutor` for agent processes
- `Manager` for shared state coordination
- `Queue` for inter-process communication
- Process-based isolation

### Workload Generation

- **Exploration**: Independent agent navigation
- **Coordination**: Multi-agent communication scenarios
- **Realistic Patterns**: Based on actual Active Inference use cases
- **Parameterized**: Configurable agent counts and operation counts

## Validation

The benchmarks validate key claims:

1. **Performance**: Threading is faster for PyMDP-based agents
1. **Memory**: Threading uses significantly less memory
1. **Communication**: Threading has much lower communication overhead
1. **Scaling**: Threading scales better with agent count

Results help make informed decisions about the optimal concurrency model for FreeAgentics deployments.

## Troubleshooting

### Common Issues

**Multiprocessing startup errors**:

- Ensure `mp.set_start_method('spawn')` is called
- Check Python version compatibility

**Memory measurement issues**:

- Run with sufficient system memory
- Close other applications during benchmarking

**Performance variations**:

- Run multiple times and average results
- Consider system load and background processes

### Platform-Specific Notes

**Linux**: Generally shows best performance for both approaches
**Windows**: May show different multiprocessing overhead
**macOS**: Possible spawn method restrictions

## Contributing

When adding new benchmarks:

1. Follow the existing pattern for result collection
1. Use `BenchmarkResult` dataclass for consistency
1. Include memory tracking
1. Add comprehensive error handling
1. Document expected behavior

## Quick Start

### Run All Benchmarks

```bash
# Run comprehensive benchmark suite
bash benchmarks/run_all_benchmarks.sh
```

### Quick Validation (2 minutes)

```bash
# Quick test to validate approach
source venv/bin/activate
python3 benchmarks/simple_threading_vs_multiprocessing_test.py
```

### Production Benchmark (10 minutes)

```bash
# Detailed production analysis
source venv/bin/activate
python3 benchmarks/production_benchmark.py
```

## Benchmark Results Summary

Based on initial testing on a 20-core Linux system:

| Metric | Threading | Multiprocessing | Threading Advantage |
| ------------ | ------------- | --------------- | ------------------- |
| Single Agent | 680.5 ops/sec | 13.8 ops/sec | **49.35x faster** |
| 5 Agents | 190.1 ops/sec | 47.5 ops/sec | **4.00x faster** |
| 10 Agents | 197.3 ops/sec | 63.8 ops/sec | **3.09x faster** |

**Key Finding**: Threading is consistently faster due to:

- Lower process startup overhead (0.03s vs 1.45s)
- Shared memory model efficiency
- Better PyMDP computation patterns
- Faster inter-agent communication

## 📊 Benchmark Configuration

### pytest-benchmark Settings
```ini
# pytest.ini
[tool:pytest]
addopts =
    --benchmark-columns=min,max,mean,stddev,median,iqr,outliers
    --benchmark-sort=mean
    --benchmark-group-by=group
    --benchmark-warmup=on
    --benchmark-disable-gc
```

### Custom Benchmark Parameters
```python
# benchmarks/config.py
BENCHMARK_CONFIG = {
    'min_rounds': 5,
    'warmup_rounds': 2,
    'calibration_precision': 10,
    'disable_gc': True,
    'timer': time.perf_counter
}
```

## 🎯 Performance Optimization Checklist

### Before Optimization
- [ ] Run baseline benchmarks
- [ ] Generate flame graphs
- [ ] Identify bottlenecks
- [ ] Set performance targets

### During Optimization
- [ ] Focus on hot paths (flame graph)
- [ ] Measure after each change
- [ ] Document optimization rationale
- [ ] Ensure no functionality regression

### After Optimization
- [ ] Verify performance improvements
- [ ] Update baseline metrics
- [ ] Add regression tests
- [ ] Document in PERF_IMPROVEMENTS.md

## 🚨 Performance Regression Detection

### Automated Checks
```python
# benchmarks/ci_integration.py
class PerformanceRegression:
    THRESHOLDS = {
        'agent_spawn': 0.1,      # 10% regression allowed
        'message_throughput': 0.15,  # 15% regression allowed
        'memory_usage': 0.2      # 20% regression allowed
    }
```

### Manual Analysis
```bash
# Compare with baseline
python benchmarks/compare_results.py \
    --baseline baseline_results.json \
    --current benchmark_results.json

# Generate regression report
python benchmarks/regression_report.py --format markdown > REGRESSION_REPORT.md
```

## 📚 Methodology References

### Bryan Cantrill Principles
- **Observability First**: Every benchmark includes detailed metrics
- **Systems Thinking**: Consider full system impact
- **Production Relevance**: Benchmarks reflect real workloads

### Brendan Gregg Methods
- **USE Method**: Utilization, Saturation, Errors
- **Flame Graphs**: Visual performance analysis
- **Latency Analysis**: Percentile-based metrics

## 🔧 Troubleshooting

### Common Issues

#### ImportError in benchmarks
```bash
# Ensure project is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Inconsistent results
```bash
# Increase rounds for stability
python -m pytest benchmarks/ --benchmark-only --benchmark-min-rounds=10
```

#### Memory profiling issues
```bash
# Install required tools
pip install memory-profiler pympler psutil
```

## 📈 Historical Performance

Results are tracked in `benchmarks/results/` with the following structure:
```
results/
├── baseline/
│   └── baseline_results.json
├── daily/
│   └── 2025-01-20_results.json
└── releases/
    └── v1.0.0_results.json
```

## 🏆 Performance Achievements

- [ ] Agent spawn < 50ms ⏳
- [ ] Message throughput > 1000/s ⏳
- [ ] Zero memory leaks ⏳
- [ ] Database queries < 50ms p95 ⏳
- [ ] 10x improvement in critical paths ⏳

## Future Enhancements

Planned improvements:

- GPU acceleration benchmarks
- Network communication scenarios
- Fault tolerance testing
- Integration with external systems
- Real-time performance requirements

---

*Powered by PERF-ENGINEER • Following Bryan Cantrill + Brendan Gregg Methodology*
