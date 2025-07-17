# Performance Benchmarks

This directory contains a comprehensive performance benchmarking suite for the FreeAgentics project, including automated CI/CD integration, regression detection, and performance monitoring.

## Overview

The performance benchmark suite provides:

- **Comprehensive Benchmarks**: Agent spawn time, message throughput, memory usage, database queries, and WebSocket connections
- **Automated CI/CD Integration**: GitHub Actions workflow with automatic regression detection
- **Performance Regression Detection**: Fails CI if performance degrades by >10%
- **Historical Tracking**: Performance trends over time with 90-day retention
- **Performance Dashboards**: Visual reports and trend analysis
- **Continuous Profiling**: CPU and memory profiling on demand

## Quick Start

### Running Benchmarks Locally

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run all benchmarks
python -m pytest benchmarks/performance_suite.py -v --benchmark-only

# Run specific benchmark category
python -m pytest benchmarks/performance_suite.py::AgentSpawnBenchmarks -v --benchmark-only

# Save benchmark results
python -m pytest benchmarks/performance_suite.py -v \
  --benchmark-json=results/benchmark_results.json \
  --benchmark-save=my_benchmark
```

### CI Integration

```bash
# Check for regressions against baseline
python benchmarks/ci_integration.py \
  --results-file results/benchmark_results.json \
  --baseline-dir baselines \
  --fail-on-regression

# Update baseline (only if no regressions)
python benchmarks/ci_integration.py \
  --results-file results/benchmark_results.json \
  --baseline-dir baselines \
  --update-baseline
```

## Benchmark Categories

### 1. Agent Spawn Benchmarks
- Single agent initialization
- Batch agent spawning
- Concurrent agent creation

### 2. Message Throughput Benchmarks
- Single message passing
- Bulk message processing
- Async message handling

### 3. Memory Usage Benchmarks
- Agent memory lifecycle
- Belief state compression
- Matrix pooling efficiency

### 4. Database Query Benchmarks
- Single query performance
- Batch query optimization
- Connection pool efficiency

### 5. WebSocket Connection Benchmarks
- Connection setup time
- Concurrent connection handling
- Message broadcasting

## Performance Thresholds

- **Critical Regression**: >10% performance degradation (fails CI)
- **Warning**: >5% performance degradation (triggers warning)
- **Improvement**: >5% performance improvement (noted in reports)

## CI/CD Integration

The GitHub Actions workflow runs on:
- Every push to main/develop branches
- Every pull request
- Daily scheduled runs (2 AM UTC)
- Manual workflow dispatch

### Workflow Features

1. **Automatic Regression Detection**: Compares against baseline and fails on critical regressions
2. **PR Comments**: Automatically comments on PRs with performance impact
3. **Baseline Updates**: Updates baseline on main branch if no regressions
4. **Performance Dashboard**: Generates visual reports for scheduled runs
5. **Profiling**: Optional detailed profiling with `[profile]` in commit message

## File Structure

```
benchmarks/
├── performance_suite.py          # Main benchmark suite using pytest-benchmark
├── ci_integration.py            # CI/CD integration and regression detection
├── baselines/                   # Performance baseline storage
│   ├── performance_baseline.json
│   └── performance_history.json
├── results/                     # Benchmark results
│   └── *.json
└── ci_results/                  # CI integration reports
    ├── regression_report_*.json
    ├── performance_trends.json
    └── github_comment.md
```

## Best Practices

1. **Consistent Environment**: Run benchmarks in consistent environment (CI uses Ubuntu latest)
2. **Warm-up Runs**: Benchmarks include warm-up iterations for stable results
3. **Statistical Analysis**: Uses mean, median, and standard deviation for accuracy
4. **Isolation**: Each benchmark runs in isolation to avoid interference
5. **Regular Updates**: Update baselines periodically as performance improves

## Extending Benchmarks

To add new benchmarks:

1. Create a new class in `performance_suite.py`:
```python
class MyNewBenchmarks:
    @staticmethod
    def benchmark_my_feature(benchmark):
        def my_operation():
            # Your code here
            pass
        result = benchmark(my_operation)
        assert result is not None
```

2. Add to appropriate category in `BENCHMARK_CATEGORIES`

3. Update baseline after verification:
```bash
python benchmarks/ci_integration.py \
  --results-file results/benchmark_results.json \
  --update-baseline --force-baseline
```

## Troubleshooting

### Common Issues

1. **Flaky Benchmarks**: Increase `--benchmark-min-rounds` for more stable results
2. **Memory Leaks**: Use `--benchmark-disable-gc` to detect GC-related issues
3. **Regression False Positives**: Check system load and adjust thresholds if needed

### Debug Commands

```bash
# Verbose output
pytest benchmarks/performance_suite.py -vv --benchmark-verbose

# Compare with previous results
pytest benchmarks/performance_suite.py --benchmark-compare

# Generate histogram
pytest benchmarks/performance_suite.py --benchmark-histogram
```

## Performance Goals

Based on production requirements:
- **Agent Spawn Time**: <50ms per agent
- **Message Throughput**: >1000 messages/second
- **Memory per Agent**: <35MB
- **Database Query**: <10ms average
- **WebSocket Setup**: <100ms

## Contributing

When submitting performance improvements:
1. Run benchmarks before and after changes
2. Include benchmark results in PR description
3. Explain optimization techniques used
4. Update this README if adding new benchmarks