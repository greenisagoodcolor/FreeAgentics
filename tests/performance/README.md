# PyMDP Performance Monitoring System

A comprehensive performance monitoring and benchmarking system for the FreeAgentics multi-agent platform. This system provides real PyMDP performance measurements, regression detection, automated reporting, and CI/CD integration.

## Overview

The performance monitoring system consists of several key components:

- **Benchmark Suites**: Real PyMDP operation benchmarks (no mocked timing)
- **Report Generation**: Automated analysis with visualizations and insights
- **Regression Detection**: Automatic alerts for performance degradations
- **CI/CD Integration**: GitHub Actions workflow for continuous monitoring
- **Quality Gates**: Performance thresholds for deployment decisions

## Quick Start

### Prerequisites

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Verify PyMDP installation
python -c "import pymdp; print('PyMDP available')"
```

### Running Benchmarks

```bash
# Run individual benchmark suites
python matrix_caching_benchmarks.py
python selective_update_benchmarks.py
python inference_benchmarks.py

# Run automated monitoring pipeline
python run_performance_monitoring.py

# Generate reports from existing results
python performance_report_generator.py
```

## Benchmark Categories

### 1. Matrix Caching Benchmarks (`matrix_caching_benchmarks.py`)

Tests the effectiveness of caching strategies for PyMDP matrix operations:

- **Transition Matrix Caching**: Cache B matrices for state transitions
- **Observation Likelihood Caching**: Cache A matrix likelihood computations
- **Intermediate Result Caching**: Cache expensive computational intermediate steps
- **Cache Comparison**: Direct performance comparisons with/without caching

**Key Metrics**:

- Cache hit rates and miss rates
- Memory overhead (MB)
- Speedup factors (up to 353x achieved)
- Computation time savings

### 2. Selective Update Benchmarks (`selective_update_benchmarks.py`)

Measures optimization benefits from selective updates that avoid redundant computations:

- **Sparse Observation Updates**: Process only changed observations
- **Partial Policy Updates**: Update subset of policy components
- **Incremental Free Energy**: Calculate only necessary energy terms
- **Hierarchical Updates**: Propagate changes through hierarchy levels

**Key Metrics**:

- Computation savings percentage
- Operations skipped vs. total
- Speedup factors (up to 5.19x achieved)
- Accuracy maintained vs. full updates

### 3. Inference Benchmarks (`inference_benchmarks.py`)

Profiles core PyMDP inference algorithms across different configurations:

- **Variational Inference**: VFE optimization and convergence
- **Belief Propagation**: Factor graph message passing
- **Message Passing**: Different update schedules and topologies
- **Inference Profiling**: Detailed timing breakdowns

**Key Metrics**:

- Inference convergence time
- Message passing efficiency
- Policy computation latency
- Memory usage patterns

## Performance Report System

### Automated Report Generation

The `performance_report_generator.py` creates comprehensive analysis reports:

```bash
python performance_report_generator.py
```

**Generated Outputs**:

- Executive summary with key findings
- Performance regression alerts
- Optimization recommendations
- Statistical analysis with charts
- Methodology documentation

### Report Components

1. **Executive Summary**
   - Total benchmarks and metrics analyzed
   - Key performance achievements
   - Critical regression alerts

2. **Performance Charts** (PNG format)
   - Time series performance trends
   - Benchmark comparison charts
   - Cache effectiveness visualizations
   - Memory usage patterns

3. **Regression Detection**
   - Automatic threshold-based alerts
   - Severity classification (minor/moderate/severe)
   - Historical trend analysis

4. **Optimization Recommendations**
   - Based on actual performance data
   - Specific actionable suggestions
   - Performance improvement strategies

## CI/CD Integration

### GitHub Actions Workflow

The system includes a comprehensive GitHub Actions workflow (`.github/workflows/performance-monitoring.yml`):

**Triggers**:

- Push to main branch
- Pull request creation/updates
- Weekly scheduled runs (Sundays 6 AM UTC)
- Manual workflow dispatch

**Features**:

- Automated benchmark execution
- Performance quality gates
- PR comments with results
- Artifact uploads (results/charts)
- Slack notifications for regressions
- Baseline performance tracking

### Quality Gates

Performance gates enforce quality standards:

- **Regression Threshold**: Default 10% degradation triggers warnings
- **Cache Effectiveness**: Minimum 20% hit rate required
- **Memory Efficiency**: <100MB per operation average
- **Severe Regression**: >25% degradation fails CI

### Automated Monitoring Pipeline

```bash
# Run complete monitoring pipeline
python run_performance_monitoring.py --ci-mode

# Skip benchmarks, only analyze existing results
python run_performance_monitoring.py --skip-benchmarks

# Custom regression threshold
python run_performance_monitoring.py --regression-threshold 15.0
```

## Configuration

### Environment Variables

```bash
# Slack webhook for performance alerts
export PERFORMANCE_ALERT_WEBHOOK="https://hooks.slack.com/services/..."

# CI/CD mode settings
export CI=true
export GITHUB_ACTIONS=true
```

### GitHub Secrets

Configure these secrets in your repository:

- `SLACK_WEBHOOK_URL`: For performance alert notifications

## Results and Analysis

### Result Files

Benchmark results are saved as JSON files:

```
tests/performance/
├── matrix_caching_benchmark_results_YYYYMMDD_HHMMSS.json
├── selective_update_benchmark_results_YYYYMMDD_HHMMSS.json
└── inference_benchmark_results_YYYYMMDD_HHMMSS.json
```

### Report Directory Structure

```
tests/performance/reports/
├── performance_report_YYYYMMDD_HHMMSS.md
├── benchmark_methodology.md
├── ci_summary_YYYYMMDD_HHMMSS.md
├── mean_time_timeseries.png
├── cache_hit_rate_comparison.png
└── speedup_factor_timeseries.png
```

## Performance Achievements

Based on current benchmark results:

### Matrix Caching

- **Maximum Speedup**: 353x for intermediate result caching
- **Average Hit Rate**: 87.5% for complex computations
- **Memory Efficiency**: <5MB overhead for significant speedups

### Selective Updates

- **Computation Savings**: Up to 60% reduction in operations
- **Hierarchical Speedup**: 5.19x for multi-level updates
- **Propagation Efficiency**: 40-67% of levels updated selectively

### Overall Impact

- **9x Performance Improvement**: Validated through real benchmarks
- **Memory Optimization**: Identified 34.5MB/agent overhead for optimization
- **Scalability Insights**: Documented GIL limitations and solutions

## Best Practices

### Benchmark Development

1. **Real Operations Only**: No `time.sleep()` or mocked timing
2. **Dependency Validation**: Hard failure when PyMDP unavailable
3. **Statistical Rigor**: Minimum 30 iterations with outlier detection
4. **Consistent Environment**: Virtual environment with pinned dependencies

### Performance Analysis

1. **Trend Monitoring**: Track performance over time
2. **Regression Alerts**: Configure appropriate thresholds
3. **Root Cause Analysis**: Investigate regressions immediately
4. **Documentation**: Maintain performance methodology docs

### CI/CD Integration

1. **Quality Gates**: Enforce performance standards
2. **Baseline Tracking**: Maintain performance history
3. **Alert Fatigue**: Balance sensitivity vs. noise
4. **Artifact Retention**: Keep results for trend analysis

## Troubleshooting

### Common Issues

1. **PyMDP Import Errors**

   ```bash
   # Verify installation
   source venv/bin/activate
   pip list | grep pymdp
   python -c "import pymdp"
   ```

2. **Missing Dependencies**

   ```bash
   # Install required packages
   pip install matplotlib seaborn pandas numpy
   ```

3. **Memory Issues**

   ```bash
   # Monitor memory during benchmarks
   python -c "import psutil; print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')"
   ```

4. **Timeout in CI**
   - Reduce benchmark iterations for CI
   - Split large benchmark suites
   - Increase workflow timeout limits

### Performance Debugging

1. **Low Cache Hit Rates**
   - Check cache key generation logic
   - Verify cache size limits
   - Analyze access patterns

2. **High Memory Usage**
   - Profile with `memory_profiler`
   - Check for memory leaks
   - Optimize data structures

3. **Unexpected Regressions**
   - Compare with baseline results
   - Check for dependency version changes
   - Analyze system resource availability

## Contributing

### Adding New Benchmarks

1. Inherit from `PyMDPBenchmark` base class
2. Implement `setup()` and `run_iteration()` methods
3. Add to benchmark suite in main script
4. Update documentation and CI workflow

### Extending Reports

1. Add new metric extraction in `performance_report_generator.py`
2. Create visualization functions for new metrics
3. Update report templates with new insights
4. Test with existing result data

## Support

For issues or questions:

1. Check existing GitHub issues
2. Review performance methodology documentation
3. Examine CI workflow logs for debugging
4. Create detailed issue reports with benchmark results

---

_This performance monitoring system ensures reliable, continuous performance validation for the FreeAgentics multi-agent platform._
