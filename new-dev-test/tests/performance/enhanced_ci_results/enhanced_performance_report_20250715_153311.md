# Enhanced CI Performance Report

**Generated**: 2025-07-15T15:32:59.781016
**Duration**: 11.40s
**Status**: FAIL
**Suite Version**: 2.0

## Performance Analysis

- **Overall Health**: poor
- **Production Readiness**: needs_attention
- **Critical Issues**: 1

### Critical Issues

- **scalability_analysis**: efficiency_at_10_agents (4.2% regression)

### Recommendations

- Address critical performance regressions before production deployment

## Benchmark Results

### comprehensive_single_agent

**Category**: agent
**Description**: Comprehensive single agent performance analysis
**Duration**: 7.80s
**Status**: PASS

**Key Metrics**:

- avg_inference_time_ms: 6.20
- memory_per_agent_mb: 0.62
- sustained_throughput_ops_sec: 161.22

**Improvements**:

- avg_inference_time_ms: 87.6% improvement
- p95_inference_time_ms: 92.1% improvement
- p99_inference_time_ms: 94.2% improvement
- memory_per_agent_mb: 98.2% improvement
- sustained_throughput_ops_sec: 706.1% improvement

### scalability_analysis

**Category**: coordination
**Description**: Comprehensive multi-agent scalability analysis
**Duration**: 3.60s
**Status**: FAIL

**Key Metrics**:

- scalability_rating: moderate

**Regressions**:
