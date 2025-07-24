# Enhanced CI Performance Report

**Generated**: 2025-07-15T18:04:24.802539
**Duration**: 11.77s
**Status**: FAIL
**Suite Version**: 2.0

## Performance Analysis

- **Overall Health**: poor
- **Production Readiness**: needs_attention
- **Critical Issues**: 1

### Critical Issues

- **scalability_analysis**: efficiency_at_10_agents (3.4% regression)

### Recommendations

- Address critical performance regressions before production deployment

## Benchmark Results

### comprehensive_single_agent

**Category**: agent
**Description**: Comprehensive single agent performance analysis
**Duration**: 7.86s
**Status**: PASS

**Key Metrics**:

- avg_inference_time_ms: 6.50
- memory_per_agent_mb: 0.62
- sustained_throughput_ops_sec: 164.77

**Improvements**:

- avg_inference_time_ms: 87.0% improvement
- p95_inference_time_ms: 91.0% improvement
- p99_inference_time_ms: 85.3% improvement
- memory_per_agent_mb: 98.2% improvement
- sustained_throughput_ops_sec: 723.8% improvement

### scalability_analysis

**Category**: coordination
**Description**: Comprehensive multi-agent scalability analysis
**Duration**: 3.90s
**Status**: FAIL

**Key Metrics**:

- scalability_rating: excellent

**Regressions**:

- efficiency_at_10_agents: 3.4% (critical)
