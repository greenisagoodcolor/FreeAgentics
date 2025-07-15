# Enhanced CI Performance Report

**Generated**: 2025-07-15T15:33:45.453294
**Duration**: 11.38s
**Status**: FAIL
**Suite Version**: 2.0

## Performance Analysis

- **Overall Health**: poor
- **Production Readiness**: needs_attention
- **Critical Issues**: 1

### Critical Issues

- **scalability_analysis**: efficiency_at_10_agents (2.9% regression)

### Recommendations

- Address critical performance regressions before production deployment

## Benchmark Results

### comprehensive_single_agent

**Category**: agent
**Description**: Comprehensive single agent performance analysis
**Duration**: 7.59s
**Status**: PASS

**Key Metrics**:
- avg_inference_time_ms: 5.47
- memory_per_agent_mb: 0.47
- sustained_throughput_ops_sec: 176.19

**Improvements**:
- avg_inference_time_ms: 89.1% improvement
- p95_inference_time_ms: 92.9% improvement
- p99_inference_time_ms: 94.6% improvement
- memory_per_agent_mb: 98.6% improvement
- sustained_throughput_ops_sec: 780.9% improvement

### scalability_analysis

**Category**: coordination
**Description**: Comprehensive multi-agent scalability analysis
**Duration**: 3.79s
**Status**: FAIL

**Key Metrics**:
- scalability_rating: excellent

**Regressions**:
- efficiency_at_10_agents: 2.9% (critical)

