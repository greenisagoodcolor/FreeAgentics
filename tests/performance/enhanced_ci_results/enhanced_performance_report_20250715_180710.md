# Enhanced CI Performance Report

**Generated**: 2025-07-15T18:06:59.618735
**Duration**: 11.20s
**Status**: FAIL
**Suite Version**: 2.0

## Performance Analysis

- **Overall Health**: poor
- **Production Readiness**: needs_attention
- **Critical Issues**: 1

### Critical Issues

- **scalability_analysis**: efficiency_at_10_agents (3.2% regression)

### Recommendations

- Address critical performance regressions before production deployment

## Benchmark Results

### comprehensive_single_agent

**Category**: agent
**Description**: Comprehensive single agent performance analysis
**Duration**: 7.63s
**Status**: PASS

**Key Metrics**:
- avg_inference_time_ms: 5.60
- memory_per_agent_mb: 0.47
- sustained_throughput_ops_sec: 177.64

**Improvements**:
- avg_inference_time_ms: 88.8% improvement
- p95_inference_time_ms: 92.5% improvement
- p99_inference_time_ms: 93.2% improvement
- memory_per_agent_mb: 98.6% improvement
- sustained_throughput_ops_sec: 788.2% improvement

### scalability_analysis

**Category**: coordination
**Description**: Comprehensive multi-agent scalability analysis
**Duration**: 3.57s
**Status**: FAIL

**Key Metrics**:
- scalability_rating: excellent

**Regressions**:
- efficiency_at_10_agents: 3.2% (critical)

