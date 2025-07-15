# Rate Limiting Test Suite

This directory contains comprehensive tests for the rate limiting and DDoS protection system.

## Test Categories

### 1. Basic Rate Limiting Tests (`test_rate_limiting_comprehensive.py`)
- Request per second/minute/hour limits
- Burst capacity handling
- Rate limit recovery
- IP-based and user-based limiting
- Endpoint-specific limits

### 2. Security and Edge Cases (`test_rate_limiting_edge_cases.py`)
- Complex attack scenarios (Slowloris, amplification, botnets)
- Unicode and special character handling
- Time boundary conditions
- Redis failure scenarios
- Security vulnerability testing

### 3. Performance Tests (`../performance/test_rate_limiting_performance.py`)
- Throughput measurements
- Latency benchmarks
- Memory efficiency
- Scalability testing
- Redis optimization

## Prerequisites

1. **Redis Server**: Tests require a local Redis instance
   ```bash
   # Start Redis (if not already running)
   docker run -d -p 6379:6379 redis:latest
   ```

2. **Python Dependencies**:
   ```bash
   pip install -r requirements-test.txt
   ```

## Running Tests

### Run All Rate Limiting Tests
```bash
# Run all security tests including rate limiting
pytest tests/security/ -v

# Run only rate limiting tests
pytest tests/security/test_rate_limiting*.py -v
```

### Run Specific Test Categories

```bash
# Basic rate limiting tests
pytest tests/security/test_rate_limiting_comprehensive.py -v

# Edge cases and security tests
pytest tests/security/test_rate_limiting_edge_cases.py -v

# Performance tests
pytest tests/performance/test_rate_limiting_performance.py -v -m performance
```

### Run Individual Test Classes

```bash
# Test basic rate limiting
pytest tests/security/test_rate_limiting_comprehensive.py::TestBasicRateLimiting -v

# Test advanced scenarios
pytest tests/security/test_rate_limiting_comprehensive.py::TestAdvancedRateLimiting -v

# Test bypass attempts
pytest tests/security/test_rate_limiting_comprehensive.py::TestRateLimitBypassAttempts -v

# Test performance
pytest tests/performance/test_rate_limiting_performance.py::TestRateLimitingPerformance -v
```

### Run with Coverage

```bash
# Run with coverage report
pytest tests/security/test_rate_limiting*.py --cov=api.middleware.ddos_protection --cov=config.rate_limiting --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Configuration

### Environment Variables

```bash
# Redis configuration
export REDIS_TEST_URL="redis://localhost:6379/1"  # Use database 1 for tests

# Rate limiting configuration
export ENVIRONMENT="test"
export RATE_LIMITING_ENABLED="true"
export RATE_LIMITING_LOG_LEVEL="WARNING"

# Custom rate limits for testing
export RATE_LIMIT_AUTH_PER_MINUTE="5"
export RATE_LIMIT_API_PER_MINUTE="100"
```

### Pytest Markers

Tests are marked with various markers for selective execution:

```bash
# Run only performance tests
pytest -m performance

# Run only security tests
pytest -m security

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

## Performance Testing

### Generate Performance Report

```bash
# Run performance benchmark
python tests/performance/test_rate_limiting_performance.py

# This generates: rate_limiting_performance_report.json
```

### Analyze Performance Results

The performance report includes:
- Throughput (requests per second)
- Latency percentiles (p50, p95, p99)
- Memory usage
- CPU utilization
- Redis operation counts

## Debugging Failed Tests

### Enable Debug Logging

```bash
# Run with debug logging
pytest tests/security/test_rate_limiting_comprehensive.py -v -s --log-cli-level=DEBUG
```

### Check Redis State

```bash
# Connect to Redis CLI
redis-cli -n 1  # Test database

# Check rate limit keys
KEYS rate_limit:*

# Check blocked IPs
KEYS blocked:*

# Check DDoS blocks
KEYS ddos_blocked:*
```

### Common Issues

1. **Redis Connection Failed**
   - Ensure Redis is running: `redis-cli ping`
   - Check Redis URL in environment

2. **Tests Timing Out**
   - May indicate Redis performance issues
   - Check for memory/CPU constraints

3. **Flaky Tests**
   - Time-based tests may be sensitive to system load
   - Use `mock_time` fixture for deterministic testing

## Test Data

### Sample Attack Patterns

The test suite includes realistic attack simulations:
- Distributed bot networks
- Rotating proxy pools
- Slowloris attacks
- Amplification attacks
- Rate limit evasion attempts

### Security Payloads

Security tests include various malicious payloads:
- SQL/NoSQL injection attempts
- Command injection
- CRLF injection
- Buffer overflow attempts
- Unicode/encoding attacks

## Continuous Integration

### GitHub Actions Configuration

```yaml
- name: Start Redis
  run: |
    docker run -d -p 6379:6379 redis:latest
    sleep 5  # Wait for Redis to start

- name: Run Rate Limiting Tests
  run: |
    pytest tests/security/test_rate_limiting*.py -v --tb=short
    pytest tests/performance/test_rate_limiting_performance.py -v -m "not slow"
```

## Extending Tests

### Adding New Test Cases

1. **Create test in appropriate file**:
   - Basic functionality → `test_rate_limiting_comprehensive.py`
   - Security/edge cases → `test_rate_limiting_edge_cases.py`
   - Performance → `test_rate_limiting_performance.py`

2. **Use provided fixtures**:
   ```python
   async def test_new_scenario(rate_limiter, redis_client):
       # Your test implementation
   ```

3. **Follow naming conventions**:
   - `test_` prefix for all test functions
   - Descriptive names indicating what is tested

### Custom Fixtures

Add new fixtures to `conftest.py` for shared test utilities.

## Maintenance

### Regular Tasks

1. **Update attack patterns** - Keep attack simulations current
2. **Benchmark baselines** - Update performance expectations
3. **Security payloads** - Add new vulnerability patterns
4. **Redis compatibility** - Test with new Redis versions

### Performance Regression Testing

Run performance tests regularly to catch regressions:

```bash
# Save baseline
python tests/performance/test_rate_limiting_performance.py
mv rate_limiting_performance_report.json baseline_report.json

# Compare with baseline after changes
python tests/performance/test_rate_limiting_performance.py
python scripts/compare_performance_reports.py baseline_report.json rate_limiting_performance_report.json
```