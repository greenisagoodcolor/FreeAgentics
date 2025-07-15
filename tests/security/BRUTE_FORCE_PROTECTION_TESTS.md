# Brute Force Protection Test Suite

## Overview

This comprehensive test suite validates the robustness of brute force protection mechanisms across the FreeAgentics platform. The tests cover authentication attacks, token manipulation, resource enumeration, and sophisticated attack patterns while ensuring minimal impact on legitimate users.

## Test Coverage

### 1. Authentication Brute Force Protection (`test_brute_force_protection.py`)

#### Login Attack Protection
- **Rate Limiting**: Validates request throttling after threshold
- **Password Brute Force**: Tests protection against common password attacks
- **Account Lockout**: Ensures accounts lock after excessive failures
- **Progressive Delays**: Verifies increasing delays with repeated failures

#### Token Security
- **JWT Brute Forcing**: Prevents token manipulation attempts
- **API Key Enumeration**: Blocks systematic API key guessing
- **Session Token Guessing**: Protects against session hijacking
- **Refresh Token Attacks**: Prevents token reuse and manipulation

#### Resource Enumeration Defense
- **Directory Brute Forcing**: Blocks directory discovery attempts
- **File Enumeration**: Prevents sensitive file discovery
- **API Endpoint Discovery**: Protects hidden endpoints
- **Parameter Fuzzing**: Blocks parameter manipulation attacks

### 2. Advanced Attack Scenarios (`test_brute_force_advanced_scenarios.py`)

#### Timing Attack Prevention
- **User Enumeration Protection**: Constant-time responses prevent user discovery
- **Password Length Timing**: No correlation between password length and response time
- **Hash Computation Protection**: Consistent timing regardless of password complexity

#### Distributed Attack Defense
- **Botnet Simulation**: Detects coordinated attacks from multiple IPs
- **Rotating Proxy Detection**: Identifies proxy rotation patterns
- **Coordinated Timing Attacks**: Recognizes synchronized attack waves

#### Sophisticated Protection
- **Credential Spray Detection**: Identifies password spray patterns
- **Lockout Evasion Detection**: Catches attempts to bypass lockouts
- **Zero-Day Pattern Learning**: Adapts to novel attack patterns
- **Multi-Vector Attack Defense**: Protects against combined attack strategies

### 3. Performance Benchmarks (`benchmark_brute_force_protection.py`)

#### Impact Analysis
- **Baseline Performance**: Measures normal operation metrics
- **Light Attack Impact**: Tests performance under minimal attack
- **Heavy Attack Resilience**: Validates stability under intense attacks
- **Resource Consumption**: Monitors CPU and memory usage

## Running the Tests

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Ensure Redis is running
docker run -d -p 6379:6379 redis:latest

# Start the application
python main.py
```

### Run All Security Tests

```bash
# Run all brute force protection tests
pytest tests/security/test_brute_force_protection.py -v

# Run advanced scenario tests
pytest tests/security/test_brute_force_advanced_scenarios.py -v

# Run performance benchmarks
python tests/security/benchmark_brute_force_protection.py
```

### Run Specific Test Categories

```bash
# Authentication tests only
pytest tests/security/test_brute_force_protection.py::TestAuthenticationBruteForce -v

# Token security tests
pytest tests/security/test_brute_force_protection.py::TestTokenBruteForce -v

# Timing attack tests
pytest tests/security/test_brute_force_advanced_scenarios.py::TestTimingAttackPrevention -v

# Performance impact tests
pytest tests/security/test_brute_force_protection.py::TestPerformanceImpact -v
```

## Key Protection Mechanisms

### 1. Rate Limiting Configuration

```python
# Production settings (auth endpoints)
RateLimitConfig(
    requests_per_minute=3,      # Very strict for auth
    requests_per_hour=50,       # Cumulative limit
    burst_limit=2,              # Max burst size
    block_duration=900,         # 15 minute block
    ddos_threshold=500,         # DDoS detection
    ddos_block_duration=7200    # 2 hour DDoS block
)
```

### 2. Progressive Protection

- **Initial Attempts**: Standard rate limiting
- **Repeated Failures**: Progressive delays (exponential backoff)
- **Persistent Attacks**: IP blocking and account protection
- **Distributed Attacks**: Pattern recognition and global blocking

### 3. Intelligent Detection

- **Pattern Recognition**: Identifies common attack patterns
- **Behavioral Analysis**: Detects anomalous request patterns
- **Reputation Scoring**: Tracks IP and user reputation
- **Adaptive Thresholds**: Adjusts limits based on attack intensity

## Test Assertions

### Security Requirements

1. **Authentication Protection**
   - Max 3 login attempts per minute per IP
   - Account lockout after 5 failed attempts
   - Progressive delays increase with failures
   - Lockout persists even with correct password

2. **Token Security**
   - Block token manipulation attempts
   - Prevent timing-based token discovery
   - Invalidate tokens after suspicious activity
   - Rate limit token refresh endpoints

3. **Resource Protection**
   - Block directory/file enumeration
   - Hide existence of protected resources
   - Consistent error responses (no information leakage)
   - Rate limit all discovery attempts

### Performance Requirements

1. **Legitimate User Impact**
   - Success rate > 95% under attack
   - Average latency increase < 2x
   - P99 latency < 1 second
   - No false positives for normal usage

2. **Attack Effectiveness**
   - Block rate > 90% for attacks
   - Detection time < 1 minute
   - Distributed attack recognition
   - Adaptive response to new patterns

3. **Resource Efficiency**
   - Memory increase < 100MB under attack
   - CPU usage < 80% during attacks
   - Redis operations optimized
   - Efficient pattern matching

## Monitoring and Alerts

### Security Events Logged

```python
SecurityEventType.RATE_LIMIT_EXCEEDED     # Rate limit violations
SecurityEventType.DDOS_ATTACK            # DDoS detection
SecurityEventType.BRUTE_FORCE_ATTEMPT    # Brute force patterns
SecurityEventType.ACCOUNT_LOCKOUT        # Account protection triggered
```

### Metrics Tracked

- Request rates per IP/user
- Failed authentication attempts
- Token manipulation attempts
- Resource enumeration attempts
- Protection effectiveness rates
- Performance impact measurements

## Best Practices

### For Developers

1. **Consistent Timing**: Use constant-time operations for security checks
2. **Generic Errors**: Return identical errors for different failure modes
3. **Rate Limit Early**: Apply limits before expensive operations
4. **Log Everything**: Record all suspicious activity for analysis

### For Operations

1. **Monitor Metrics**: Watch for attack patterns in real-time
2. **Tune Thresholds**: Adjust limits based on legitimate usage patterns
3. **Update Blocklists**: Maintain IP and pattern blocklists
4. **Test Regularly**: Run protection tests in staging environments

## Troubleshooting

### Common Issues

1. **High False Positive Rate**
   - Review legitimate usage patterns
   - Adjust rate limits for authenticated users
   - Implement reputation-based filtering

2. **Performance Degradation**
   - Check Redis connection pooling
   - Optimize pattern matching algorithms
   - Consider caching for expensive checks

3. **Sophisticated Attack Bypass**
   - Update detection patterns
   - Implement machine learning models
   - Add behavioral analysis

## Future Enhancements

1. **Machine Learning Integration**
   - Anomaly detection models
   - Predictive attack prevention
   - Automated pattern learning

2. **Enhanced Protection**
   - CAPTCHA integration
   - Multi-factor authentication triggers
   - Geolocation-based filtering

3. **Performance Optimizations**
   - Edge-based rate limiting
   - Distributed caching
   - Hardware acceleration

## Compliance

The brute force protection implementation meets or exceeds requirements from:

- OWASP Top 10 (A07:2021 - Identification and Authentication Failures)
- NIST 800-63B (Authentication and Lifecycle Management)
- PCI DSS 3.2.1 (Requirement 8.2.4 - Account Lockout)
- ISO 27001:2013 (A.9.4.2 - Secure log-on procedures)