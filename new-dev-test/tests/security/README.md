# FreeAgentics Security Testing Suite

## Error Handling Information Disclosure Testing

This directory contains comprehensive security tests for validating error handling and preventing information disclosure vulnerabilities in the FreeAgentics platform.

## Test Coverage

### 1. Error Handling Information Disclosure (`test_error_handling_information_disclosure.py`)

- **Database Error Information Leakage**: Tests for SQL error exposure, connection details, and database implementation disclosure
- **Stack Trace Exposure Detection**: Validates that stack traces and internal code paths are not exposed
- **Debug Information Disclosure**: Checks for debug flags, development mode indicators, and verbose logging
- **Internal Path Revelation**: Tests for file system path disclosure and sensitive directory exposure
- **Version Information Leakage**: Validates that software versions and implementation details are hidden

### 2. Authentication Error Disclosure (`test_authentication_error_disclosure.py`)

- **Username Enumeration Attacks**: Tests timing and response differences that could reveal valid usernames
- **Password Policy Disclosure**: Ensures password requirements aren't exposed in error messages
- **Account Lockout Information Leakage**: Validates that lockout details aren't disclosed
- **Session Management Error Disclosure**: Tests session handling error responses
- **JWT Error Information Leakage**: Checks for JWT implementation details in error responses
- **Authentication Timing Attacks**: Validates consistent response timing across scenarios

### 3. API Security Response Validation (`test_api_security_responses.py`)

- **Security Headers Validation**: Ensures all required security headers are present and properly configured
- **Rate Limiting Response Testing**: Tests rate limiting implementation and response security
- **CORS Policy Validation**: Validates Cross-Origin Resource Sharing configuration
- **Content-Type Security**: Tests content type handling and validation
- **Response Sanitization**: Ensures all responses are properly sanitized
- **Error Response Consistency**: Validates consistent error response structure
- **Response Timing Consistency**: Tests for timing side-channel vulnerabilities

### 4. Production Hardening Validation (`test_production_hardening_validation.py`)

- **Debug Mode Disabled Verification**: Ensures debug mode is disabled in production
- **Environment Configuration**: Validates production environment variables and settings
- **Security Headers Production**: Tests production-ready security header configuration
- **Error Handling Production**: Validates production error handling implementation
- **Logging Configuration**: Tests production logging setup and security
- **Database Security Configuration**: Validates database connection security
- **SSL/TLS Configuration**: Tests HTTPS and certificate configuration
- **Infrastructure Security**: Validates infrastructure security settings

## Quick Start

### Running All Tests

```bash
# Run comprehensive test suite
python tests/security/run_comprehensive_error_disclosure_tests.py

# Generate HTML and JSON reports
python tests/security/run_comprehensive_error_disclosure_tests.py --output-format both

# Run quietly (minimal output)
python tests/security/run_comprehensive_error_disclosure_tests.py --quiet
```

### Running Individual Test Suites

```bash
# Run with pytest
cd /home/green/FreeAgentics

# Run all security tests
pytest tests/security/ -v

# Run specific test categories
pytest tests/security/ -m error_handling -v
pytest tests/security/ -m authentication -v
pytest tests/security/ -m api_security -v
pytest tests/security/ -m production_hardening -v

# Run individual test files
pytest tests/security/test_error_handling_information_disclosure.py -v
pytest tests/security/test_authentication_error_disclosure.py -v
pytest tests/security/test_api_security_responses.py -v
pytest tests/security/test_production_hardening_validation.py -v
```

### Running Individual Tests Standalone

```bash
# Run error handling tests
python tests/security/test_error_handling_information_disclosure.py

# Run authentication error tests
python tests/security/test_authentication_error_disclosure.py

# Run API security response tests
python tests/security/test_api_security_responses.py

# Run production hardening tests
python tests/security/test_production_hardening_validation.py
```

## Test Results and Reports

### Report Formats

The test suite generates comprehensive reports in multiple formats:

1. **JSON Reports**: Machine-readable detailed results with all test data
2. **HTML Reports**: Human-readable reports with visual formatting and charts
3. **Console Output**: Real-time test progress and summary

### Report Locations

Reports are saved to `/home/green/FreeAgentics/tests/security/` with timestamps:

- `comprehensive_error_disclosure_report_YYYYMMDD_HHMMSS.json`
- `comprehensive_error_disclosure_report_YYYYMMDD_HHMMSS.html`
- Individual test suite reports with specific naming

### Understanding Report Scores

- **Security Score**: Overall security posture (0-100)
- **Production Readiness Score**: Readiness for production deployment (0-100)
- **Pass Rate**: Percentage of tests that passed
- **Issue Severity**: Critical, High, Medium, Low classifications

## Security Standards and Compliance

### OWASP Top 10 Coverage

These tests address several OWASP Top 10 security risks:

- **A01:2021 – Broken Access Control**: Authentication and authorization testing
- **A03:2021 – Injection**: SQL injection and command injection testing
- **A04:2021 – Insecure Design**: Security design validation
- **A05:2021 – Security Misconfiguration**: Production hardening validation
- **A06:2021 – Vulnerable and Outdated Components**: Version disclosure testing
- **A07:2021 – Identification and Authentication Failures**: Authentication security testing
- **A09:2021 – Security Logging and Monitoring Failures**: Error handling and logging testing

### Security Benchmarks

The tests validate against industry security benchmarks:

- **CIS Controls**: Basic security hygiene and configuration
- **NIST Cybersecurity Framework**: Risk management and security controls
- **ISO 27001**: Information security management
- **SANS Top 25**: Most dangerous software errors

## Common Issues and Remediation

### Critical Issues (Must Fix Before Production)

1. **Debug Mode Enabled**

   - **Issue**: Debug information exposed in responses
   - **Fix**: Set `DEBUG=False` and `PRODUCTION=True`

2. **Sensitive Information in Error Messages**

   - **Issue**: Database errors, stack traces, or file paths exposed
   - **Fix**: Implement generic error handling middleware

3. **Missing Security Headers**

   - **Issue**: Required security headers not present
   - **Fix**: Configure comprehensive security headers middleware

4. **Weak Authentication Error Handling**
   - **Issue**: Username enumeration or timing attacks possible
   - **Fix**: Implement consistent authentication responses

### High Severity Issues

1. **Information Disclosure in API Responses**

   - **Issue**: Internal details exposed in API responses
   - **Fix**: Implement response sanitization

2. **Insufficient Rate Limiting**

   - **Issue**: APIs vulnerable to abuse
   - **Fix**: Configure proper rate limiting

3. **CORS Misconfiguration**
   - **Issue**: Overly permissive CORS policy
   - **Fix**: Restrict CORS to known origins

### Medium Severity Issues

1. **Missing Content Security Policy**

   - **Issue**: XSS protection not optimal
   - **Fix**: Implement strict CSP

2. **Inconsistent Error Responses**
   - **Issue**: Error format varies across endpoints
   - **Fix**: Standardize error response structure

## Integration with CI/CD

### Automated Testing

Add to your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
name: Security Tests
on: [push, pull_request]
jobs:
  security-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.8"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run security tests
        run: python tests/security/run_comprehensive_error_disclosure_tests.py --quiet
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: security-reports
          path: tests/security/*_report_*.html
```

### Quality Gates

Recommended quality gates for production deployment:

- **Security Score**: Minimum 85/100
- **Critical Issues**: 0
- **High Issues**: Maximum 2
- **Pass Rate**: Minimum 95%
- **Production Readiness**: Must be marked as "Production Ready"

## Configuration

### Environment Variables

Required for production testing:

```bash
export PRODUCTION=true
export SECRET_KEY="your-production-secret-key"
export JWT_SECRET="your-production-jwt-secret"
export DATABASE_URL="your-production-database-url"
export REDIS_URL="your-production-redis-url"
```

### Test Configuration

Customize test behavior in `conftest.py`:

- Mock configurations for testing
- Test data and payloads
- Security policy configurations
- Environment setup

## Contributing

### Adding New Tests

1. Create test class inheriting from appropriate base
2. Implement test methods following naming convention
3. Add appropriate pytest markers
4. Update this README with new test coverage

### Test Patterns

Follow these patterns for consistency:

```python
def test_security_feature(self, client):
    """Test description."""
    # Arrange
    test_data = {...}

    # Act
    response = client.post('/endpoint', json=test_data)

    # Assert
    assert response.status_code == expected_status
    assert 'sensitive_info' not in response.text
```

### Reporting Issues

When tests fail, include:

1. Full test output and error messages
2. Environment configuration
3. Steps to reproduce
4. Expected vs actual behavior

## Support

For questions or issues:

1. Check the test output and recommendations
2. Review this README and test documentation
3. Examine the test code for implementation details
4. Create an issue with detailed information

## Security Notice

These tests are designed to identify security vulnerabilities. Some tests may:

- Generate intentionally malicious payloads
- Attempt to trigger error conditions
- Test authentication and authorization bypasses

**Do not run these tests against production systems without proper authorization and during maintenance windows.**

## License

This security testing suite is part of the FreeAgentics project and follows the same licensing terms.
