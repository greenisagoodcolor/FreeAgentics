# Security Testing Guide for FreeAgentics

## Overview

This guide documents the comprehensive security testing framework implemented for the FreeAgentics platform. Our security testing suite covers multiple layers of security validation to ensure the platform remains secure against common and advanced attack vectors.

## Security Test Categories

### 1. SQL Injection Testing

- **File**: `comprehensive_security_test_suite.py`
- **Coverage**: All database-interacting endpoints
- **Test Types**:
  - Classic SQL injection patterns
  - Blind SQL injection
  - Time-based SQL injection
  - Union-based attacks
  - Second-order SQL injection

### 2. Cross-Site Scripting (XSS) Testing

- **Types Covered**:
  - Reflected XSS
  - Stored/Persistent XSS
  - DOM-based XSS
- **Test Locations**:
  - Search parameters
  - User input fields
  - API responses
  - Error messages

### 3. CSRF Protection Testing

- **Validation Points**:
  - State-changing operations
  - API endpoints
  - Form submissions
- **Token Types Tested**:
  - Double-submit cookies
  - Synchronizer tokens
  - Custom headers

### 4. Authentication Security

- **Test Scenarios**:
  - Brute force protection
  - Password policy enforcement
  - Session management
  - Multi-factor authentication
  - Account lockout mechanisms

### 5. Authorization Testing

- **Coverage**:
  - Horizontal privilege escalation
  - Vertical privilege escalation
  - IDOR (Insecure Direct Object Reference)
  - Resource access control
  - Role-based permissions

### 6. JWT Security Testing

- **Attack Vectors**:
  - Algorithm confusion attacks
  - None algorithm bypass
  - Weak secret brute forcing
  - Token expiration handling
  - Claim tampering
  - Key confusion attacks

### 7. Rate Limiting Tests

- **Bypass Attempts**:
  - Header manipulation
  - IP spoofing
  - Path variations
  - Parameter pollution
- **Protected Endpoints**:
  - Login/authentication
  - API calls
  - Resource creation
  - Password reset

### 8. Input Validation & Fuzzing

- **Test Inputs**:
  - Buffer overflow attempts
  - Null byte injection
  - Path traversal patterns
  - Command injection
  - Template injection
  - NoSQL injection

### 9. API Security Testing

- **Scenarios**:
  - Resource exhaustion
  - Batch operation abuse
  - GraphQL depth attacks
  - Webhook flooding
  - File upload vulnerabilities
  - API versioning attacks

## Test Execution

### Running Individual Test Categories

```bash
# Run SQL injection tests only
pytest tests/security/comprehensive_security_test_suite.py::TestComprehensiveSecurity::test_sql_injection_protection -v

# Run XSS tests only
pytest tests/security/comprehensive_security_test_suite.py::TestComprehensiveSecurity::test_xss_protection -v

# Run authentication tests
pytest tests/security/comprehensive_security_test_suite.py::TestComprehensiveSecurity::test_authentication_security -v
```

### Running Full Security Suite

```bash
# Run all security tests
python tests/security/comprehensive_security_test_suite.py

# Run with pytest
pytest tests/security/comprehensive_security_test_suite.py -v

# Run penetration tests
python tests/security/run_comprehensive_penetration_tests.py
```

### OWASP ZAP Integration

```bash
# Start ZAP in daemon mode
docker run -d -p 8080:8080 --name zap owasp/zap2docker-stable zap.sh -daemon -port 8080 -host 0.0.0.0 -config api.key=your-api-key

# Run ZAP integration tests
python tests/security/owasp_zap_integration.py
```

## CI/CD Integration

### GitHub Actions

The security tests are automatically run in GitHub Actions on:

- Every push to main/develop branches
- All pull requests

Configuration: `.github/workflows/security-tests.yml`

### GitLab CI

Security stage in pipeline includes:

- Security test suite
- Dependency scanning
- Container scanning
- SAST analysis

Configuration: `.gitlab-ci-security.yml`

### Jenkins

Security pipeline stages:

1. Unit security tests
1. Penetration tests
1. SAST scanning
1. Dependency scanning
1. OWASP ZAP scan
1. Security gate validation

Configuration: `Jenkinsfile.security`

## Security Gates

Security gates enforce the following thresholds:

| Metric | Threshold | Action |
|--------|-----------|--------|
| High-risk vulnerabilities | 0 | Build fails |
| Medium-risk vulnerabilities | ≤ 5 | Build fails if exceeded |
| Low-risk vulnerabilities | ≤ 20 | Warning if exceeded |
| Security score | ≥ 85/100 | Build fails if lower |
| Test coverage | ≥ 90% | Build fails if lower |
| OWASP compliance | ≥ 9/10 | Build fails if lower |

## Test Reports

### Generated Reports

1. **Security Test Report** (`security_test_report.json`)

   - Comprehensive test results
   - Vulnerability summary
   - OWASP compliance status
   - Recommendations

1. **ZAP Report** (`zap_security_report.html/json/xml`)

   - DAST scan results
   - Risk categorization
   - Detailed vulnerability information

1. **Bandit Report** (`bandit-report.json`)

   - SAST analysis results
   - Code security issues
   - Severity classification

1. **Safety Report** (`safety-report.json`)

   - Dependency vulnerabilities
   - Package security status
   - Update recommendations

1. **Penetration Test Report** (`penetration_test_report.json`)

   - Exploit attempts
   - Critical findings
   - Attack success rate

### Report Analysis

Use the security gate validator to analyze reports:

```bash
python tests/security/check_security_gates.py
```

## Best Practices

### 1. Regular Testing

- Run security tests before every deployment
- Schedule weekly full security scans
- Perform monthly manual penetration testing

### 2. Test Data Management

- Use dedicated test accounts
- Clean up test data after runs
- Never use production data in tests

### 3. False Positive Management

- Document known false positives
- Configure exclusions appropriately
- Review and update regularly

### 4. Remediation Process

1. Identify vulnerability
1. Assess risk and impact
1. Develop fix
1. Test fix thoroughly
1. Deploy with monitoring
1. Verify remediation

### 5. Security Test Maintenance

- Update test payloads regularly
- Add tests for new features
- Monitor security advisories
- Update OWASP mappings

## Troubleshooting

### Common Issues

1. **ZAP Connection Failed**

   - Ensure ZAP is running on port 8080
   - Check API key configuration
   - Verify network connectivity

1. **Test Timeouts**

   - Increase timeout values for slow endpoints
   - Check rate limiting configuration
   - Verify service availability

1. **False Positives**

   - Review security headers
   - Check input validation rules
   - Verify encoding/escaping

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Compliance Mapping

### OWASP Top 10 (2021)

| Category | Tests | Status |
|----------|-------|--------|
| A01: Broken Access Control | Authorization tests, CSRF tests | ✓ |
| A02: Cryptographic Failures | JWT tests, Encryption validation | ✓ |
| A03: Injection | SQL injection, XSS, Command injection | ✓ |
| A04: Insecure Design | Architecture review required | ⚠️ |
| A05: Security Misconfiguration | Header tests, Error handling | ✓ |
| A06: Vulnerable Components | Dependency scanning | ✓ |
| A07: Auth Failures | Authentication tests | ✓ |
| A08: Software Integrity | CSRF, Input validation | ✓ |
| A09: Security Logging | Logging tests | ✓ |
| A10: SSRF | URL validation tests | ✓ |

### PCI DSS Requirements

- Requirement 6.5: Addressed common coding vulnerabilities
- Requirement 11.3: Penetration testing implementation
- Requirement 12.3: Security testing in SDLC

### GDPR Compliance

- Data protection by design
- Security of processing (Article 32)
- Regular testing and evaluation

## Continuous Improvement

### Monthly Reviews

1. Analyze security trends
1. Update test cases
1. Review false positives
1. Update thresholds

### Quarterly Assessments

1. Full penetration test
1. Third-party security audit
1. Compliance review
1. Training updates

### Annual Planning

1. Security roadmap review
1. Tool evaluation
1. Process improvements
1. Budget planning

## Contact Information

For security-related questions or to report vulnerabilities:

- Security Team: security@freeagentics.com
- Security Hotline: +1-555-SEC-URITY
- Bug Bounty: https://freeagentics.com/security/bug-bounty

## References

- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
