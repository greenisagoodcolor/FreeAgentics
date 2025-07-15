# Security Testing Quick Start Guide

## Overview

This quick start guide helps teams get up and running with security testing in the FreeAgentics project. For comprehensive documentation, see [SECURITY_TESTING_COMPREHENSIVE_GUIDE.md](./SECURITY_TESTING_COMPREHENSIVE_GUIDE.md).

## Prerequisites

### Required Tools
```bash
# Install security testing tools
pip install bandit[toml] safety semgrep pytest-security
pip install -r requirements-core.txt
pip install -r requirements-dev.txt

# Install container security tools (optional)
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh
```

### Environment Setup
```bash
# Copy environment template
cp .env.example .env.security-test

# Set required environment variables
export ENVIRONMENT=security-test
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/freeagentics_security
export REDIS_URL=redis://localhost:6379/1
export JWT_SECRET_KEY=test-secret-key-for-security-testing
```

## Quick Security Tests

### 1. Run Basic Security Linting
```bash
# Static security analysis
bandit -r . -f txt --skip B101,B601,B603 --exclude .archive,web,tests

# Dependency vulnerability check
safety check

# Secret detection
semgrep --config=p/secrets .
```

### 2. Run Authentication Security Tests
```bash
# Basic authentication tests
python -m pytest tests/security/test_comprehensive_auth_security.py::TestAuthenticationSecurity::test_login_timing_attack_prevention -v

# JWT security tests
python -m pytest tests/security/test_jwt_manipulation_vulnerabilities.py::TestJWTSecurity::test_token_manipulation_detection -v
```

### 3. Run Authorization Security Tests
```bash
# RBAC tests
python -m pytest tests/security/test_rbac_authorization_matrix.py::TestRBACMatrix::test_role_permissions -v

# IDOR tests
python -m pytest tests/security/test_idor_validation_suite.py::TestIDORPrevention::test_horizontal_privilege_escalation -v
```

### 4. Run Input Validation Tests
```bash
# SQL injection prevention
python -m pytest tests/security/test_injection_prevention.py::TestSQLInjectionPrevention -v

# XSS prevention
python -m pytest tests/security/test_injection_prevention.py::TestXSSPrevention -v
```

## Security Test Categories

| Category | Test Files | Command |
|----------|------------|---------|
| Authentication | `test_comprehensive_auth_security.py` | `pytest tests/security/test_comprehensive_auth_security.py -v` |
| Authorization | `test_rbac_authorization_matrix.py`, `test_idor_*.py` | `pytest tests/security/test_rbac_authorization_matrix.py -v` |
| Input Validation | `test_injection_prevention.py` | `pytest tests/security/test_injection_prevention.py -v` |
| Rate Limiting | `test_rate_limiting_*.py` | `pytest tests/security/test_rate_limiting_comprehensive.py -v` |
| Session Security | `test_session_security.py` | `pytest tests/integration/test_session_security.py -v` |

## CI/CD Integration

### GitHub Actions
The project includes a comprehensive security CI/CD pipeline at `.github/workflows/security-ci.yml` that runs:

- Static Application Security Testing (SAST)
- Dependency vulnerability scanning
- Container security scanning
- Dynamic Application Security Testing (DAST)
- Security integration tests

### Manual Pipeline Trigger
```bash
# Trigger security pipeline manually
gh workflow run security-ci.yml
```

## Security Metrics Dashboard

### Key Metrics to Monitor
- **Security Score**: Overall security posture (0-100)
- **Vulnerability Count**: By severity (Critical, High, Medium, Low)
- **Test Coverage**: Percentage of security tests passing
- **Compliance Status**: OWASP Top 10, GDPR, SOC 2 compliance

### Viewing Security Reports
```bash
# Generate security report
python scripts/security/generate_security_report.py --output security-report.html

# Calculate security score
python scripts/security/calculate_security_score.py
```

## Common Security Test Scenarios

### 1. Testing Authentication Flow
```python
def test_login_security():
    # Valid login
    response = client.post("/api/v1/auth/login", json={
        "username": "testuser",
        "password": "SecurePassword123!"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()
    
    # Invalid login
    response = client.post("/api/v1/auth/login", json={
        "username": "testuser",
        "password": "WrongPassword"
    })
    assert response.status_code == 401
```

### 2. Testing Authorization
```python
def test_rbac_enforcement():
    # Admin user should access admin endpoints
    admin_response = admin_client.get("/api/v1/admin/users")
    assert admin_response.status_code == 200
    
    # Regular user should be denied
    user_response = user_client.get("/api/v1/admin/users")
    assert user_response.status_code == 403
```

### 3. Testing Input Validation
```python
def test_sql_injection_prevention():
    malicious_input = "'; DROP TABLE users; --"
    response = client.get(f"/api/v1/search?query={malicious_input}")
    
    # Should not cause server error
    assert response.status_code in [200, 400]
    
    # Should not leak sensitive data
    assert "password" not in response.text.lower()
```

## Troubleshooting

### Common Issues

1. **Tests Failing Due to Missing Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Database Connection Issues**
   ```bash
   # Check database is running
   pg_isready -h localhost -p 5432
   
   # Create test database if needed
   createdb freeagentics_security
   ```

3. **Redis Connection Issues**
   ```bash
   # Check Redis is running
   redis-cli ping
   
   # Start Redis if needed
   redis-server
   ```

4. **Security Tool Installation Issues**
   ```bash
   # Update pip and try again
   pip install --upgrade pip
   pip install bandit[toml] safety semgrep
   ```

### Debug Mode
```bash
# Enable security test debugging
export SECURITY_DEBUG=true
export LOG_LEVEL=DEBUG

# Run tests with verbose output
python -m pytest tests/security/ -v -s --tb=long
```

## Security Testing Best Practices

### 1. Test Isolation
- Use separate test databases and Redis instances
- Clean up test data between tests
- Use mock objects for external dependencies

### 2. Test Data Management
- Use realistic but non-sensitive test data
- Rotate test credentials regularly
- Document test user accounts and permissions

### 3. Continuous Testing
- Run security tests on every commit
- Include security tests in feature branch testing
- Monitor security metrics over time

### 4. Security Test Maintenance
- Review and update security tests quarterly
- Keep security tools and dependencies updated
- Add new tests for emerging threat patterns

## Next Steps

1. **Run Full Security Test Suite**: `make security-test`
2. **Review Security Dashboard**: Check security metrics and compliance status
3. **Address Security Issues**: Prioritize and fix identified vulnerabilities
4. **Implement Continuous Monitoring**: Set up automated security scanning
5. **Security Training**: Ensure team members understand security testing practices

## Resources

- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [Security Testing Documentation](./SECURITY_TESTING_COMPREHENSIVE_GUIDE.md)
- [Security Test Catalog](./SECURITY_TEST_CATALOG.md)
- [Compliance Documentation](./COMPLIANCE_GUIDE.md)

## Support

For security testing questions or issues:
1. Check the troubleshooting section above
2. Review existing security test examples
3. Consult the comprehensive security testing guide
4. Contact the security team for complex issues