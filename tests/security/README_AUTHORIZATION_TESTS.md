# FreeAgentics Authorization Security Testing Framework

This directory contains comprehensive authorization boundary testing for the FreeAgentics platform, ensuring production-grade security for all authorization scenarios.

## Overview

The authorization testing framework validates:
- Role-Based Access Control (RBAC) boundaries
- Resource-level authorization
- API endpoint authorization
- Advanced authorization scenarios (ABAC, context-aware, time-based)
- Defense against authorization attacks

## Test Structure

### Core Test Modules

1. **`test_authorization_boundaries.py`**
   - Role-based authorization testing
   - Permission boundary validation
   - Resource ownership and access controls
   - API endpoint authorization
   - Advanced ABAC scenarios

2. **`test_authorization_attacks.py`**
   - IDOR vulnerability testing
   - Privilege escalation vectors
   - Authorization bypass techniques
   - Token manipulation attacks
   - Race condition testing

3. **`test_authorization_integration.py`**
   - Integration with security headers
   - Rate limiting behavior
   - Audit logging validation
   - Database operation security
   - Complex real-world scenarios

4. **`test_rbac_authorization_matrix.py`** (existing)
   - Complete RBAC matrix validation
   - Permission inheritance testing
   - Cross-tenant isolation

## Running Tests

### Run All Authorization Tests
```bash
python tests/security/run_authorization_tests.py
```

### Run Specific Test Suite
```bash
# Test authorization boundaries
pytest tests/security/test_authorization_boundaries.py -v

# Test attack vectors
pytest tests/security/test_authorization_attacks.py -v

# Test integration scenarios
pytest tests/security/test_authorization_integration.py -v
```

### Run Specific Test Class
```bash
# Test only IDOR vulnerabilities
pytest tests/security/test_authorization_attacks.py::TestIDORVulnerabilities -v

# Test only role-based boundaries
pytest tests/security/test_authorization_boundaries.py::TestRoleBasedAuthorizationBoundaries -v
```

## Test Coverage

### Authorization Scenarios Covered

#### 1. Role-Based Authorization
- [x] Permission boundary validation
- [x] Role hierarchy enforcement
- [x] Permission inheritance
- [x] Cross-role access attempts
- [x] Role elevation attacks

#### 2. Resource-Level Authorization
- [x] Resource ownership validation
- [x] Cross-tenant isolation
- [x] Resource hierarchy permissions
- [x] Resource-specific access policies

#### 3. API Endpoint Authorization
- [x] Endpoint permission validation
- [x] HTTP method-based access control
- [x] API versioning authorization
- [x] Parameter-based access control
- [x] Middleware chain validation

#### 4. Advanced Authorization
- [x] Attribute-Based Access Control (ABAC)
- [x] Context-aware authorization
- [x] Time-based access restrictions
- [x] Location-based access control
- [x] Dynamic permission evaluation

#### 5. Attack Vector Testing
- [x] Horizontal privilege escalation
- [x] Vertical privilege escalation
- [x] IDOR vulnerabilities
- [x] Authorization bypass attempts
- [x] Token manipulation
- [x] Race conditions
- [x] Cache poisoning
- [x] HTTP verb tampering

## Security Test Patterns

### 1. Boundary Testing Pattern
```python
def test_permission_boundary_validation(self, test_users):
    """Test that each role can only access permissions within their boundary."""
    for role, permissions in ROLE_PERMISSIONS.items():
        user_data = test_users[role]
        token_data = auth_manager.verify_token(user_data["token"])

        # Verify exact permission set
        assert set(token_data.permissions) == set(permissions)

        # Verify no unauthorized permissions
        unauthorized = set(Permission) - set(permissions)
        for perm in unauthorized:
            assert perm not in token_data.permissions
```

### 2. Attack Simulation Pattern
```python
def test_idor_attack(self, client, users):
    """Test IDOR vulnerability defense."""
    # User 1 creates resource
    resource_id = create_resource(user1)

    # User 2 attempts unauthorized access
    response = client.get(f"/api/v1/resource/{resource_id}",
                         headers=user2_headers)

    # Verify proper authorization
    assert response.status_code == status.HTTP_403_FORBIDDEN
```

### 3. ABAC Evaluation Pattern
```python
def test_context_aware_authorization(self):
    """Test authorization based on context."""
    context = AccessContext(
        user_id="user_001",
        role=UserRole.RESEARCHER,
        permissions=list(ROLE_PERMISSIONS[UserRole.RESEARCHER]),
        ip_address="192.168.1.100",
        risk_score=0.2
    )

    resource = ResourceContext(
        resource_type="sensitive_data",
        sensitivity_level="restricted"
    )

    granted, reason, rules = enhanced_rbac_manager.evaluate_abac_access(
        context, resource, "view"
    )

    # Verify decision based on context
    assert not granted if context.risk_score > 0.8 else True
```

## Test Reports

The test runner generates comprehensive reports:

### JSON Report (`authorization_test_report.json`)
```json
{
  "metadata": {
    "test_run_id": "auth_test_20240114_120000",
    "environment": "test",
    "python_version": "3.9.0"
  },
  "test_suites": {
    "Authorization Boundaries": {
      "summary": {
        "total": 45,
        "passed": 45,
        "failed": 0
      }
    }
  },
  "security_analysis": {
    "authorization_coverage": {
      "coverage_percentage": 95.2
    },
    "vulnerability_summary": {
      "critical": [],
      "high": []
    },
    "recommendations": [
      "Enable comprehensive audit logging",
      "Implement rate limiting on failures"
    ]
  }
}
```

### Markdown Report (`AUTHORIZATION_TEST_REPORT.md`)
Human-readable report with:
- Test summary and statistics
- Critical security issues
- Failed test details
- Security analysis
- Recommendations

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Clean up resources after tests
- Use fixtures for setup/teardown

### 2. Comprehensive Coverage
- Test both positive and negative cases
- Include edge cases and boundaries
- Simulate real attack scenarios

### 3. Performance Considerations
- Monitor authorization latency
- Test under concurrent load
- Verify consistency under stress

### 4. Security Focus
- Fail fast on security violations
- Log all suspicious activities
- Generate actionable reports

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Authorization Security Tests

on: [push, pull_request]

jobs:
  auth-security-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-json-report

      - name: Run Authorization Tests
        run: python tests/security/run_authorization_tests.py

      - name: Upload Test Report
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: authorization-test-report
          path: |
            authorization_test_report.json
            AUTHORIZATION_TEST_REPORT.md
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH includes project root
   export PYTHONPATH=/home/green/FreeAgentics:$PYTHONPATH
   ```

2. **Database Connection Issues**
   ```bash
   # Use test database
   export DATABASE_URL="sqlite:///test.db"
   ```

3. **Missing Dependencies**
   ```bash
   pip install pytest pytest-json-report pytest-asyncio
   ```

## Security Considerations

1. **Never Skip Security Tests**
   - All authorization tests must pass before deployment
   - Failed tests indicate potential vulnerabilities

2. **Regular Updates**
   - Update test cases for new features
   - Add tests for discovered vulnerabilities
   - Review and update ABAC rules

3. **Production Validation**
   - Run subset of tests against staging
   - Monitor authorization metrics in production
   - Set up alerts for authorization failures

## Contributing

When adding new authorization features:
1. Add corresponding test cases
2. Update test documentation
3. Ensure all tests pass
4. Review security implications

## License

These tests are part of the FreeAgentics security framework and follow the same license as the main project.
