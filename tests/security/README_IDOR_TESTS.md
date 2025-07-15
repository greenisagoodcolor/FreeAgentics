# IDOR Vulnerability Test Suite

## Overview

This comprehensive test suite validates that the FreeAgentics platform is protected against IDOR (Insecure Direct Object Reference) vulnerabilities. The tests cover all major IDOR attack patterns as defined by OWASP and security best practices.

## Test Coverage

### 1. Sequential ID Enumeration (`test_idor_vulnerabilities.py`)
- **Agent ID Enumeration**: Tests protection against sequential agent ID access
- **User ID Enumeration**: Validates user profile protection
- **Resource ID Prediction**: Ensures resources can't be accessed through ID patterns
- **Coalition ID Brute Forcing**: Tests coalition access control

### 2. UUID/GUID Attacks (`test_idor_vulnerabilities.py`)
- **UUID Prediction**: Tests against timestamp-based UUID prediction
- **UUID Enumeration**: Validates random UUID generation
- **UUID Collision Testing**: Ensures proper handling of UUID collisions
- **Timestamp-based UUID Attacks**: Tests v1 UUID vulnerabilities

### 3. Parameter Manipulation (`test_idor_vulnerabilities.py`)
- **Query Parameter IDOR**: Tests `?user_id=` style attacks
- **Path Parameter IDOR**: Validates `/api/v1/agents/{id}` protection
- **Form Data IDOR**: Tests POST form parameter manipulation
- **JSON Payload IDOR**: Validates JSON request body protection

### 4. Authorization Bypass (`test_idor_vulnerabilities.py`)
- **Direct Object Access**: Tests unauthorized direct resource access
- **Resource Ownership Bypass**: Validates ownership verification
- **Cross-tenant Access**: Tests multi-tenant isolation
- **File Path Traversal**: Validates path traversal protection

### 5. Advanced IDOR Attacks (`test_idor_advanced_patterns.py`)
- **Blind IDOR Detection**: Tests timing-based information leakage
- **Time-based IDOR**: Validates timestamp-based access attempts
- **Mass Assignment IDOR**: Tests parameter pollution attacks
- **Indirect Object References**: Validates reference chain protection

### 6. Specialized Attack Vectors (`test_idor_advanced_patterns.py`)
- **GraphQL IDOR**: Tests GraphQL-specific vulnerabilities
- **WebSocket IDOR**: Validates real-time connection security
- **Batch Operation IDOR**: Tests bulk operation authorization
- **Cache Poisoning IDOR**: Validates cache key security
- **API Versioning IDOR**: Tests legacy API endpoint protection

### 7. Integration Testing (`test_idor_integration.py`)
- **Cross-User Access**: Comprehensive user isolation testing
- **Role-Based Access**: Validates RBAC implementation
- **Coalition Membership**: Tests group-based access control
- **Knowledge Graph Access**: Validates graph database security

### 8. File Operation Security (`test_idor_file_operations.py`)
- **File Upload/Download**: Tests file access control
- **Document Access**: Validates document permissions
- **Model File Protection**: Tests ML model security
- **Configuration Files**: Validates config file isolation
- **Archive Extraction**: Tests zip/tar security

## Running the Tests

### Run All IDOR Tests
```bash
# Run the complete test suite
python -m pytest tests/security/test_idor_validation_suite.py -v

# Run with coverage report
python -m pytest tests/security/test_idor_*.py --cov=api --cov=auth --cov-report=html
```

### Run Specific Test Categories
```bash
# Basic IDOR tests
pytest tests/security/test_idor_vulnerabilities.py -v

# Advanced patterns
pytest tests/security/test_idor_advanced_patterns.py -v

# Integration tests
pytest tests/security/test_idor_integration.py -v

# File operations
pytest tests/security/test_idor_file_operations.py -v
```

### Run Individual Test Cases
```bash
# Test specific vulnerability
pytest tests/security/test_idor_vulnerabilities.py::TestSequentialIDEnumeration::test_agent_id_enumeration -v

# Test with specific marker
pytest tests/security -m "idor" -v
```

## Test Structure

Each test module follows this structure:

```python
class TestCategoryName(IDORTestBase):
    """Test specific IDOR category."""
    
    @pytest.fixture(autouse=True)
    def setup(self, db: Session):
        """Set up test environment."""
        # Create test users with different roles
        # Create test resources
        # Establish relationships
    
    def test_specific_vulnerability(self):
        """Test specific IDOR attack pattern."""
        # Attempt unauthorized access
        # Verify protection is effective
        # Check error responses don't leak info
```

## Key Testing Principles

### 1. Comprehensive Coverage
- Test all resource types (agents, users, coalitions, files)
- Cover all HTTP methods (GET, POST, PUT, DELETE, PATCH)
- Test all authentication methods (JWT, API keys, sessions)

### 2. Realistic Attack Scenarios
- Use actual attack patterns from security research
- Test both simple and sophisticated attacks
- Include timing and blind attacks

### 3. Error Response Validation
- Ensure consistent error messages (404 for all unauthorized)
- Verify no information leakage in errors
- Test error response timing

### 4. Authorization Verification
- Test object-level authorization
- Verify role-based access control
- Validate ownership checks

## Security Best Practices Validated

### 1. Use UUIDs
- ✅ All resources use UUIDs instead of sequential IDs
- ✅ UUIDs are properly random (version 4)
- ✅ No predictable patterns in ID generation

### 2. Object-Level Authorization
- ✅ Every request validates resource ownership
- ✅ Authorization checked at object level, not just endpoint
- ✅ Consistent authorization across all operations

### 3. Consistent Error Responses
- ✅ Return 404 for both non-existent and unauthorized
- ✅ No timing differences in responses
- ✅ No information leakage in error messages

### 4. Input Validation
- ✅ Validate all ID formats
- ✅ Reject malformed requests early
- ✅ Sanitize file paths and names

### 5. Audit Logging
- ✅ Log all authorization failures
- ✅ Track access patterns
- ✅ Enable security monitoring

## Interpreting Results

### Success Criteria
- All tests must pass (0 failures)
- No security warnings in output
- Coverage above 90% for security modules

### Common Failure Patterns
1. **Sequential ID Found**: Resource uses predictable IDs
2. **Ownership Not Verified**: Missing authorization check
3. **Information Leakage**: Error reveals resource existence
4. **Timing Attack Possible**: Different response times

### Remediation Steps
1. Replace sequential IDs with UUIDs
2. Add `@require_permission` decorators
3. Implement consistent error responses
4. Add ownership validation middleware

## Continuous Security

### Pre-commit Hooks
```yaml
- repo: local
  hooks:
    - id: idor-tests
      name: IDOR Security Tests
      entry: pytest tests/security/test_idor_vulnerabilities.py
      language: system
      pass_filenames: false
      always_run: true
```

### CI/CD Integration
```yaml
security-tests:
  runs-on: ubuntu-latest
  steps:
    - name: Run IDOR Tests
      run: |
        python -m pytest tests/security/test_idor_*.py -v
        python tests/security/test_idor_validation_suite.py
```

## Maintenance

### Adding New Tests
1. Identify new IDOR pattern
2. Add test to appropriate module
3. Follow existing test structure
4. Update this README

### Updating Tests
1. Keep tests current with API changes
2. Add tests for new endpoints
3. Update attack patterns as needed
4. Review OWASP updates

## Resources

- [OWASP IDOR Prevention](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/05-Authorization_Testing/04-Testing_for_Insecure_Direct_Object_References)
- [IDOR Vulnerability Guide](https://portswigger.net/web-security/access-control/idor)
- [API Security Best Practices](https://owasp.org/www-project-api-security/)

## Contact

For security concerns or questions about these tests:
- File a security issue (private)
- Contact the security team
- See SECURITY.md for disclosure policy