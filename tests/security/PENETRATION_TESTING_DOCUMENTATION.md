# FreeAgentics Penetration Testing Documentation

## Overview

This document provides comprehensive documentation for the penetration testing framework implemented for the FreeAgentics platform. The testing suite is designed to identify security vulnerabilities and provide actionable remediation guidance.

## Table of Contents

1. [Testing Framework Architecture](#testing-framework-architecture)
2. [Test Coverage Areas](#test-coverage-areas)
3. [Vulnerability Classification](#vulnerability-classification)
4. [Proof of Concept Examples](#proof-of-concept-examples)
5. [Remediation Guidelines](#remediation-guidelines)
6. [Running the Tests](#running-the-tests)
7. [Report Generation](#report-generation)
8. [Integration with CI/CD](#integration-with-cicd)

## Testing Framework Architecture

### Core Components

The penetration testing framework consists of the following components:

```
tests/security/
├── test_error_handling_information_disclosure.py    # Error handling security
├── test_authentication_error_disclosure.py          # Authentication security
├── test_api_security_responses.py                   # API security validation
├── test_production_hardening_validation.py          # Production readiness
├── test_file_upload_security.py                     # File upload security
├── test_path_traversal_prevention.py                # Path traversal testing
├── run_comprehensive_penetration_tests.py           # Main test runner
├── run_comprehensive_error_disclosure_tests.py      # Error disclosure runner
└── conftest.py                                       # Test configuration
```

### Testing Methodology

The framework follows industry-standard testing methodologies:

- **OWASP Testing Guide v4.0**: Comprehensive web application security testing
- **NIST Cybersecurity Framework**: Risk-based security assessment
- **SANS Top 25**: Most dangerous software errors
- **Custom Security Patterns**: FreeAgentics-specific security requirements

## Test Coverage Areas

### 1. Error Handling Information Disclosure

**Scope**: Detection of sensitive information leakage through error messages

**Test Categories**:
- Database error information leakage
- Stack trace exposure detection
- Debug information disclosure
- Internal path revelation
- Version information leakage

**Risk Level**: High
**Impact**: Information disclosure can lead to reconnaissance for further attacks

### 2. Authentication Error Disclosure

**Scope**: Authentication mechanism security validation

**Test Categories**:
- Username enumeration attacks
- Password policy disclosure
- Account lockout information leakage
- Session management error disclosure
- JWT error information leakage
- Authentication timing attacks

**Risk Level**: Critical
**Impact**: Authentication bypass, credential harvesting

### 3. API Security Response Validation

**Scope**: Comprehensive API security testing

**Test Categories**:
- Security headers validation
- Rate limiting response testing
- CORS policy validation
- Content-Type security
- Response sanitization
- Error response consistency
- Response timing consistency

**Risk Level**: High
**Impact**: API abuse, data exfiltration, XSS attacks

### 4. Production Hardening Validation

**Scope**: Production deployment security readiness

**Test Categories**:
- Debug mode disabled verification
- Environment configuration validation
- Security headers production readiness
- Error handling production configuration
- Logging configuration security
- Database security configuration
- SSL/TLS configuration validation
- Infrastructure security validation

**Risk Level**: Critical
**Impact**: Complete system compromise in production

### 5. File Upload Security

**Scope**: File upload functionality security validation

**Test Categories**:
- File extension validation and restrictions
- File size limit enforcement
- Malicious file content detection
- File metadata security
- Upload directory security
- File execution prevention
- MIME type validation and spoofing prevention

**Risk Level**: Critical
**Impact**: Remote code execution, file system access

### 6. Path Traversal Prevention

**Scope**: Directory traversal attack prevention

**Test Categories**:
- API parameter path traversal testing
- Static file serving security
- Template inclusion security
- Log file access prevention
- Configuration file access prevention
- Directory listing prevention
- Symbolic link exploitation prevention

**Risk Level**: Critical
**Impact**: File system access, configuration disclosure

## Vulnerability Classification

### Severity Levels

#### Critical (Score Impact: -25 points each)
- Remote code execution vulnerabilities
- Authentication bypass
- Complete system access
- Configuration file disclosure
- Database access without authentication

#### High (Score Impact: -15 points each)
- Information disclosure of sensitive data
- Privilege escalation
- File upload vulnerabilities
- Path traversal attacks
- XSS vulnerabilities with significant impact

#### Medium (Score Impact: -5 points each)
- Security header misconfigurations
- Minor information disclosure
- Rate limiting issues
- CORS misconfigurations
- Directory listing enabled

#### Low (Score Impact: -1 point each)
- Version disclosure
- Minor configuration issues
- Informational findings

### Risk Assessment Matrix

| Likelihood | Critical | High | Medium | Low |
|------------|----------|------|--------|-----|
| High | Critical Risk | High Risk | Medium Risk | Low Risk |
| Medium | High Risk | Medium Risk | Medium Risk | Low Risk |
| Low | Medium Risk | Low Risk | Low Risk | Informational |

## Proof of Concept Examples

### 1. Path Traversal Attack

**Vulnerability**: Unvalidated file path parameters

**Proof of Concept**:
```bash
# Attack vector
curl "http://localhost:8000/api/v1/files?filename=../../../etc/passwd"

# Expected secure response
HTTP/1.1 400 Bad Request
{"detail": "Invalid file path"}

# Vulnerable response (should NOT happen)
HTTP/1.1 200 OK
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
...
```

**Remediation**:
```python
import os
from pathlib import Path

def validate_file_path(filename: str, allowed_dir: str) -> str:
    """Safely validate and resolve file paths."""
    # Resolve the path and check if it's within allowed directory
    safe_path = Path(allowed_dir) / filename
    resolved_path = safe_path.resolve()
    
    # Ensure the resolved path is within the allowed directory
    if not str(resolved_path).startswith(str(Path(allowed_dir).resolve())):
        raise ValueError("Path traversal attempt detected")
    
    return str(resolved_path)
```

### 2. File Upload RCE

**Vulnerability**: Executable file upload with direct access

**Proof of Concept**:
```bash
# Upload malicious PHP file
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@shell.php" \
  -H "Content-Type: multipart/form-data"

# Access uploaded file to execute code
curl "http://localhost:8000/uploads/shell.php?cmd=whoami"
```

**Remediation**:
```python
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.jpg', '.png', '.gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_upload(file: UploadFile) -> bool:
    """Validate uploaded file for security."""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type {file_ext} not allowed")
    
    # Check file size
    if file.size > MAX_FILE_SIZE:
        raise ValueError("File too large")
    
    # Validate MIME type matches extension
    expected_mime = get_expected_mime_type(file_ext)
    if file.content_type != expected_mime:
        raise ValueError("MIME type mismatch")
    
    return True
```

### 3. Information Disclosure via Error Messages

**Vulnerability**: Database errors exposed to users

**Proof of Concept**:
```bash
# Trigger database error
curl "http://localhost:8000/api/v1/users/'; DROP TABLE users; --"

# Vulnerable response (should NOT happen)
HTTP/1.1 500 Internal Server Error
{
  "detail": "psycopg2.errors.SyntaxError: syntax error at or near \"DROP\" 
  LINE 1: SELECT * FROM users WHERE id = ''; DROP TABLE users; --'
  Connection: host=localhost port=5432 dbname=freeagentics user=dbuser"
}
```

**Remediation**:
```python
def handle_database_error(e: Exception) -> HTTPException:
    """Safely handle database errors without information disclosure."""
    # Log the actual error for debugging
    logger.error(f"Database error: {str(e)}")
    
    # Return generic error to user
    return HTTPException(
        status_code=500,
        detail="An internal error occurred. Please try again later."
    )
```

## Remediation Guidelines

### Immediate Actions (Critical Vulnerabilities)

1. **Stop Production Deployment**
   - Do not deploy any application with critical vulnerabilities
   - Conduct emergency security review

2. **Implement Input Validation**
   ```python
   from pydantic import BaseModel, validator
   
   class FileRequest(BaseModel):
       filename: str
       
       @validator('filename')
       def validate_filename(cls, v):
           if '..' in v or '/' in v or '\\' in v:
               raise ValueError('Invalid filename')
           return v
   ```

3. **Secure Error Handling**
   ```python
   @app.exception_handler(Exception)
   async def global_exception_handler(request: Request, exc: Exception):
       # Log detailed error
       logger.error(f"Unhandled exception: {exc}", exc_info=True)
       
       # Return generic response
       return JSONResponse(
           status_code=500,
           content={"detail": "Internal server error"}
       )
   ```

### High Priority Actions

1. **Implement Security Headers**
   ```python
   @app.middleware("http")
   async def security_headers_middleware(request: Request, call_next):
       response = await call_next(request)
       response.headers["X-Content-Type-Options"] = "nosniff"
       response.headers["X-Frame-Options"] = "DENY"
       response.headers["X-XSS-Protection"] = "1; mode=block"
       response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
       return response
   ```

2. **Implement Rate Limiting**
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
   
   @app.post("/api/v1/login")
   @limiter.limit("5/minute")
   async def login(request: Request, credentials: LoginRequest):
       # Login logic
       pass
   ```

3. **Secure File Upload**
   ```python
   UPLOAD_DIR = "/var/app/uploads"  # Outside web root
   
   async def upload_file(file: UploadFile):
       # Validate file
       validate_upload(file)
       
       # Generate safe filename
       safe_filename = f"{uuid4()}_{secure_filename(file.filename)}"
       file_path = os.path.join(UPLOAD_DIR, safe_filename)
       
       # Save file
       with open(file_path, "wb") as buffer:
           shutil.copyfileobj(file.file, buffer)
       
       return {"filename": safe_filename}
   ```

### Medium Priority Actions

1. **CORS Configuration**
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://freeagentics.com"],  # Specific origins only
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   ```

2. **Content Security Policy**
   ```python
   CSP_POLICY = (
       "default-src 'self'; "
       "script-src 'self' 'unsafe-inline'; "
       "style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: https:; "
       "font-src 'self'; "
       "connect-src 'self'; "
       "frame-ancestors 'none';"
   )
   
   response.headers["Content-Security-Policy"] = CSP_POLICY
   ```

## Running the Tests

### Quick Start

```bash
# Run all penetration tests
python tests/security/run_comprehensive_penetration_tests.py

# Run specific test category
python tests/security/test_file_upload_security.py

# Run with pytest
pytest tests/security/ -v

# Run with specific markers
pytest tests/security/ -m "security" -v
```

### Advanced Usage

```bash
# Generate HTML report only
python tests/security/run_comprehensive_penetration_tests.py --output-format html

# Run quietly and save to custom directory
python tests/security/run_comprehensive_penetration_tests.py --quiet --output-dir /tmp/security-reports

# Run specific security domain tests
pytest tests/security/ -m "authentication" -v
pytest tests/security/ -m "api_security" -v
pytest tests/security/ -m "error_handling" -v
```

### Environment Setup

```bash
# Set production testing environment
export PRODUCTION=true
export SECRET_KEY="production-secret-key"
export JWT_SECRET="production-jwt-secret"
export DATABASE_URL="postgresql://user:pass@localhost/testdb"

# Run tests
python tests/security/run_comprehensive_penetration_tests.py
```

## Report Generation

### Report Types

1. **JSON Report**: Machine-readable detailed results
   - File: `comprehensive_penetration_test_report_YYYYMMDD_HHMMSS.json`
   - Contains: Full test results, metadata, recommendations

2. **HTML Report**: Human-readable formatted report
   - File: `comprehensive_penetration_test_report_YYYYMMDD_HHMMSS.html`
   - Contains: Executive summary, visualizations, recommendations

### Report Structure

```json
{
  "metadata": {
    "test_execution_time": "2024-01-15 14:30:00",
    "test_duration_seconds": 45.2,
    "platform": "FreeAgentics",
    "test_type": "Comprehensive Penetration Testing"
  },
  "executive_summary": {
    "security_status": "MEDIUM RISK - SECURITY IMPROVEMENTS NEEDED",
    "security_level": "MEDIUM",
    "penetration_score": 78.5,
    "production_ready": false,
    "critical_vulnerabilities": 0,
    "high_vulnerabilities": 3,
    "medium_vulnerabilities": 7
  },
  "security_domain_assessment": {
    "authentication_security": {"score": 85, "status": "good"},
    "api_security": {"score": 72, "status": "acceptable"},
    "file_security": {"score": 65, "status": "poor"}
  }
}
```

## Integration with CI/CD

### GitHub Actions Integration

```yaml
name: Security Testing
on: [push, pull_request]

jobs:
  security-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run penetration tests
        run: |
          python tests/security/run_comprehensive_penetration_tests.py --quiet
        env:
          PRODUCTION: true
          SECRET_KEY: ${{ secrets.SECRET_KEY }}
          JWT_SECRET: ${{ secrets.JWT_SECRET }}
      
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: tests/security/*_report_*.html
```

### Quality Gates

Recommended quality gates for different environments:

**Development**:
- Critical issues: 0
- High issues: ≤ 5
- Security score: ≥ 60

**Staging**:
- Critical issues: 0
- High issues: ≤ 2
- Security score: ≥ 80

**Production**:
- Critical issues: 0
- High issues: 0
- Security score: ≥ 90
- Production ready: true

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: security-tests
        name: Security Tests
        entry: python tests/security/run_comprehensive_penetration_tests.py --quiet
        language: system
        always_run: true
        pass_filenames: false
```

## Best Practices

### Security Testing Guidelines

1. **Regular Testing**
   - Run security tests on every commit
   - Perform comprehensive testing weekly
   - Conduct manual penetration testing quarterly

2. **Test Environment Management**
   - Use production-like test environments
   - Never run tests against production
   - Isolate test environments from production networks

3. **Vulnerability Management**
   - Prioritize fixes based on CVSS scores
   - Track remediation progress
   - Validate fixes with regression testing

4. **Documentation**
   - Document all vulnerabilities found
   - Maintain proof-of-concept examples
   - Update remediation guidelines regularly

### False Positive Management

Some tests may generate false positives. Common scenarios:

1. **Development vs Production Configurations**
   - Debug mode may be acceptable in development
   - Adjust test expectations based on environment

2. **Business Requirements**
   - Some file types may be business-critical
   - Implement compensating controls instead of blocking

3. **Framework Limitations**
   - Some security headers may conflict with framework behavior
   - Implement alternative security measures

### Continuous Improvement

1. **Test Coverage Enhancement**
   - Add new tests based on threat modeling
   - Update tests for new attack vectors
   - Incorporate security research findings

2. **Automation Improvements**
   - Reduce false positives through better detection
   - Improve test performance
   - Enhance reporting capabilities

3. **Integration Enhancements**
   - Better CI/CD integration
   - Integration with security tools (SAST, DAST)
   - Integration with vulnerability management systems

## Conclusion

This penetration testing framework provides comprehensive security validation for the FreeAgentics platform. Regular use of these tests, combined with proper remediation of identified vulnerabilities, will significantly improve the security posture of the application.

For questions or additional security testing requirements, please refer to the security team or create an issue in the project repository.

---

**Last Updated**: 2024-01-15
**Version**: 1.0.0
**Maintained By**: FreeAgentics Security Team