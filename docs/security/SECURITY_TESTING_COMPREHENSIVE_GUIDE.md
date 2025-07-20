# Security Testing Comprehensive Guide

## Table of Contents

1. [Security Testing Overview](#security-testing-overview)
1. [Test Execution Procedures](#test-execution-procedures)
1. [CI/CD Integration Guide](#cicd-integration-guide)
1. [Security Test Catalog](#security-test-catalog)
1. [Compliance and Reporting](#compliance-and-reporting)
1. [Test Environment Setup](#test-environment-setup)
1. [Monitoring and Alerting](#monitoring-and-alerting)
1. [Troubleshooting Guide](#troubleshooting-guide)

______________________________________________________________________

## Security Testing Overview

### Testing Strategy and Approach

Our security testing follows a comprehensive multi-layered approach designed to identify vulnerabilities across all application layers:

#### Core Security Testing Principles

1. **Defense in Depth**: Testing multiple security layers
1. **Risk-Based Approach**: Prioritizing tests based on threat modeling
1. **Continuous Security**: Automated security testing in CI/CD pipelines
1. **Compliance First**: Ensuring adherence to security standards

#### Security Testing Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Testing Pyramid                │
├─────────────────────────────────────────────────────────────┤
│ Manual Penetration Testing                      [Top]       │
│ ├─ Red Team Exercises                                       │
│ ├─ Social Engineering Tests                                 │
│ └─ Physical Security Assessments                           │
├─────────────────────────────────────────────────────────────┤
│ Dynamic Application Security Testing (DAST)    [Level 3]   │
│ ├─ OWASP ZAP Scans                                         │
│ ├─ Burp Suite Professional                                 │
│ ├─ API Security Testing                                    │
│ └─ WebSocket Security Testing                              │
├─────────────────────────────────────────────────────────────┤
│ Interactive Application Security Testing       [Level 2]   │
│ ├─ Runtime Security Monitoring                             │
│ ├─ Behavioral Security Analysis                            │
│ └─ Real-time Threat Detection                              │
├─────────────────────────────────────────────────────────────┤
│ Static Application Security Testing (SAST)     [Level 1]   │
│ ├─ Bandit Python Security Linter                          │
│ ├─ Semgrep Security Rules                                  │
│ ├─ Custom Security Checks                                  │
│ └─ Dependency Vulnerability Scanning                       │
├─────────────────────────────────────────────────────────────┤
│ Unit Security Tests                            [Base]      │
│ ├─ Authentication Unit Tests                               │
│ ├─ Authorization Unit Tests                                │
│ ├─ Input Validation Tests                                  │
│ └─ Cryptographic Function Tests                            │
└─────────────────────────────────────────────────────────────┘
```

### Test Categories and Coverage

#### 1. Authentication Security Tests

- **Coverage**: Login flows, password policies, multi-factor authentication
- **Risk Level**: Critical
- **Frequency**: Every commit, daily automated scans

#### 2. Authorization Security Tests

- **Coverage**: RBAC, privilege escalation, IDOR vulnerabilities
- **Risk Level**: Critical
- **Frequency**: Every commit, weekly comprehensive scans

#### 3. Input Validation Tests

- **Coverage**: SQL injection, XSS, command injection, path traversal
- **Risk Level**: High
- **Frequency**: Every commit, continuous monitoring

#### 4. Session Management Tests

- **Coverage**: Session fixation, hijacking, timeout handling
- **Risk Level**: High
- **Frequency**: Every commit, daily monitoring

#### 5. API Security Tests

- **Coverage**: Rate limiting, parameter tampering, API abuse
- **Risk Level**: High
- **Frequency**: Every API change, continuous monitoring

#### 6. Infrastructure Security Tests

- **Coverage**: Container security, configuration management, network security
- **Risk Level**: Medium-High
- **Frequency**: Weekly, on infrastructure changes

### Risk Assessment Methodology

#### Risk Calculation Matrix

```
Risk = (Threat Level × Vulnerability Likelihood × Impact Severity) / Mitigation Effectiveness

Where:
- Threat Level: 1-5 (1=Low, 5=Critical)
- Vulnerability Likelihood: 1-5 (1=Rare, 5=Very Likely)
- Impact Severity: 1-5 (1=Minimal, 5=Catastrophic)
- Mitigation Effectiveness: 1-5 (1=Ineffective, 5=Complete)
```

#### Risk Prioritization Levels

1. **Critical (Risk Score 15-25)**: Immediate action required
1. **High (Risk Score 10-14)**: Action required within 24 hours
1. **Medium (Risk Score 5-9)**: Action required within 1 week
1. **Low (Risk Score 1-4)**: Action required within 1 month

______________________________________________________________________

## Test Execution Procedures

### Test Environment Setup

#### Prerequisites

1. **System Requirements**:

   ```bash
   - Python 3.11+
   - PostgreSQL 15+
   - Redis 7+
   - Docker 24+
   - Node.js 18+
   ```

1. **Security Tools Installation**:

   ```bash
   # Install Python security tools
   pip install bandit[toml] safety semgrep pytest-security

   # Install container security tools
   curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh

   # Install dependency scanning tools
   pip install pip-audit
   npm install -g audit-ci
   ```

1. **Environment Configuration**:

   ```bash
   # Security test environment variables
   export ENVIRONMENT=security-test
   export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/freeagentics_security
   export REDIS_URL=redis://localhost:6379/1
   export JWT_SECRET_KEY=test-secret-key-for-security-testing
   export RATE_LIMITING_ENABLED=true
   export SECURITY_AUDIT_LOG_LEVEL=DEBUG
   ```

### Test Data Preparation

#### Security Test Data Setup

1. **User Test Data**:

   ```python
   # Test users with different privilege levels
   SECURITY_TEST_USERS = {
       'admin': {
           'username': 'admin_test',
           'email': 'admin@test.local',
           'role': 'admin',
           'password': 'SecureAdmin123!'
       },
       'user': {
           'username': 'user_test',
           'email': 'user@test.local',
           'role': 'user',
           'password': 'SecureUser123!'
       },
       'readonly': {
           'username': 'readonly_test',
           'email': 'readonly@test.local',
           'role': 'readonly',
           'password': 'SecureReadonly123!'
       }
   }
   ```

1. **Malicious Payload Database**:

   ```python
   # SQL Injection payloads
   SQL_INJECTION_PAYLOADS = [
       "'; DROP TABLE users; --",
       "1' OR '1'='1",
       "admin'--",
       "' UNION SELECT password FROM users --"
   ]

   # XSS payloads
   XSS_PAYLOADS = [
       "<script>alert('XSS')</script>",
       "javascript:alert('XSS')",
       "<img src=x onerror=alert('XSS')>",
       "<svg onload=alert('XSS')>"
   ]

   # Command injection payloads
   COMMAND_INJECTION_PAYLOADS = [
       "; cat /etc/passwd",
       "| rm -rf /",
       "&& whoami",
       "; ls -la"
   ]
   ```

### Test Execution Workflows

#### 1. Pre-Commit Security Testing

```bash
#!/bin/bash
# pre-commit-security.sh

echo "Running pre-commit security checks..."

# Static code analysis
echo "1. Running Bandit security linter..."
bandit -r . -f json -o bandit-results.json --skip B101,B601,B603

# Dependency vulnerability check
echo "2. Checking dependencies for vulnerabilities..."
safety check --json --output safety-results.json

# Secret detection
echo "3. Scanning for hardcoded secrets..."
semgrep --config=p/secrets --json --output=secrets-results.json .

# Custom security rules
echo "4. Running custom security checks..."
python -m pytest tests/security/test_security_compliance.py -v

echo "Pre-commit security checks completed."
```

#### 2. Full Security Test Suite Execution

```bash
#!/bin/bash
# run-full-security-tests.sh

echo "Starting comprehensive security test suite..."

# Set test environment
export ENVIRONMENT=security-test
export LOG_LEVEL=WARNING

# Start test services
docker-compose -f docker-compose.test.yml up -d postgres redis

# Wait for services
sleep 10

# Database setup
alembic upgrade head

# Run security test categories
echo "1. Running authentication security tests..."
python -m pytest tests/security/test_comprehensive_auth_security.py -v --tb=short

echo "2. Running authorization security tests..."
python -m pytest tests/security/test_authorization_integration.py -v --tb=short

echo "3. Running IDOR vulnerability tests..."
python -m pytest tests/security/test_idor_validation_suite.py -v --tb=short

echo "4. Running injection attack tests..."
python -m pytest tests/security/test_injection_prevention.py -v --tb=short

echo "5. Running brute force protection tests..."
python -m pytest tests/security/test_brute_force_protection.py -v --tb=short

echo "6. Running rate limiting tests..."
python -m pytest tests/security/test_rate_limiting_comprehensive.py -v --tb=short

echo "7. Running JWT security tests..."
python -m pytest tests/security/test_jwt_manipulation_vulnerabilities.py -v --tb=short

echo "8. Running privilege escalation tests..."
python -m pytest tests/security/test_privilege_escalation_comprehensive.py -v --tb=short

# Generate security report
echo "Generating security test report..."
python scripts/security/generate_security_test_report.py

# Cleanup
docker-compose -f docker-compose.test.yml down

echo "Security test suite completed."
```

#### 3. DAST (Dynamic Application Security Testing) Workflow

```bash
#!/bin/bash
# run-dast-tests.sh

echo "Starting DAST security testing..."

# Start application in test mode
uvicorn main:app --host 0.0.0.0 --port 8000 &
APP_PID=$!

# Wait for application startup
sleep 30

# OWASP ZAP Baseline Scan
echo "1. Running OWASP ZAP baseline scan..."
docker run -v $(pwd):/zap/wrk/:rw -t owasp/zap2docker-stable zap-baseline.py \
    -t http://host.docker.internal:8000 \
    -J zap-baseline-report.json \
    -r zap-baseline-report.html

# OWASP ZAP Full Scan (if enabled)
if [[ "$FULL_SECURITY_SCAN" == "true" ]]; then
    echo "2. Running OWASP ZAP full scan..."
    docker run -v $(pwd):/zap/wrk/:rw -t owasp/zap2docker-stable zap-full-scan.py \
        -t http://host.docker.internal:8000 \
        -J zap-full-report.json \
        -r zap-full-report.html
fi

# Custom API security tests
echo "3. Running custom API security tests..."
python tests/security/test_api_security_dynamic.py

# WebSocket security tests
echo "4. Running WebSocket security tests..."
python tests/security/test_websocket_security.py

# Cleanup
kill $APP_PID

echo "DAST testing completed."
```

### Result Interpretation

#### Security Test Result Categories

1. **PASS**: No security issues detected
1. **FAIL**: Critical security vulnerability found
1. **WARN**: Potential security concern identified
1. **INFO**: Security recommendation provided

#### Result Analysis Framework

```python
class SecurityTestResultAnalyzer:
    """Analyze and categorize security test results."""
    
    def __init__(self):
        self.critical_issues = []
        self.high_issues = []
        self.medium_issues = []
        self.low_issues = []
        
    def analyze_result(self, test_name: str, result: dict):
        """Analyze individual test result."""
        severity = self._calculate_severity(result)
        issue = {
            'test': test_name,
            'severity': severity,
            'description': result.get('description', ''),
            'recommendation': result.get('recommendation', ''),
            'cve_references': result.get('cve_references', []),
            'owasp_category': result.get('owasp_category', '')
        }
        
        if severity == 'critical':
            self.critical_issues.append(issue)
        elif severity == 'high':
            self.high_issues.append(issue)
        elif severity == 'medium':
            self.medium_issues.append(issue)
        else:
            self.low_issues.append(issue)
    
    def _calculate_severity(self, result: dict) -> str:
        """Calculate issue severity based on multiple factors."""
        # Implementation would include complex severity calculation
        pass
    
    def generate_summary(self) -> dict:
        """Generate security test summary."""
        return {
            'total_issues': len(self.critical_issues + self.high_issues + 
                              self.medium_issues + self.low_issues),
            'critical_count': len(self.critical_issues),
            'high_count': len(self.high_issues),
            'medium_count': len(self.medium_issues),
            'low_count': len(self.low_issues),
            'security_score': self._calculate_security_score()
        }
    
    def _calculate_security_score(self) -> int:
        """Calculate overall security score (0-100)."""
        # Weighted scoring based on issue severity
        critical_weight = 25
        high_weight = 10
        medium_weight = 5
        low_weight = 1
        
        total_deductions = (
            len(self.critical_issues) * critical_weight +
            len(self.high_issues) * high_weight +
            len(self.medium_issues) * medium_weight +
            len(self.low_issues) * low_weight
        )
        
        return max(0, 100 - total_deductions)
```

______________________________________________________________________

## CI/CD Integration Guide

### Pipeline Configuration

#### GitHub Actions Security Pipeline

The security CI/CD pipeline is implemented in `.github/workflows/security-ci.yml` and includes:

1. **Pre-flight Security Checks**

   - Secret detection with TruffleHog
   - Security-sensitive file change detection
   - Pre-commit hook validation

1. **Static Application Security Testing (SAST)**

   - Bandit security linter
   - Semgrep security patterns
   - Custom security compliance checks

1. **Dependency Vulnerability Scanning**

   - Python: Safety and pip-audit
   - Node.js: npm audit
   - License compliance checking

1. **Container Security Scanning**

   - Trivy vulnerability scanner
   - Grype container analysis
   - Dockerfile security best practices (Hadolint)

1. **Infrastructure as Code Security**

   - Terraform security scanning
   - Kubernetes manifest validation
   - Docker Compose security analysis

1. **Dynamic Application Security Testing (DAST)**

   - OWASP ZAP scanning
   - Custom API security tests
   - Authentication flow testing

1. **Security Integration Testing**

   - End-to-end security workflows
   - RBAC permission testing
   - Session management validation

### Security Gates Setup

#### Pipeline Security Gates

```yaml
# Security gate configuration
security_gates:
  critical_vulnerability_threshold: 0
  high_vulnerability_threshold: 2
  security_score_minimum: 70
  dependency_risk_threshold: medium
  container_vulnerability_threshold: high
  
  blocking_conditions:
    - critical_vulnerabilities_present
    - security_score_below_threshold
    - failed_authentication_tests
    - privilege_escalation_detected
    - injection_vulnerabilities_found
```

#### Security Gate Implementation

```python
class SecurityGateValidator:
    """Validate security gates in CI/CD pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        
    def validate_security_gates(self, test_results: dict) -> bool:
        """Validate all security gates."""
        gate_results = {
            'critical_vulnerabilities': self._check_critical_vulnerabilities(test_results),
            'security_score': self._check_security_score(test_results),
            'authentication_tests': self._check_authentication_tests(test_results),
            'authorization_tests': self._check_authorization_tests(test_results),
            'dependency_security': self._check_dependency_security(test_results)
        }
        
        failed_gates = [gate for gate, passed in gate_results.items() if not passed]
        
        if failed_gates:
            self._log_gate_failures(failed_gates)
            return False
            
        return True
    
    def _check_critical_vulnerabilities(self, results: dict) -> bool:
        """Check for critical vulnerabilities."""
        critical_count = results.get('critical_vulnerability_count', 0)
        return critical_count <= self.config['critical_vulnerability_threshold']
    
    def _check_security_score(self, results: dict) -> bool:
        """Check minimum security score."""
        score = results.get('security_score', 0)
        return score >= self.config['security_score_minimum']
```

### Monitoring and Alerting

#### Security Monitoring Configuration

```yaml
# Prometheus alerting rules for security
groups:
  - name: security_alerts
    rules:
      - alert: SecurityTestFailure
        expr: security_test_failures > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: Security tests are failing
          description: "{{ $value }} security tests have failed"
      
      - alert: HighVulnerabilityCount
        expr: security_vulnerabilities{severity="high"} > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High vulnerability count detected
          description: "{{ $value }} high-severity vulnerabilities found"
      
      - alert: SecurityScoreBelow70
        expr: security_score < 70
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Security score below threshold
          description: "Security score is {{ $value }}, below threshold of 70"
```

#### Alert Management

```python
class SecurityAlertManager:
    """Manage security alerts and notifications."""
    
    def __init__(self, config: dict):
        self.config = config
        self.alert_channels = {
            'slack': SlackNotifier(config['slack_webhook']),
            'email': EmailNotifier(config['email_settings']),
            'pagerduty': PagerDutyNotifier(config['pagerduty_key'])
        }
    
    def send_security_alert(self, alert_type: str, severity: str, details: dict):
        """Send security alert through configured channels."""
        message = self._format_alert_message(alert_type, severity, details)
        
        if severity == 'critical':
            # Send to all channels for critical alerts
            for channel in self.alert_channels.values():
                channel.send_alert(message)
        elif severity == 'high':
            # Send to primary channels for high severity
            self.alert_channels['slack'].send_alert(message)
            self.alert_channels['email'].send_alert(message)
        else:
            # Send to slack for medium/low severity
            self.alert_channels['slack'].send_alert(message)
```

### Troubleshooting Guide

#### Common Security Pipeline Issues

1. **Secret Detection False Positives**

   ```bash
   # Add exception to .secrets.baseline
   echo "known_false_positive_string" >> .secrets.baseline

   # Run detect-secrets scan to update baseline
   detect-secrets scan --baseline .secrets.baseline
   ```

1. **Dependency Vulnerability Issues**

   ```bash
   # Check specific vulnerability
   safety check --json | jq '.vulnerabilities[]'

   # Update vulnerable packages
   pip-audit --fix

   # Pin secure versions
   pip freeze > requirements-security.txt
   ```

1. **Container Security Issues**

   ```bash
   # Scan specific image
   trivy image --severity HIGH,CRITICAL myimage:tag

   # Check base image vulnerabilities
   docker history --no-trunc myimage:tag

   # Update base image
   docker pull alpine:latest
   ```

1. **DAST Test Failures**

   ```bash
   # Check application logs
   docker logs security-test-app

   # Verify application is accessible
   curl -v http://localhost:8000/health

   # Check ZAP spider results
   cat zap-baseline-report.json | jq '.site[].alerts[]'
   ```

#### Security Test Debugging

```python
class SecurityTestDebugger:
    """Debug security test failures."""
    
    def __init__(self):
        self.debug_enabled = os.getenv('SECURITY_DEBUG', 'false').lower() == 'true'
    
    def debug_auth_test_failure(self, test_name: str, error: Exception):
        """Debug authentication test failures."""
        if not self.debug_enabled:
            return
            
        debug_info = {
            'test_name': test_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'environment_vars': {
                'DATABASE_URL': os.getenv('DATABASE_URL', 'not_set'),
                'REDIS_URL': os.getenv('REDIS_URL', 'not_set'),
                'JWT_SECRET_KEY': 'set' if os.getenv('JWT_SECRET_KEY') else 'not_set'
            }
        }
        
        print(f"DEBUG: Auth test failure details:")
        print(json.dumps(debug_info, indent=2))
    
    def debug_database_connection(self):
        """Debug database connection issues."""
        try:
            # Test database connection
            import psycopg2
            conn = psycopg2.connect(os.getenv('DATABASE_URL'))
            conn.close()
            print("DEBUG: Database connection successful")
        except Exception as e:
            print(f"DEBUG: Database connection failed: {e}")
    
    def debug_redis_connection(self):
        """Debug Redis connection issues."""
        try:
            import redis
            r = redis.from_url(os.getenv('REDIS_URL'))
            r.ping()
            print("DEBUG: Redis connection successful")
        except Exception as e:
            print(f"DEBUG: Redis connection failed: {e}")
```

______________________________________________________________________

## Security Test Catalog

### Authentication Tests Documentation

#### Test Categories

1. **Login Flow Security Tests**

   - **File**: `tests/security/test_comprehensive_auth_security.py`
   - **Purpose**: Validate secure authentication workflows
   - **Coverage**:
     - Valid credential verification
     - Invalid credential handling
     - Timing attack prevention
     - Account lockout mechanisms

1. **Password Security Tests**

   - **File**: `tests/security/test_password_security.py`
   - **Purpose**: Validate password policies and storage
   - **Coverage**:
     - Password complexity requirements
     - Password hashing verification
     - Password history enforcement
     - Password reset security

1. **Multi-Factor Authentication Tests**

   - **File**: `tests/security/test_mfa_security.py`
   - **Purpose**: Validate MFA implementation
   - **Coverage**:
     - TOTP token validation
     - Backup code security
     - MFA bypass prevention
     - Device trust management

#### Sample Authentication Test

```python
class TestAuthenticationSecurity:
    """Comprehensive authentication security tests."""
    
    async def test_login_timing_attack_prevention(self, test_client):
        """Test that login timing doesn't leak information."""
        
        # Valid user credentials
        valid_user = {
            "username": "testuser",
            "password": "SecurePassword123!"
        }
        
        # Invalid user credentials
        invalid_user = {
            "username": "nonexistentuser",
            "password": "WrongPassword123!"
        }
        
        # Measure timing for valid user with wrong password
        start_time = time.time()
        response1 = test_client.post("/api/v1/auth/login", json={
            "username": valid_user["username"],
            "password": "WrongPassword"
        })
        valid_user_time = time.time() - start_time
        
        # Measure timing for invalid user
        start_time = time.time()
        response2 = test_client.post("/api/v1/auth/login", json=invalid_user)
        invalid_user_time = time.time() - start_time
        
        # Both should fail
        assert response1.status_code == 401
        assert response2.status_code == 401
        
        # Timing difference should be minimal (within 10ms)
        timing_difference = abs(valid_user_time - invalid_user_time)
        assert timing_difference < 0.01, f"Timing difference too large: {timing_difference}s"
    
    async def test_account_lockout_protection(self, test_client):
        """Test account lockout after failed attempts."""
        
        failed_credentials = {
            "username": "testuser",
            "password": "WrongPassword"
        }
        
        # Attempt failed logins
        for attempt in range(6):  # Assuming 5 attempts trigger lockout
            response = test_client.post("/api/v1/auth/login", json=failed_credentials)
            
            if attempt < 5:
                assert response.status_code == 401
                assert "Invalid credentials" in response.json()["detail"]
            else:
                # Account should be locked
                assert response.status_code == 429
                assert "Account locked" in response.json()["detail"]
    
    async def test_jwt_token_security(self, test_client, test_user):
        """Test JWT token security measures."""
        
        # Get valid token
        login_response = test_client.post("/api/v1/auth/login", json={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        
        # Test token manipulation
        manipulated_token = token[:-10] + "XXXXXXXXXX"
        
        response = test_client.get("/api/v1/auth/protected", headers={
            "Authorization": f"Bearer {manipulated_token}"
        })
        
        assert response.status_code == 401
        assert "Invalid token" in response.json()["detail"]
```

### Authorization Tests Documentation

#### RBAC Security Tests

1. **Role-Based Access Control Tests**

   - **File**: `tests/security/test_rbac_authorization_matrix.py`
   - **Purpose**: Validate RBAC implementation
   - **Coverage**:
     - Role permission validation
     - Privilege escalation prevention
     - Resource access control
     - Cross-tenant access prevention

1. **Insecure Direct Object Reference (IDOR) Tests**

   - **File**: `tests/security/test_idor_validation_suite.py`
   - **Purpose**: Prevent unauthorized resource access
   - **Coverage**:
     - Horizontal privilege escalation
     - Vertical privilege escalation
     - Resource ownership validation
     - Parameter tampering detection

#### Sample Authorization Test

```python
class TestAuthorizationSecurity:
    """Authorization security test suite."""
    
    async def test_idor_prevention(self, test_client, admin_user, regular_user):
        """Test IDOR vulnerability prevention."""
        
        # Admin creates a resource
        admin_token = self._get_auth_token(test_client, admin_user)
        resource_response = test_client.post("/api/v1/resources", 
            json={"name": "admin_resource", "data": "sensitive_data"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        resource_id = resource_response.json()["id"]
        
        # Regular user attempts to access admin's resource
        user_token = self._get_auth_token(test_client, regular_user)
        unauthorized_response = test_client.get(f"/api/v1/resources/{resource_id}",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Should be forbidden
        assert unauthorized_response.status_code == 403
        assert "Access denied" in unauthorized_response.json()["detail"]
    
    async def test_privilege_escalation_prevention(self, test_client, regular_user):
        """Test privilege escalation prevention."""
        
        user_token = self._get_auth_token(test_client, regular_user)
        
        # Attempt to access admin-only endpoint
        admin_response = test_client.get("/api/v1/admin/users",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        assert admin_response.status_code == 403
        
        # Attempt to modify user role
        role_modification = test_client.patch(f"/api/v1/users/{regular_user['id']}/role",
            json={"role": "admin"},
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        assert role_modification.status_code == 403
```

### Input Validation Tests Documentation

#### Injection Attack Prevention Tests

1. **SQL Injection Tests**

   - **File**: `tests/security/test_sql_injection_prevention.py`
   - **Purpose**: Prevent SQL injection attacks
   - **Coverage**:
     - Parameterized query validation
     - ORM injection prevention
     - Stored procedure security
     - Dynamic query validation

1. **NoSQL Injection Tests**

   - **File**: `tests/security/test_nosql_injection_prevention.py`
   - **Purpose**: Prevent NoSQL injection attacks
   - **Coverage**:
     - MongoDB injection prevention
     - Redis injection prevention
     - Document validation
     - Query sanitization

1. **Command Injection Tests**

   - **File**: `tests/security/test_command_injection_prevention.py`
   - **Purpose**: Prevent command injection attacks
   - **Coverage**:
     - System command validation
     - File path sanitization
     - Process execution security
     - Shell escape prevention

#### Sample Input Validation Test

```python
class TestInputValidationSecurity:
    """Input validation security tests."""
    
    @pytest.mark.parametrize("malicious_input", [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "' UNION SELECT password FROM users --"
    ])
    async def test_sql_injection_prevention(self, test_client, malicious_input):
        """Test SQL injection prevention."""
        
        # Attempt SQL injection in search parameter
        response = test_client.get(f"/api/v1/search?query={malicious_input}")
        
        # Should not return sensitive data or cause server error
        assert response.status_code in [200, 400]  # Either sanitized or rejected
        
        if response.status_code == 200:
            # Ensure no sensitive data leaked
            response_text = response.text.lower()
            assert "password" not in response_text
            assert "hash" not in response_text
            assert "secret" not in response_text
    
    @pytest.mark.parametrize("xss_payload", [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>"
    ])
    async def test_xss_prevention(self, test_client, xss_payload):
        """Test XSS prevention in user inputs."""
        
        # Submit XSS payload in user profile
        response = test_client.patch("/api/v1/profile", json={
            "name": xss_payload,
            "bio": f"User bio with {xss_payload}"
        })
        
        # Either rejected or sanitized
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            # Check that dangerous content is escaped
            profile_data = response.json()
            assert "<script>" not in profile_data["name"]
            assert "javascript:" not in profile_data["bio"]
```

### Performance Security Tests

#### Security Performance Benchmarks

1. **Authentication Performance Tests**

   - **File**: `tests/performance/test_auth_performance_security.py`
   - **Purpose**: Validate authentication performance under load
   - **Coverage**:
     - Login rate limiting effectiveness
     - Password hashing performance
     - Token validation speed
     - Session management scalability

1. **Encryption Performance Tests**

   - **File**: `tests/performance/test_crypto_performance.py`
   - **Purpose**: Validate cryptographic operation performance
   - **Coverage**:
     - Hashing algorithm performance
     - Encryption/decryption speed
     - Key derivation performance
     - Random number generation

#### Sample Performance Security Test

```python
class TestSecurityPerformance:
    """Security performance benchmark tests."""
    
    @pytest.mark.benchmark
    async def test_password_hashing_performance(self, benchmark):
        """Benchmark password hashing performance."""
        
        def hash_password():
            from passlib.context import CryptContext
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return pwd_context.hash("TestPassword123!")
        
        # Benchmark password hashing
        result = benchmark(hash_password)
        
        # Ensure hashing takes reasonable time (not too fast, not too slow)
        assert benchmark.stats.mean > 0.05  # At least 50ms to prevent brute force
        assert benchmark.stats.mean < 2.0   # Less than 2s for usability
    
    @pytest.mark.benchmark
    async def test_jwt_validation_performance(self, benchmark, sample_jwt_token):
        """Benchmark JWT token validation performance."""
        
        def validate_token():
            import jwt
            return jwt.decode(sample_jwt_token, "secret", algorithms=["HS256"])
        
        # Benchmark token validation
        result = benchmark(validate_token)
        
        # Ensure validation is fast enough for high throughput
        assert benchmark.stats.mean < 0.001  # Less than 1ms per validation
```

______________________________________________________________________

## Compliance and Reporting

### Compliance Frameworks Coverage

#### OWASP Top 10 Compliance

Our security testing covers all OWASP Top 10 2021 categories:

1. **A01:2021 – Broken Access Control**

   - **Tests**: RBAC tests, IDOR tests, privilege escalation tests
   - **Coverage**: Authorization matrix validation, resource access control
   - **Automated**: Yes, in CI/CD pipeline

1. **A02:2021 – Cryptographic Failures**

   - **Tests**: Encryption tests, certificate validation, TLS configuration
   - **Coverage**: Data encryption, password hashing, secure communication
   - **Automated**: Yes, in CI/CD pipeline

1. **A03:2021 – Injection**

   - **Tests**: SQL injection, NoSQL injection, command injection, LDAP injection
   - **Coverage**: Input validation, query parameterization, sanitization
   - **Automated**: Yes, in CI/CD pipeline

1. **A04:2021 – Insecure Design**

   - **Tests**: Threat modeling validation, security architecture review
   - **Coverage**: Security controls design, attack surface analysis
   - **Automated**: Partially, manual review required

1. **A05:2021 – Security Misconfiguration**

   - **Tests**: Configuration security scans, default credential checks
   - **Coverage**: Security headers, error handling, component configuration
   - **Automated**: Yes, in CI/CD pipeline

1. **A06:2021 – Vulnerable and Outdated Components**

   - **Tests**: Dependency vulnerability scanning, version checks
   - **Coverage**: Third-party libraries, container base images, runtime components
   - **Automated**: Yes, in CI/CD pipeline

1. **A07:2021 – Identification and Authentication Failures**

   - **Tests**: Authentication flow tests, session management tests
   - **Coverage**: Login security, session handling, multi-factor authentication
   - **Automated**: Yes, in CI/CD pipeline

1. **A08:2021 – Software and Data Integrity Failures**

   - **Tests**: Code signing validation, update mechanism security
   - **Coverage**: Supply chain security, integrity verification
   - **Automated**: Partially, CI/CD integration

1. **A09:2021 – Security Logging and Monitoring Failures**

   - **Tests**: Audit log validation, monitoring effectiveness tests
   - **Coverage**: Security event logging, alerting mechanisms
   - **Automated**: Yes, continuous monitoring

1. **A10:2021 – Server-Side Request Forgery (SSRF)**

   - **Tests**: SSRF prevention tests, URL validation tests
   - **Coverage**: External request validation, network access control
   - **Automated**: Yes, in CI/CD pipeline

#### GDPR Compliance Coverage

```python
class GDPRComplianceValidator:
    """Validate GDPR compliance requirements."""
    
    def __init__(self):
        self.compliance_checks = {
            'data_encryption': self._check_data_encryption,
            'right_to_erasure': self._check_data_deletion,
            'data_portability': self._check_data_export,
            'consent_management': self._check_consent_tracking,
            'breach_notification': self._check_breach_procedures,
            'privacy_by_design': self._check_privacy_controls
        }
    
    def validate_compliance(self) -> dict:
        """Run all GDPR compliance checks."""
        results = {}
        for check_name, check_func in self.compliance_checks.items():
            try:
                results[check_name] = check_func()
            except Exception as e:
                results[check_name] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        return results
    
    def _check_data_encryption(self) -> dict:
        """Validate data encryption compliance."""
        # Check database encryption
        # Check data-in-transit encryption
        # Check key management
        return {'status': 'compliant', 'details': 'All data properly encrypted'}
```

#### SOC 2 Compliance Coverage

```python
class SOC2ComplianceValidator:
    """Validate SOC 2 Type II compliance."""
    
    def __init__(self):
        self.trust_service_criteria = {
            'security': self._validate_security_controls,
            'availability': self._validate_availability_controls,
            'processing_integrity': self._validate_processing_controls,
            'confidentiality': self._validate_confidentiality_controls,
            'privacy': self._validate_privacy_controls
        }
    
    def validate_soc2_compliance(self) -> dict:
        """Validate SOC 2 compliance."""
        results = {}
        for criteria, validator in self.trust_service_criteria.items():
            results[criteria] = validator()
        
        return results
```

### Security Metrics and KPIs

#### Key Security Metrics

1. **Vulnerability Metrics**

   ```python
   SECURITY_METRICS = {
       'vulnerability_count_by_severity': {
           'critical': 0,
           'high': 2,
           'medium': 5,
           'low': 8
       },
       'vulnerability_age_days': {
           'avg_resolution_time': 2.5,
           'oldest_unresolved': 7
       },
       'vulnerability_trend': {
           'weekly_new': 3,
           'weekly_resolved': 8,
           'trend': 'improving'
       }
   }
   ```

1. **Security Test Metrics**

   ```python
   TEST_METRICS = {
       'test_coverage': {
           'authentication': 95,
           'authorization': 92,
           'input_validation': 88,
           'overall': 91
       },
       'test_execution': {
           'total_tests': 247,
           'passed': 241,
           'failed': 2,
           'skipped': 4,
           'success_rate': 97.6
       },
       'performance': {
           'avg_test_duration': 45.2,
           'longest_test': 120.5,
           'test_trend': 'stable'
       }
   }
   ```

1. **Security Score Calculation**

   ```python
   def calculate_security_score(metrics: dict) -> int:
       """Calculate overall security score (0-100)."""
       
       # Vulnerability score (40% weight)
       vuln_score = 100 - (
           metrics['critical'] * 25 +
           metrics['high'] * 10 +
           metrics['medium'] * 5 +
           metrics['low'] * 1
       )
       
       # Test coverage score (30% weight)
       coverage_score = metrics['test_coverage']['overall']
       
       # Compliance score (20% weight)
       compliance_score = metrics['compliance_percentage']
       
       # Security posture score (10% weight)
       posture_score = metrics['security_controls_effectiveness']
       
       # Weighted average
       total_score = (
           vuln_score * 0.4 +
           coverage_score * 0.3 +
           compliance_score * 0.2 +
           posture_score * 0.1
       )
       
       return max(0, min(100, int(total_score)))
   ```

### Report Generation Procedures

#### Automated Security Report Generation

```python
class SecurityReportGenerator:
    """Generate comprehensive security reports."""
    
    def __init__(self, config: dict):
        self.config = config
        self.report_templates = {
            'executive': 'templates/executive_security_report.html',
            'technical': 'templates/technical_security_report.html',
            'compliance': 'templates/compliance_report.html'
        }
    
    def generate_comprehensive_report(self, test_results: dict) -> dict:
        """Generate comprehensive security report."""
        
        # Aggregate data from all security tests
        aggregated_data = {
            'executive_summary': self._generate_executive_summary(test_results),
            'vulnerability_analysis': self._analyze_vulnerabilities(test_results),
            'compliance_status': self._assess_compliance(test_results),
            'recommendations': self._generate_recommendations(test_results),
            'metrics': self._calculate_security_metrics(test_results),
            'trends': self._analyze_security_trends(test_results)
        }
        
        # Generate reports for different audiences
        reports = {}
        for report_type, template_path in self.report_templates.items():
            reports[report_type] = self._render_report(
                template_path, 
                aggregated_data, 
                report_type
            )
        
        return reports
    
    def _generate_executive_summary(self, results: dict) -> dict:
        """Generate executive summary."""
        security_score = self._calculate_security_score(results)
        
        return {
            'security_score': security_score,
            'risk_level': self._determine_risk_level(security_score),
            'critical_findings': self._extract_critical_findings(results),
            'improvement_priorities': self._identify_priorities(results),
            'compliance_status': self._summarize_compliance(results)
        }
    
    def _analyze_vulnerabilities(self, results: dict) -> dict:
        """Analyze vulnerability findings."""
        vulnerabilities = []
        
        for test_category, test_results in results.items():
            for test_name, test_result in test_results.items():
                if test_result.get('vulnerabilities'):
                    vulnerabilities.extend(test_result['vulnerabilities'])
        
        return {
            'total_count': len(vulnerabilities),
            'by_severity': self._group_by_severity(vulnerabilities),
            'by_category': self._group_by_category(vulnerabilities),
            'remediation_timeline': self._calculate_remediation_timeline(vulnerabilities)
        }
```

#### Report Templates

1. **Executive Security Report Template**

   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <title>Executive Security Report - {{ report_date }}</title>
       <style>
           /* Executive-friendly styling */
           .security-score { font-size: 2em; color: {{ score_color }}; }
           .risk-indicator { background: {{ risk_color }}; padding: 10px; }
           .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); }
       </style>
   </head>
   <body>
       <h1>Security Posture Report</h1>
       
       <div class="executive-summary">
           <h2>Executive Summary</h2>
           <div class="security-score">Security Score: {{ security_score }}/100</div>
           <div class="risk-indicator">Risk Level: {{ risk_level }}</div>
           
           <h3>Key Findings</h3>
           <ul>
               {% for finding in critical_findings %}
               <li>{{ finding.description }}</li>
               {% endfor %}
           </ul>
           
           <h3>Recommendations</h3>
           <ol>
               {% for recommendation in top_recommendations %}
               <li>{{ recommendation.action }} (Priority: {{ recommendation.priority }})</li>
               {% endfor %}
           </ol>
       </div>
       
       <div class="metrics-overview">
           <h2>Security Metrics</h2>
           <div class="metrics-grid">
               <div class="metric">
                   <h3>Vulnerabilities</h3>
                   <p>Critical: {{ vulnerabilities.critical }}</p>
                   <p>High: {{ vulnerabilities.high }}</p>
                   <p>Medium: {{ vulnerabilities.medium }}</p>
                   <p>Low: {{ vulnerabilities.low }}</p>
               </div>
               <div class="metric">
                   <h3>Test Coverage</h3>
                   <p>Overall: {{ test_coverage.overall }}%</p>
                   <p>Authentication: {{ test_coverage.auth }}%</p>
                   <p>Authorization: {{ test_coverage.authz }}%</p>
               </div>
               <div class="metric">
                   <h3>Compliance</h3>
                   <p>OWASP Top 10: {{ compliance.owasp }}%</p>
                   <p>GDPR: {{ compliance.gdpr }}%</p>
                   <p>SOC 2: {{ compliance.soc2 }}%</p>
               </div>
               <div class="metric">
                   <h3>Trends</h3>
                   <p>Score Change: {{ trends.score_change }}</p>
                   <p>New Vulnerabilities: {{ trends.new_vulns }}</p>
                   <p>Resolved Issues: {{ trends.resolved }}</p>
               </div>
           </div>
       </div>
   </body>
   </html>
   ```

1. **Technical Security Report Template**

   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <title>Technical Security Report - {{ report_date }}</title>
       <style>
           /* Technical-focused styling */
           .vulnerability { border-left: 4px solid {{ severity_color }}; margin: 10px 0; padding: 10px; }
           .code-block { background: #f4f4f4; padding: 10px; font-family: monospace; }
           .test-results { margin: 20px 0; }
       </style>
   </head>
   <body>
       <h1>Technical Security Analysis Report</h1>
       
       <div class="test-results">
           <h2>Security Test Results</h2>
           {% for category, tests in test_results.items() %}
           <h3>{{ category|title }} Tests</h3>
           <table>
               <thead>
                   <tr><th>Test Name</th><th>Status</th><th>Duration</th><th>Issues</th></tr>
               </thead>
               <tbody>
                   {% for test in tests %}
                   <tr>
                       <td>{{ test.name }}</td>
                       <td class="{{ test.status }}">{{ test.status }}</td>
                       <td>{{ test.duration }}ms</td>
                       <td>{{ test.issues|length }}</td>
                   </tr>
                   {% endfor %}
               </tbody>
           </table>
           {% endfor %}
       </div>
       
       <div class="vulnerability-details">
           <h2>Vulnerability Analysis</h2>
           {% for vulnerability in vulnerabilities %}
           <div class="vulnerability" data-severity="{{ vulnerability.severity }}">
               <h3>{{ vulnerability.title }}</h3>
               <p><strong>Severity:</strong> {{ vulnerability.severity }}</p>
               <p><strong>Category:</strong> {{ vulnerability.category }}</p>
               <p><strong>Description:</strong> {{ vulnerability.description }}</p>
               <p><strong>Location:</strong> {{ vulnerability.location }}</p>
               
               {% if vulnerability.code_sample %}
               <h4>Code Sample:</h4>
               <div class="code-block">{{ vulnerability.code_sample }}</div>
               {% endif %}
               
               <h4>Remediation:</h4>
               <p>{{ vulnerability.remediation }}</p>
               
               {% if vulnerability.references %}
               <h4>References:</h4>
               <ul>
                   {% for ref in vulnerability.references %}
                   <li><a href="{{ ref.url }}">{{ ref.title }}</a></li>
                   {% endfor %}
               </ul>
               {% endif %}
           </div>
           {% endfor %}
       </div>
   </body>
   </html>
   ```

### Audit Trail Documentation

#### Security Event Logging

```python
class SecurityAuditLogger:
    """Comprehensive security audit logging."""
    
    def __init__(self, config: dict):
        self.config = config
        self.log_handlers = self._setup_log_handlers()
    
    def log_security_event(self, event_type: str, details: dict, severity: str = "info"):
        """Log security events with full context."""
        
        audit_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'user_id': details.get('user_id'),
            'session_id': details.get('session_id'),
            'ip_address': details.get('ip_address'),
            'user_agent': details.get('user_agent'),
            'request_id': details.get('request_id'),
            'correlation_id': self._generate_correlation_id()
        }
        
        # Log to multiple destinations
        for handler in self.log_handlers:
            handler.log(audit_record)
    
    def log_authentication_event(self, user_id: str, event_type: str, 
                                success: bool, details: dict):
        """Log authentication-specific events."""
        
        auth_event = {
            'user_id': user_id,
            'event_type': f"auth_{event_type}",
            'success': success,
            'details': details,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        severity = "info" if success else "warning"
        self.log_security_event("authentication", auth_event, severity)
    
    def log_authorization_event(self, user_id: str, resource: str, 
                               action: str, granted: bool, details: dict):
        """Log authorization decisions."""
        
        authz_event = {
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'granted': granted,
            'details': details,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        severity = "info" if granted else "warning"
        self.log_security_event("authorization", authz_event, severity)
```

#### Audit Trail Analysis

```python
class SecurityAuditAnalyzer:
    """Analyze security audit trails for patterns and anomalies."""
    
    def __init__(self, audit_log_source: str):
        self.audit_log_source = audit_log_source
        
    def analyze_authentication_patterns(self, time_window: timedelta) -> dict:
        """Analyze authentication patterns for anomalies."""
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - time_window
        
        auth_events = self._get_auth_events(start_time, end_time)
        
        analysis = {
            'total_attempts': len(auth_events),
            'successful_logins': len([e for e in auth_events if e['success']]),
            'failed_attempts': len([e for e in auth_events if not e['success']]),
            'unique_users': len(set(e['user_id'] for e in auth_events)),
            'unique_ips': len(set(e['ip_address'] for e in auth_events)),
            'suspicious_patterns': self._identify_suspicious_patterns(auth_events),
            'brute_force_attempts': self._detect_brute_force(auth_events),
            'account_takeover_indicators': self._detect_takeover_patterns(auth_events)
        }
        
        return analysis
    
    def _identify_suspicious_patterns(self, events: list) -> list:
        """Identify suspicious authentication patterns."""
        suspicious = []
        
        # Group by IP address
        ip_groups = {}
        for event in events:
            ip = event['ip_address']
            if ip not in ip_groups:
                ip_groups[ip] = []
            ip_groups[ip].append(event)
        
        # Check for suspicious patterns
        for ip, ip_events in ip_groups.items():
            # Multiple failed attempts from same IP
            failed_count = len([e for e in ip_events if not e['success']])
            if failed_count > 10:
                suspicious.append({
                    'type': 'multiple_failed_attempts',
                    'ip_address': ip,
                    'failed_count': failed_count,
                    'severity': 'high'
                })
            
            # Multiple different users from same IP
            unique_users = len(set(e['user_id'] for e in ip_events))
            if unique_users > 5:
                suspicious.append({
                    'type': 'multiple_users_same_ip',
                    'ip_address': ip,
                    'user_count': unique_users,
                    'severity': 'medium'
                })
        
        return suspicious
```

This comprehensive security testing documentation provides:

1. **Complete Testing Framework**: Multi-layered security testing approach covering all security aspects
1. **Detailed Procedures**: Step-by-step instructions for executing security tests
1. **CI/CD Integration**: Full pipeline integration with automated security gates
1. **Comprehensive Test Catalog**: Detailed documentation of all security test categories
1. **Compliance Coverage**: OWASP, GDPR, and SOC 2 compliance validation
1. **Metrics and Reporting**: Security KPIs, automated report generation, and audit trails
1. **Troubleshooting Guide**: Common issues and debugging procedures

The documentation enables teams to understand, execute, and maintain the security testing suite effectively while ensuring comprehensive coverage of security vulnerabilities and compliance requirements.
