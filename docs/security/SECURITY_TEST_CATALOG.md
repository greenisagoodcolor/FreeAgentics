# Security Test Catalog

## Overview

This catalog provides a comprehensive reference for all security tests in the FreeAgentics project, organized by category with detailed descriptions, test objectives, and implementation guidance.

## Test Categories

### 1. Authentication Security Tests

#### 1.1 Login Flow Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_login_timing_attack_prevention` | `test_comprehensive_auth_security.py` | Prevent timing-based username enumeration | High |
| `test_account_lockout_protection` | `test_comprehensive_auth_security.py` | Validate brute force protection | Critical |
| `test_password_complexity_enforcement` | `test_password_security.py` | Ensure strong password requirements | Medium |
| `test_login_rate_limiting` | `test_rate_limiting_comprehensive.py` | Prevent automated login attacks | High |

#### 1.2 JWT Token Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_jwt_token_manipulation` | `test_jwt_manipulation_vulnerabilities.py` | Detect token tampering attempts | Critical |
| `test_jwt_signature_validation` | `test_jwt_manipulation_vulnerabilities.py` | Validate signature verification | Critical |
| `test_jwt_expiration_handling` | `test_jwt_manipulation_vulnerabilities.py` | Ensure proper token expiration | High |
| `test_jwt_algorithm_confusion` | `test_jwt_manipulation_vulnerabilities.py` | Prevent algorithm substitution attacks | High |

#### 1.3 Multi-Factor Authentication
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_mfa_bypass_prevention` | `test_mfa_security.py` | Prevent MFA bypass attempts | Critical |
| `test_totp_validation` | `test_mfa_security.py` | Validate TOTP implementation | High |
| `test_backup_code_security` | `test_mfa_security.py` | Secure backup code handling | Medium |

#### 1.4 Session Management
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_session_fixation_prevention` | `test_session_security.py` | Prevent session fixation attacks | High |
| `test_session_hijacking_protection` | `test_session_security.py` | Detect session hijacking | High |
| `test_session_timeout_enforcement` | `test_session_security.py` | Ensure proper session expiration | Medium |
| `test_concurrent_session_management` | `test_session_security.py` | Manage concurrent user sessions | Medium |

### 2. Authorization Security Tests

#### 2.1 Role-Based Access Control (RBAC)
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_role_permission_matrix` | `test_rbac_authorization_matrix.py` | Validate role-permission mappings | Critical |
| `test_privilege_escalation_prevention` | `test_privilege_escalation_comprehensive.py` | Prevent unauthorized privilege elevation | Critical |
| `test_role_assignment_security` | `test_rbac_authorization_matrix.py` | Secure role assignment processes | High |
| `test_permission_inheritance` | `test_rbac_authorization_matrix.py` | Validate permission inheritance rules | Medium |

#### 2.2 Insecure Direct Object Reference (IDOR)
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_horizontal_privilege_escalation` | `test_idor_validation_suite.py` | Prevent access to peer resources | Critical |
| `test_vertical_privilege_escalation` | `test_idor_validation_suite.py` | Prevent access to higher privilege resources | Critical |
| `test_resource_ownership_validation` | `test_idor_validation_suite.py` | Validate resource ownership checks | High |
| `test_parameter_tampering_detection` | `test_idor_validation_suite.py` | Detect parameter manipulation attempts | High |

#### 2.3 Advanced IDOR Patterns
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_idor_through_references` | `test_idor_advanced_patterns.py` | Detect IDOR via object references | High |
| `test_idor_in_batch_operations` | `test_idor_advanced_patterns.py` | Validate bulk operation security | Medium |
| `test_idor_via_search_filters` | `test_idor_advanced_patterns.py` | Prevent IDOR through search queries | Medium |

#### 2.4 File Operation Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_file_access_authorization` | `test_idor_file_operations.py` | Validate file access permissions | High |
| `test_file_upload_security` | `test_idor_file_operations.py` | Secure file upload handling | High |
| `test_file_download_authorization` | `test_idor_file_operations.py` | Control file download access | Medium |

### 3. Input Validation Security Tests

#### 3.1 SQL Injection Prevention
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_sql_injection_prevention` | `test_injection_prevention.py` | Prevent SQL injection attacks | Critical |
| `test_blind_sql_injection_prevention` | `test_injection_prevention.py` | Detect blind SQL injection attempts | Critical |
| `test_second_order_sql_injection` | `test_injection_prevention.py` | Prevent stored SQL injection | High |
| `test_sql_injection_in_search` | `test_injection_prevention.py` | Validate search parameter security | High |

#### 3.2 NoSQL Injection Prevention
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_nosql_injection_prevention` | `test_nosql_injection_prevention.py` | Prevent NoSQL injection attacks | High |
| `test_mongodb_injection_prevention` | `test_nosql_injection_prevention.py` | Specific MongoDB injection tests | High |
| `test_redis_injection_prevention` | `test_nosql_injection_prevention.py` | Redis command injection prevention | Medium |

#### 3.3 Cross-Site Scripting (XSS) Prevention
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_reflected_xss_prevention` | `test_xss_prevention.py` | Prevent reflected XSS attacks | High |
| `test_stored_xss_prevention` | `test_xss_prevention.py` | Prevent stored XSS attacks | High |
| `test_dom_xss_prevention` | `test_xss_prevention.py` | Prevent DOM-based XSS | Medium |
| `test_xss_in_api_responses` | `test_xss_prevention.py` | Validate API response encoding | Medium |

#### 3.4 Command Injection Prevention
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_command_injection_prevention` | `test_command_injection_prevention.py` | Prevent OS command injection | Critical |
| `test_path_traversal_prevention` | `test_command_injection_prevention.py` | Prevent directory traversal attacks | High |
| `test_file_inclusion_prevention` | `test_command_injection_prevention.py` | Prevent file inclusion vulnerabilities | High |

#### 3.5 XML Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_xxe_prevention` | `test_xml_security.py` | Prevent XML External Entity attacks | High |
| `test_xml_bomb_prevention` | `test_xml_security.py` | Prevent XML bomb attacks | Medium |
| `test_xpath_injection_prevention` | `test_xml_security.py` | Prevent XPath injection | Medium |

### 4. Rate Limiting and DDoS Protection Tests

#### 4.1 API Rate Limiting
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_api_rate_limiting` | `test_rate_limiting_comprehensive.py` | Validate API rate limits | High |
| `test_per_user_rate_limiting` | `test_rate_limiting_comprehensive.py` | User-specific rate limiting | Medium |
| `test_endpoint_specific_limits` | `test_rate_limiting_comprehensive.py` | Different limits per endpoint | Medium |
| `test_rate_limit_bypass_prevention` | `test_rate_limiting_comprehensive.py` | Prevent rate limit circumvention | High |

#### 4.2 Authentication Rate Limiting
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_login_attempt_limiting` | `test_auth_rate_limiting.py` | Limit login attempts | Critical |
| `test_password_reset_limiting` | `test_auth_rate_limiting.py` | Limit password reset requests | High |
| `test_registration_rate_limiting` | `test_auth_rate_limiting.py` | Limit account registration | Medium |

#### 4.3 DDoS Protection
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_connection_flood_protection` | `test_ddos_protection.py` | Protect against connection floods | High |
| `test_request_size_limiting` | `test_ddos_protection.py` | Limit request payload sizes | Medium |
| `test_slowloris_protection` | `test_ddos_protection.py` | Protect against slowloris attacks | Medium |

#### 4.4 WebSocket Rate Limiting
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_websocket_connection_limiting` | `test_websocket_rate_limiting.py` | Limit WebSocket connections | Medium |
| `test_websocket_message_limiting` | `test_websocket_rate_limiting.py` | Limit WebSocket message rates | Medium |

### 5. Brute Force Protection Tests

#### 5.1 Account Protection
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_account_lockout_mechanism` | `test_brute_force_protection.py` | Account lockout after failures | Critical |
| `test_progressive_delays` | `test_brute_force_protection.py` | Increasing delays between attempts | High |
| `test_captcha_integration` | `test_brute_force_protection.py` | CAPTCHA challenge integration | Medium |

#### 5.2 Advanced Brute Force Scenarios
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_distributed_brute_force` | `test_brute_force_advanced_scenarios.py` | Distributed attack detection | High |
| `test_credential_stuffing_protection` | `test_brute_force_advanced_scenarios.py` | Protect against credential stuffing | High |
| `test_password_spraying_detection` | `test_brute_force_advanced_scenarios.py` | Detect password spraying attacks | Medium |

### 6. Cryptographic Security Tests

#### 6.1 Password Hashing
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_password_hashing_strength` | `test_crypto_security.py` | Validate hashing algorithm strength | Critical |
| `test_salt_generation` | `test_crypto_security.py` | Ensure proper salt usage | High |
| `test_hash_timing_consistency` | `test_crypto_security.py` | Consistent hashing timing | Medium |

#### 6.2 Encryption Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_data_encryption_at_rest` | `test_encryption_security.py` | Validate data encryption | Critical |
| `test_encryption_key_management` | `test_encryption_security.py` | Secure key management practices | Critical |
| `test_encryption_algorithm_strength` | `test_encryption_security.py` | Use strong encryption algorithms | High |

#### 6.3 TLS/SSL Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_tls_configuration` | `test_tls_security.py` | Validate TLS configuration | High |
| `test_certificate_validation` | `test_tls_security.py` | Proper certificate validation | High |
| `test_ssl_cipher_suites` | `test_tls_security.py` | Secure cipher suite selection | Medium |

### 7. API Security Tests

#### 7.1 REST API Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_api_authentication` | `test_api_security.py` | API authentication mechanisms | Critical |
| `test_api_authorization` | `test_api_security.py` | API authorization checks | Critical |
| `test_api_input_validation` | `test_api_security.py` | API input validation | High |
| `test_api_error_handling` | `test_api_security.py` | Secure API error responses | Medium |

#### 7.2 GraphQL Security (if applicable)
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_graphql_query_complexity` | `test_graphql_security.py` | Limit query complexity | High |
| `test_graphql_depth_limiting` | `test_graphql_security.py` | Prevent deep nested queries | Medium |
| `test_graphql_introspection_security` | `test_graphql_security.py` | Control schema introspection | Medium |

#### 7.3 WebSocket Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_websocket_authentication` | `test_websocket_security.py` | WebSocket authentication | High |
| `test_websocket_authorization` | `test_websocket_security.py` | WebSocket message authorization | High |
| `test_websocket_origin_validation` | `test_websocket_security.py` | Validate WebSocket origins | Medium |

### 8. Security Headers and Configuration Tests

#### 8.1 HTTP Security Headers
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_security_headers_presence` | `test_security_headers.py` | Ensure security headers are set | High |
| `test_csp_header_validation` | `test_security_headers.py` | Content Security Policy validation | High |
| `test_hsts_header_validation` | `test_security_headers.py` | HTTP Strict Transport Security | Medium |
| `test_x_frame_options` | `test_security_headers.py` | X-Frame-Options header | Medium |

#### 8.2 CORS Configuration
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_cors_configuration` | `test_cors_security.py` | Validate CORS settings | High |
| `test_cors_preflight_handling` | `test_cors_security.py` | Proper preflight request handling | Medium |
| `test_cors_origin_validation` | `test_cors_security.py` | Validate allowed origins | Medium |

### 9. Infrastructure Security Tests

#### 9.1 Container Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_container_vulnerabilities` | `test_container_security.py` | Scan for container vulnerabilities | High |
| `test_container_configuration` | `test_container_security.py` | Validate container security config | Medium |
| `test_container_secrets_management` | `test_container_security.py` | Secure secrets in containers | High |

#### 9.2 Database Security
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_database_access_controls` | `test_database_security.py` | Database access permissions | Critical |
| `test_database_encryption` | `test_database_security.py` | Database encryption at rest | High |
| `test_database_connection_security` | `test_database_security.py` | Secure database connections | High |

### 10. Compliance Tests

#### 10.1 OWASP Top 10 Compliance
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_owasp_a01_broken_access_control` | `test_owasp_validation.py` | A01: Broken Access Control | Critical |
| `test_owasp_a02_cryptographic_failures` | `test_owasp_validation.py` | A02: Cryptographic Failures | Critical |
| `test_owasp_a03_injection` | `test_owasp_validation.py` | A03: Injection | Critical |
| `test_owasp_a04_insecure_design` | `test_owasp_validation.py` | A04: Insecure Design | High |
| `test_owasp_a05_security_misconfiguration` | `test_owasp_validation.py` | A05: Security Misconfiguration | High |
| `test_owasp_a06_vulnerable_components` | `test_owasp_validation.py` | A06: Vulnerable Components | High |
| `test_owasp_a07_identification_auth_failures` | `test_owasp_validation.py` | A07: Auth Failures | Critical |
| `test_owasp_a08_software_data_integrity` | `test_owasp_validation.py` | A08: Software/Data Integrity | Medium |
| `test_owasp_a09_security_logging_monitoring` | `test_owasp_validation.py` | A09: Logging/Monitoring | Medium |
| `test_owasp_a10_ssrf` | `test_owasp_validation.py` | A10: SSRF | High |

#### 10.2 Privacy Compliance
| Test Name | File | Purpose | Risk Level |
|-----------|------|---------|------------|
| `test_gdpr_data_encryption` | `test_privacy_compliance.py` | GDPR data encryption requirements | High |
| `test_gdpr_right_to_erasure` | `test_privacy_compliance.py` | GDPR right to be forgotten | High |
| `test_gdpr_data_portability` | `test_privacy_compliance.py` | GDPR data portability | Medium |
| `test_gdpr_consent_management` | `test_privacy_compliance.py` | GDPR consent tracking | Medium |

## Test Execution Commands

### Run All Security Tests
```bash
# Complete security test suite
python -m pytest tests/security/ -v

# Run with coverage report
python -m pytest tests/security/ --cov=. --cov-report=html
```

### Run Tests by Category
```bash
# Authentication tests
python -m pytest tests/security/test_comprehensive_auth_security.py -v

# Authorization tests
python -m pytest tests/security/test_rbac_authorization_matrix.py tests/security/test_idor_validation_suite.py -v

# Input validation tests
python -m pytest tests/security/test_injection_prevention.py -v

# Rate limiting tests
python -m pytest tests/security/test_rate_limiting_comprehensive.py -v

# Compliance tests
python -m pytest tests/security/test_owasp_validation.py -v
```

### Run Specific Risk Level Tests
```bash
# Critical risk tests only
python -m pytest tests/security/ -m "critical" -v

# High and critical risk tests
python -m pytest tests/security/ -m "critical or high" -v
```

## Test Maintenance

### Adding New Security Tests
1. Choose appropriate test file based on category
2. Follow existing test naming conventions
3. Include proper risk level markers
4. Add comprehensive docstrings
5. Update this catalog with new test information

### Test Review Schedule
- **Monthly**: Review and update critical risk tests
- **Quarterly**: Review all security tests for relevance
- **Annually**: Complete security test strategy review

### Performance Considerations
- Use `@pytest.mark.slow` for tests taking >5 seconds
- Use `@pytest.mark.performance` for benchmark tests
- Consider test parallelization for large test suites

## Risk Assessment Matrix

| Risk Level | Definition | Response Time | Review Frequency |
|------------|------------|---------------|------------------|
| Critical | Could lead to complete system compromise | Immediate | Daily |
| High | Could lead to significant data breach | Within 24 hours | Weekly |
| Medium | Could lead to limited security impact | Within 1 week | Monthly |
| Low | Minimal security impact | Within 1 month | Quarterly |

This catalog serves as the authoritative reference for all security tests in the FreeAgentics project. Keep it updated as new tests are added or existing tests are modified.