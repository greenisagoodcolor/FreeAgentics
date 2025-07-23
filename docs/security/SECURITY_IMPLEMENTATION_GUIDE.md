# FreeAgentics Security Implementation Guide

## Overview

This document provides comprehensive documentation of the security features implemented in FreeAgentics, including authentication, authorization, rate limiting, security headers, and monitoring capabilities.

## Table of Contents

1. [Authentication System](#authentication-system)
2. [Authorization (RBAC)](#authorization-rbac)
3. [Rate Limiting & DDoS Protection](#rate-limiting--ddos-protection)
4. [Security Headers](#security-headers)
5. [Security Monitoring & Logging](#security-monitoring--logging)
6. [API Security](#api-security)
7. [SSL/TLS Configuration](#ssltls-configuration)
8. [Incident Response](#incident-response)

## Authentication System

### JWT Implementation

The authentication system uses JWT (JSON Web Tokens) with the following features:

- **Algorithm**: RS256 (RSA with SHA-256)
- **Token Expiration**:
  - Access tokens: 30 minutes
  - Refresh tokens: 7 days
- **Key Rotation**: Automatic key rotation every 30 days
- **Token Storage**: HTTPOnly secure cookies with SameSite=Strict

### Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant AuthService
    participant Database
    participant Redis

    Client->>API: POST /auth/login
    API->>AuthService: Validate credentials
    AuthService->>Database: Check user
    Database-->>AuthService: User data
    AuthService->>Redis: Store session
    AuthService-->>API: Generate JWT
    API-->>Client: Set secure cookie
```

### Key Security Features

1. **Password Security**
   - Bcrypt hashing with cost factor 12
   - Password complexity requirements enforced
   - Password history tracking (last 5 passwords)
   - Account lockout after 5 failed attempts

2. **Session Management**
   - Redis-based session storage
   - Automatic session expiration
   - Concurrent session limiting
   - Session invalidation on logout

3. **Multi-Factor Authentication**
   - TOTP support (Time-based One-Time Password)
   - Backup codes generation
   - WebAuthn/FIDO2 support (planned)

## Authorization (RBAC)

### Role-Based Access Control

The system implements a hierarchical RBAC model with the following roles:

| Role | Description | Key Permissions |
|------|-------------|-----------------|
| Admin | Full system access | All permissions |
| Manager | Team management | Create/manage agents, view metrics |
| User | Standard access | Create own agents, view own data |
| Viewer | Read-only access | View public resources |

### Permissions Matrix

| Permission | Admin | Manager | User | Viewer |
|------------|-------|---------|------|--------|
| CREATE_AGENT | ✓ | ✓ | ✓ | ✗ |
| MODIFY_AGENT | ✓ | ✓ | Own only | ✗ |
| DELETE_AGENT | ✓ | ✓ | Own only | ✗ |
| VIEW_AGENTS | ✓ | ✓ | Own only | Public only |
| VIEW_METRICS | ✓ | ✓ | ✗ | ✗ |
| ADMIN_SYSTEM | ✓ | ✗ | ✗ | ✗ |
| CREATE_COALITION | ✓ | ✓ | ✗ | ✗ |

### Resource-Based Access Control

Enhanced authorization with ownership validation:

```python
@require_resource_access(
    resource_type="agent",
    action="modify",
    resource_id_param="agent_id",
    owner_check=True
)
async def update_agent(agent_id: str, current_user: TokenData):
    # Only owner or admin can modify
    pass
```

### Attribute-Based Access Control (ABAC)

Dynamic access control based on attributes:

- User attributes (department, clearance level)
- Resource attributes (sensitivity, classification)
- Environmental attributes (time, location, IP)

## Rate Limiting & DDoS Protection

### Rate Limiting Configuration

Default limits:
- **Anonymous users**: 60 requests/minute
- **Authenticated users**: 300 requests/minute
- **API endpoints**: Custom limits per endpoint

### DDoS Protection Features

1. **Pattern Detection**
   - Rapid 404 detection (10 in 60 seconds)
   - Path scanning detection (15 different paths in 30 seconds)
   - Large request filtering (>10MB)
   - Header size limits (8KB)

2. **Automatic Blocking**
   - IP-based blocking for violations
   - 30-minute default block duration
   - Distributed blocking via Redis

3. **Algorithms Supported**
   - Sliding window (default)
   - Token bucket
   - Fixed window
   - Leaky bucket

### Configuration Example

```yaml
endpoints:
  - path: /api/auth/login
    anonymous:
      max_requests: 5
      window_seconds: 300
    authenticated:
      max_requests: 20
      window_seconds: 300
    priority: 100

ddos_protection:
  max_request_size: 10485760  # 10MB
  max_header_size: 8192       # 8KB
  connection_limit_per_ip: 50
  block_duration_minutes: 30
```

## Security Headers

### Implemented Headers

| Header | Value | Purpose |
|--------|-------|---------|
| Strict-Transport-Security | max-age=31536000; includeSubDomains; preload | Force HTTPS |
| X-Content-Type-Options | nosniff | Prevent MIME sniffing |
| X-Frame-Options | DENY | Prevent clickjacking |
| X-XSS-Protection | 1; mode=block | XSS protection |
| Content-Security-Policy | [Complex CSP] | Content restrictions |
| Referrer-Policy | strict-origin-when-cross-origin | Control referrer |
| Permissions-Policy | [Restrictive policy] | Feature restrictions |

### Content Security Policy

Default CSP configuration:
```
default-src 'self';
script-src 'self' 'nonce-{random}';
style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;
img-src 'self' data: https:;
font-src 'self' data: https://fonts.gstatic.com;
connect-src 'self' wss: https:;
frame-ancestors 'self';
base-uri 'self';
form-action 'self';
object-src 'none';
```

### Certificate Pinning

Mobile app certificate pinning support:
- SHA-256 pin validation
- Backup pin support
- Pin rotation mechanism
- Platform-specific implementations

## Security Monitoring & Logging

### Audit Logging

Comprehensive security event logging:

1. **Authentication Events**
   - Login success/failure
   - Logout events
   - Token refresh
   - Session management

2. **Authorization Events**
   - Access granted/denied
   - Permission checks
   - Privilege escalation attempts

3. **Security Incidents**
   - Rate limit violations
   - Suspicious patterns
   - Attack attempts

### Log Format

```json
{
  "timestamp": "2025-01-16T10:00:00Z",
  "event_type": "login_failure",
  "severity": "warning",
  "user_id": "user123",
  "username": "john.doe",
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "endpoint": "/auth/login",
  "message": "Failed login attempt",
  "details": {
    "reason": "invalid_password",
    "attempt_count": 3
  }
}
```

### Security Metrics

Real-time security metrics available:
- Failed login attempts
- Rate limit violations
- Suspicious IP tracking
- Response time monitoring
- Error rate tracking

## API Security

### Input Validation

- Schema validation using Pydantic
- SQL injection prevention via parameterized queries
- XSS prevention through output encoding
- Command injection protection

### API Authentication

```python
# Bearer token authentication
Authorization: Bearer <jwt_token>

# API key authentication (for services)
X-API-Key: <api_key>
```

### Error Handling

Secure error responses without information leakage:
```json
{
  "error": "Authentication failed",
  "code": "AUTH_001",
  "timestamp": "2025-01-16T10:00:00Z"
}
```

## SSL/TLS Configuration

### Nginx Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name api.freeagentics.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/freeagentics.crt;
    ssl_certificate_key /etc/ssl/private/freeagentics.key;

    # Strong protocols
    ssl_protocols TLSv1.2 TLSv1.3;

    # Strong ciphers
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;

    # Session settings
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
}
```

### Let's Encrypt Integration

Automated certificate renewal:
```bash
certbot certonly --webroot -w /var/www/certbot \
  -d api.freeagentics.com \
  --email security@freeagentics.com \
  --agree-tos \
  --non-interactive
```

## Incident Response

### Response Procedures

1. **Detection**
   - Automated alerts via monitoring
   - Pattern detection algorithms
   - Manual security reviews

2. **Containment**
   - Automatic IP blocking
   - Rate limit enforcement
   - Service isolation

3. **Investigation**
   - Audit log analysis
   - Attack pattern identification
   - Impact assessment

4. **Recovery**
   - Service restoration
   - Security patch deployment
   - Configuration updates

### Contact Information

- Security Team: security@freeagentics.com
- Emergency Contact: +1-555-0123
- PagerDuty: freeagentics.pagerduty.com

## Security Configuration Files

### Environment Variables

```bash
# Security Settings
JWT_SECRET_KEY=<strong-secret>
JWT_ALGORITHM=RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Rate Limiting
REDIS_URL=redis://localhost:6379
RATE_LIMIT_ENABLED=true

# Security Headers
HSTS_MAX_AGE=31536000
CSP_REPORT_URI=/api/security/csp-report

# Monitoring
SECURITY_LOG_LEVEL=INFO
AUDIT_DATABASE_URL=postgresql://audit_user:pass@localhost/audit_db
```

### Docker Security

```dockerfile
# Run as non-root user
USER app:app

# Security scanning
RUN pip install safety bandit
RUN safety check
RUN bandit -r /app

# Minimal attack surface
FROM python:3.11-slim
```

## Testing Security

### Security Test Suite

```bash
# Run security tests
pytest tests/security/ -v

# Penetration testing
python -m tests.security.penetration_test

# OWASP scanning
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t https://api.freeagentics.com
```

### Security Checklist

- [ ] All endpoints require authentication
- [ ] RBAC properly configured
- [ ] Rate limiting active
- [ ] Security headers present
- [ ] SSL/TLS properly configured
- [ ] Audit logging enabled
- [ ] Monitoring alerts configured
- [ ] Incident response plan tested

## Maintenance

### Regular Tasks

1. **Daily**
   - Monitor security alerts
   - Review failed login attempts
   - Check rate limit violations

2. **Weekly**
   - Review audit logs
   - Update security rules
   - Test incident response

3. **Monthly**
   - Security scanning
   - Penetration testing
   - Certificate renewal check

4. **Quarterly**
   - Security audit
   - Policy review
   - Training updates
