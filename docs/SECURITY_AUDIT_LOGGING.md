# Security Audit Logging Implementation

## Overview

FreeAgentics v0.2 implements comprehensive security audit logging to track all security-related events, detect suspicious activities, and meet compliance requirements.

## Features

### 1. Event Types Tracked

#### Authentication Events

- `LOGIN_SUCCESS` - Successful user login
- `LOGIN_FAILURE` - Failed login attempt
- `LOGOUT` - User logout
- `TOKEN_REFRESH` - JWT token refresh
- `TOKEN_EXPIRED` - Expired token usage attempt
- `TOKEN_INVALID` - Invalid token usage attempt

#### Authorization Events

- `ACCESS_GRANTED` - Successful resource access
- `ACCESS_DENIED` - Denied resource access
- `PERMISSION_CHECK` - Permission validation
- `PRIVILEGE_ESCALATION` - Attempted privilege escalation

#### Security Incidents

- `RATE_LIMIT_EXCEEDED` - Rate limit violations
- `BRUTE_FORCE_DETECTED` - Multiple failed login attempts
- `SQL_INJECTION_ATTEMPT` - SQL injection detected
- `XSS_ATTEMPT` - Cross-site scripting attempt
- `COMMAND_INJECTION_ATTEMPT` - Command injection detected
- `SUSPICIOUS_PATTERN` - Other suspicious activity

#### System Events

- `USER_CREATED` - New user registration
- `USER_DELETED` - User account deletion
- `USER_MODIFIED` - User account changes
- `PASSWORD_CHANGED` - Password updates
- `SECURITY_CONFIG_CHANGE` - Security configuration changes

### 2. Information Captured

For each security event, the following is logged:

- **Timestamp** - Exact time of event
- **Event Type** - Category of security event
- **Severity** - INFO, WARNING, ERROR, or CRITICAL
- **User Information** - User ID and username (if applicable)
- **Request Details**:
  - IP Address (including X-Forwarded-For support)
  - User Agent
  - HTTP Method and Endpoint
  - Request ID for correlation
- **Response Status** - HTTP status code
- **Event Message** - Human-readable description
- **Additional Details** - JSON structure with event-specific data

### 3. Storage

Security events are stored in two locations:

1. **Log Files**: `logs/security_audit.log`
   - Structured JSON format
   - Separate from application logs
   - Suitable for log aggregation tools

2. **Database**: Separate audit database table
   - Indexed for fast queries
   - Supports filtering and analysis
   - Configurable via `AUDIT_DATABASE_URL`

### 4. Automatic Threat Detection

The system automatically detects and alerts on:

#### Brute Force Attacks

- Tracks failed login attempts by IP and username
- Triggers alert after 5 failures in 15 minutes
- Adds IP to suspicious list

#### Rate Limit Abuse

- Monitors rate limit violations
- Alerts after 10 violations in 1 hour
- Helps identify API abuse

#### Injection Attacks

- Immediate alerts for SQL/XSS/Command injection attempts
- IPs automatically flagged as suspicious

### 5. Security Monitoring API

Admin users can access security data via API endpoints:

#### GET /api/v1/security/summary

Returns aggregated security metrics:

```json
{
  "total_events": 142,
  "by_type": {
    "login_success": 45,
    "login_failure": 12,
    "access_denied": 3
  },
  "by_severity": {
    "info": 120,
    "warning": 18,
    "error": 3,
    "critical": 1
  },
  "failed_logins": 12,
  "suspicious_ips": ["192.168.1.100"],
  "top_ips": {
    "192.168.1.50": 89,
    "192.168.1.100": 23
  }
}
```

#### GET /api/v1/security/events

Query security events with filters:

- `event_type` - Filter by event type
- `severity` - Filter by severity level
- `user_id` - Filter by user
- `ip_address` - Filter by IP
- `hours` - Time range (max 168 hours)
- `limit` - Max events to return

#### GET /api/v1/security/suspicious-activity

View current suspicious activity tracking:

```json
{
  "suspicious_ips": ["192.168.1.100"],
  "failed_login_tracking": {
    "192.168.1.100:admin": 5
  },
  "rate_limit_violations": {
    "192.168.1.200": 12
  }
}
```

## Configuration

### Environment Variables

```bash
# Audit database (defaults to main DATABASE_URL if not set)
AUDIT_DATABASE_URL=postgresql://user:pass@host:5432/audit_db

# Log directory
LOG_DIR=./logs

# Alert thresholds
BRUTE_FORCE_THRESHOLD=5
BRUTE_FORCE_WINDOW_MINUTES=15
RATE_LIMIT_ABUSE_THRESHOLD=10
RATE_LIMIT_ABUSE_WINDOW_HOURS=1
```

### Integration Points

1. **Authentication**: Automatically logs all login/logout events
2. **Authorization**: Logs permission checks and access denials
3. **API Middleware**: Tracks slow requests and errors
4. **Rate Limiting**: Logs all rate limit violations
5. **Input Validation**: Logs injection attempts

## Usage Examples

### Manual Event Logging

```python
from auth.security_logging import security_auditor, SecurityEventType, SecurityEventSeverity

# Log a custom security event
security_auditor.log_event(
    SecurityEventType.SECURITY_CONFIG_CHANGE,
    SecurityEventSeverity.WARNING,
    "Admin changed rate limit settings",
    request=request,
    user_id=current_user.user_id,
    username=current_user.username,
    details={
        "old_limit": 100,
        "new_limit": 200,
        "endpoint": "/api/v1/agents"
    }
)
```

### Convenience Functions

```python
from auth.security_logging import log_login_success, log_access_denied

# Log successful login
log_login_success(username, user_id, request)

# Log access denied
log_access_denied(user_id, username, "/api/v1/agents", "CREATE_AGENT", request)
```

## Security Best Practices

1. **Regular Review**: Review security logs daily for suspicious patterns
2. **Alert Response**: Investigate all CRITICAL severity events immediately
3. **Log Retention**: Keep security logs for at least 90 days
4. **Access Control**: Limit security log access to authorized personnel only
5. **Integration**: Forward logs to SIEM for centralized monitoring

## Compliance

This implementation helps meet compliance requirements for:

- **GDPR**: User activity tracking and audit trails
- **HIPAA**: Access logging for protected resources
- **SOC 2**: Security event monitoring and alerting
- **PCI DSS**: Failed access attempt tracking

## Monitoring Integration

The security logs are designed to integrate with:

- **ELK Stack**: JSON format for easy parsing
- **Splunk**: Structured event data
- **Datadog**: Metrics and alerting
- **Grafana**: Security dashboards
- **CloudWatch**: AWS integration

## Future Enhancements

1. **Machine Learning**: Anomaly detection for unusual patterns
2. **GeoIP Integration**: Location-based threat detection
3. **Webhook Alerts**: Real-time notifications to Slack/PagerDuty
4. **Automated Response**: Auto-block suspicious IPs
5. **Forensics Tools**: Advanced log analysis capabilities
