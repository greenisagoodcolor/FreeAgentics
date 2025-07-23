# API Security Guidelines

## Overview

This document outlines security best practices and guidelines for developing and maintaining secure APIs in the FreeAgentics system.

## API Authentication

### JWT Token Management

#### Token Structure

```json
{
  "sub": "user_id",
  "username": "john.doe",
  "role": "user",
  "permissions": ["CREATE_AGENT", "VIEW_AGENTS"],
  "exp": 1234567890,
  "iat": 1234567890,
  "jti": "unique_token_id"
}
```

#### Token Validation

```python
from auth.security_implementation import decode_access_token

async def validate_token(token: str):
    try:
        # Decode and validate JWT
        token_data = decode_access_token(token)

        # Check token expiration
        if token_data.exp < datetime.utcnow():
            raise HTTPException(401, "Token expired")

        # Verify user still exists and is active
        user = await get_user(token_data.user_id)
        if not user or not user.is_active:
            raise HTTPException(401, "Invalid user")

        return token_data
    except JWTError:
        raise HTTPException(401, "Invalid token")
```

### API Key Authentication

For service-to-service communication:

```python
async def validate_api_key(api_key: str = Header(alias="X-API-Key")):
    # Validate against hashed API keys in database
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    service = await db.query(ServiceAccount).filter(
        ServiceAccount.api_key_hash == key_hash
    ).first()

    if not service or not service.is_active:
        raise HTTPException(401, "Invalid API key")

    # Check rate limits for service
    if not await check_service_rate_limit(service.id):
        raise HTTPException(429, "Rate limit exceeded")

    return service
```

## Input Validation

### Request Validation

Use Pydantic models for all API inputs:

```python
from pydantic import BaseModel, validator, constr, conint
from typing import Optional
import re

class CreateAgentRequest(BaseModel):
    name: constr(min_length=1, max_length=100)
    template: constr(regex="^[a-zA-Z0-9_-]+$")
    description: Optional[constr(max_length=1000)]
    max_iterations: conint(ge=1, le=100)

    @validator('name')
    def validate_name(cls, v):
        # Prevent XSS in agent names
        if re.search(r'[<>\"\'&]', v):
            raise ValueError("Invalid characters in name")
        return v

    @validator('template')
    def validate_template(cls, v):
        # Ensure template exists and is allowed
        allowed_templates = ['basic', 'advanced', 'custom']
        if v not in allowed_templates:
            raise ValueError(f"Template must be one of {allowed_templates}")
        return v
```

### SQL Injection Prevention

Always use parameterized queries:

```python
# Good - Parameterized query
agent = db.query(Agent).filter(
    Agent.id == agent_id,
    Agent.created_by == user_id
).first()

# Bad - String concatenation
# query = f"SELECT * FROM agents WHERE id = '{agent_id}'"

# For dynamic queries, use SQLAlchemy's text() with bound parameters
from sqlalchemy import text

query = text("SELECT * FROM agents WHERE status = :status")
result = db.execute(query, {"status": status_value})
```

### File Upload Security

```python
from pathlib import Path
import magic
import hashlib

ALLOWED_EXTENSIONS = {'.txt', '.json', '.csv'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

async def validate_upload(file: UploadFile):
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")

    # Validate file type by content
    file_type = magic.from_buffer(contents, mime=True)
    if file_type not in ['text/plain', 'application/json', 'text/csv']:
        raise HTTPException(400, "Invalid file type")

    # Check extension
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file extension")

    # Generate secure filename
    file_hash = hashlib.sha256(contents).hexdigest()
    secure_filename = f"{file_hash}{extension}"

    return contents, secure_filename
```

## Output Security

### Response Sanitization

```python
from html import escape
import bleach

def sanitize_output(data: dict) -> dict:
    """Sanitize user-generated content in responses."""
    sanitized = {}

    for key, value in data.items():
        if isinstance(value, str):
            # Escape HTML entities
            sanitized[key] = escape(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_output(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_output(item) if isinstance(item, dict)
                else escape(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized
```

### Error Response Handling

```python
from enum import Enum

class ErrorCode(str, Enum):
    AUTH_001 = "Authentication failed"
    AUTH_002 = "Insufficient permissions"
    VAL_001 = "Invalid input data"
    VAL_002 = "Resource not found"
    SYS_001 = "Internal server error"

def create_error_response(
    code: ErrorCode,
    status_code: int,
    details: Optional[str] = None
) -> JSONResponse:
    """Create standardized error response without leaking sensitive info."""

    # Log full error details internally
    logger.error(f"Error {code}: {details}")

    # Return sanitized error to client
    return JSONResponse(
        status_code=status_code,
        content={
            "error": code.value,
            "code": code.name,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

## CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

# Production CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.freeagentics.com",
        "https://admin.freeagentics.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=3600  # 1 hour
)
```

## Rate Limiting Implementation

### Per-Endpoint Configuration

```python
from functools import wraps
from api.middleware.rate_limiter import RateLimitConfig

# Decorator for custom rate limits
def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
    key_func: Optional[Callable] = None
):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Custom key function (default to IP)
            if key_func:
                identifier = await key_func(request)
            else:
                identifier = get_client_ip(request)

            # Check rate limit
            limiter = request.app.state.rate_limiter
            config = RateLimitConfig(max_requests, window_seconds)

            allowed, info = await limiter.check_rate_limit(identifier, config)
            if not allowed:
                raise HTTPException(
                    429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(info["retry_after"])}
                )

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# Usage
@router.post("/auth/login")
@rate_limit(max_requests=5, window_seconds=300)  # 5 attempts per 5 minutes
async def login(request: Request, credentials: LoginRequest):
    pass
```

## API Versioning

### Version Management

```python
from fastapi import APIRouter, Header
from typing import Optional

# Version routers
v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

# Version detection
async def get_api_version(
    accept: Optional[str] = Header(None),
    api_version: Optional[str] = Header(None, alias="X-API-Version")
) -> str:
    # Check explicit version header
    if api_version:
        return api_version

    # Check Accept header
    if accept and "version=" in accept:
        version = accept.split("version=")[1].split(";")[0]
        return version

    # Default to latest stable
    return "v1"
```

## Security Testing

### Automated Security Tests

```python
import pytest
from httpx import AsyncClient

class TestAPISecurity:
    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, client: AsyncClient):
        """Test SQL injection attack prevention."""
        malicious_inputs = [
            "'; DROP TABLE agents; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM users--"
        ]

        for payload in malicious_inputs:
            response = await client.get(f"/api/v1/agents/{payload}")
            assert response.status_code in [400, 404]
            assert "error" in response.json()

    @pytest.mark.asyncio
    async def test_xss_prevention(self, client: AsyncClient):
        """Test XSS attack prevention."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg/onload=alert('XSS')>"
        ]

        for payload in xss_payloads:
            response = await client.post(
                "/api/v1/agents",
                json={"name": payload, "template": "basic"}
            )
            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_authentication_required(self, client: AsyncClient):
        """Test endpoints require authentication."""
        protected_endpoints = [
            "/api/v1/agents",
            "/api/v1/users/profile",
            "/api/v1/metrics"
        ]

        for endpoint in protected_endpoints:
            response = await client.get(endpoint)
            assert response.status_code == 401
```

## API Documentation Security

### OpenAPI Security Schemes

```python
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer, APIKeyHeader

app = FastAPI(
    title="FreeAgentics API",
    description="Secure API for agent management",
    docs_url="/api/docs",  # Protect in production
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Security schemes
bearer_scheme = HTTPBearer(
    scheme_name="JWT Authentication",
    description="Enter: Bearer <JWT token>"
)

api_key_scheme = APIKeyHeader(
    name="X-API-Key",
    scheme_name="API Key Authentication"
)

# Protect documentation in production
@app.middleware("http")
async def protect_docs(request: Request, call_next):
    if request.url.path in ["/api/docs", "/api/redoc", "/api/openapi.json"]:
        # Require authentication for API docs in production
        if os.getenv("ENVIRONMENT") == "production":
            auth = request.headers.get("Authorization")
            if not auth or not validate_admin_token(auth):
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)
```

## Monitoring and Alerting

### Security Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Security metrics
auth_attempts = Counter(
    'api_auth_attempts_total',
    'Total authentication attempts',
    ['method', 'result']
)

api_errors = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'status_code', 'error_type']
)

active_sessions = Gauge(
    'api_active_sessions',
    'Number of active user sessions'
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['endpoint', 'method']
)

# Track metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()

    try:
        response = await call_next(request)

        # Track request duration
        duration = time.time() - start_time
        request_duration.labels(
            endpoint=request.url.path,
            method=request.method
        ).observe(duration)

        # Track errors
        if response.status_code >= 400:
            api_errors.labels(
                endpoint=request.url.path,
                status_code=response.status_code,
                error_type="client_error" if response.status_code < 500 else "server_error"
            ).inc()

        return response

    except Exception as e:
        api_errors.labels(
            endpoint=request.url.path,
            status_code=500,
            error_type="exception"
        ).inc()
        raise
```

## Best Practices Summary

1. **Always authenticate and authorize** - Never trust client input
1. **Validate all inputs** - Use strong typing and validation
1. **Sanitize all outputs** - Prevent XSS and injection attacks
1. **Use HTTPS everywhere** - Never transmit sensitive data over HTTP
1. **Implement rate limiting** - Protect against abuse and DDoS
1. **Log security events** - Maintain audit trail for compliance
1. **Keep dependencies updated** - Regular security patches
1. **Test security regularly** - Automated tests and pentesting
1. **Follow least privilege** - Grant minimum necessary permissions
1. **Document security measures** - Keep team informed
