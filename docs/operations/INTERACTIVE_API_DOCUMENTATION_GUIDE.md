# Interactive API Documentation Guide

## Overview

This guide provides comprehensive interactive API documentation for the FreeAgentics system, including authentication, endpoint references, code examples, and testing procedures.

## Table of Contents

1. [API Overview](#api-overview)
1. [Authentication](#authentication)
1. [Core API Endpoints](#core-api-endpoints)
1. [Agent Management API](#agent-management-api)
1. [Coalition Management API](#coalition-management-api)
1. [Monitoring and Health API](#monitoring-and-health-api)
1. [WebSocket API](#websocket-api)
1. [Error Handling](#error-handling)
1. [Rate Limiting](#rate-limiting)
1. [Testing and Examples](#testing-and-examples)

## API Overview

### Base URL

- **Production**: `https://api.freeagentics.io`
- **Staging**: `https://staging-api.freeagentics.io`
- **Development**: `http://localhost:8000`

### API Versioning

- **Current Version**: `v1`
- **Endpoint Pattern**: `/api/v1/{resource}`
- **Header**: `Accept: application/json`

### API Architecture

```
FreeAgentics API
├── Authentication (/auth)
├── Agents (/agents)
├── Coalitions (/coalitions)
├── Inference (/inference)
├── Knowledge (/knowledge)
├── Monitoring (/monitoring)
└── WebSocket (/ws)
```

## Authentication

### 1. JWT Authentication

#### Obtain JWT Token

```bash
# Request JWT token
curl -X POST https://api.freeagentics.io/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your-username",
    "password": "your-password"
  }'
```

**Response:**

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

#### Refresh Token

```bash
# Refresh JWT token
curl -X POST https://api.freeagentics.io/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -d '{
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
  }'
```

### 2. API Key Authentication

#### Using API Keys

```bash
# Authentication with API key
curl -X GET https://api.freeagentics.io/api/v1/agents \
  -H "X-API-Key: your-api-key"
```

#### Create API Key

```bash
# Create new API key
curl -X POST https://api.freeagentics.io/api/v1/auth/api-keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "name": "My API Key",
    "permissions": ["read", "write"],
    "expires_at": "2024-12-31T23:59:59Z"
  }'
```

### 3. OAuth2 Integration

#### OAuth2 Flow

```bash
# Step 1: Authorization URL
https://api.freeagentics.io/api/v1/auth/oauth/authorize?
  client_id=your-client-id&
  redirect_uri=https://your-app.com/callback&
  response_type=code&
  scope=read+write

# Step 2: Exchange code for token
curl -X POST https://api.freeagentics.io/api/v1/auth/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code&
      code=authorization-code&
      client_id=your-client-id&
      client_secret=your-client-secret&
      redirect_uri=https://your-app.com/callback"
```

## Core API Endpoints

### 1. Health Check

#### System Health

```bash
# Check system health
curl -X GET https://api.freeagentics.io/api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "agents": "healthy"
  },
  "uptime": 86400,
  "response_time": 0.045
}
```

#### Detailed Health Check

```bash
# Detailed health information
curl -X GET https://api.freeagentics.io/api/v1/health/detailed \
  -H "Authorization: Bearer your-jwt-token"
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": {
      "status": "healthy",
      "connections": 45,
      "max_connections": 100,
      "response_time": 0.023
    },
    "redis": {
      "status": "healthy",
      "memory_usage": "125MB",
      "max_memory": "1GB",
      "connected_clients": 15
    },
    "agents": {
      "status": "healthy",
      "active_agents": 23,
      "total_agents": 30,
      "success_rate": 0.987
    }
  },
  "metrics": {
    "requests_per_second": 245,
    "error_rate": 0.003,
    "average_response_time": 0.156
  }
}
```

### 2. System Information

#### System Status

```bash
# Get system status
curl -X GET https://api.freeagentics.io/api/v1/system/status \
  -H "Authorization: Bearer your-jwt-token"
```

**Response:**

```json
{
  "system": {
    "name": "FreeAgentics",
    "version": "1.0.0",
    "environment": "production",
    "deployment_time": "2024-01-15T08:00:00Z"
  },
  "infrastructure": {
    "region": "us-east-1",
    "availability_zone": "us-east-1a",
    "instance_type": "t3.large",
    "containers": 5
  },
  "performance": {
    "cpu_usage": 0.65,
    "memory_usage": 0.78,
    "disk_usage": 0.42,
    "network_io": {
      "rx_bytes": 1048576,
      "tx_bytes": 2097152
    }
  }
}
```

## Agent Management API

### 1. Agent CRUD Operations

#### List Agents

```bash
# Get all agents
curl -X GET https://api.freeagentics.io/api/v1/agents \
  -H "Authorization: Bearer your-jwt-token"
```

**Response:**

```json
{
  "agents": [
    {
      "id": "agent-123",
      "name": "Agent Alpha",
      "type": "inference",
      "status": "active",
      "created_at": "2024-01-15T09:00:00Z",
      "last_activity": "2024-01-15T10:29:45Z",
      "capabilities": ["reasoning", "planning", "learning"],
      "performance": {
        "success_rate": 0.95,
        "average_response_time": 0.234,
        "tasks_completed": 1247
      }
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 23,
    "total_pages": 2
  }
}
```

#### Create Agent

```bash
# Create new agent
curl -X POST https://api.freeagentics.io/api/v1/agents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "name": "New Agent",
    "type": "inference",
    "config": {
      "model": "gpt-4",
      "temperature": 0.7,
      "max_tokens": 1000
    },
    "capabilities": ["reasoning", "planning"]
  }'
```

**Response:**

```json
{
  "id": "agent-456",
  "name": "New Agent",
  "type": "inference",
  "status": "initializing",
  "created_at": "2024-01-15T10:30:00Z",
  "config": {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "capabilities": ["reasoning", "planning"]
}
```

#### Get Agent Details

```bash
# Get specific agent
curl -X GET https://api.freeagentics.io/api/v1/agents/agent-123 \
  -H "Authorization: Bearer your-jwt-token"
```

#### Update Agent

```bash
# Update agent configuration
curl -X PUT https://api.freeagentics.io/api/v1/agents/agent-123 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "config": {
      "temperature": 0.8,
      "max_tokens": 1500
    }
  }'
```

#### Delete Agent

```bash
# Delete agent
curl -X DELETE https://api.freeagentics.io/api/v1/agents/agent-123 \
  -H "Authorization: Bearer your-jwt-token"
```

### 2. Agent Operations

#### Execute Agent Task

```bash
# Execute task with agent
curl -X POST https://api.freeagentics.io/api/v1/agents/agent-123/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "task": "analyze_data",
    "input": {
      "data": "sample data for analysis",
      "parameters": {
        "method": "statistical",
        "confidence": 0.95
      }
    }
  }'
```

**Response:**

```json
{
  "task_id": "task-789",
  "status": "processing",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:30:30Z",
  "progress": 0.0
}
```

#### Get Task Status

```bash
# Check task status
curl -X GET https://api.freeagentics.io/api/v1/agents/agent-123/tasks/task-789 \
  -H "Authorization: Bearer your-jwt-token"
```

**Response:**

```json
{
  "task_id": "task-789",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:25Z",
  "progress": 1.0,
  "result": {
    "analysis": "Statistical analysis complete",
    "confidence": 0.97,
    "insights": ["Pattern detected", "Correlation found"],
    "recommendations": ["Increase sample size", "Apply filter"]
  }
}
```

## Coalition Management API

### 1. Coalition Operations

#### Create Coalition

```bash
# Create new coalition
curl -X POST https://api.freeagentics.io/api/v1/coalitions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "name": "Analysis Coalition",
    "purpose": "Data analysis and insights",
    "members": ["agent-123", "agent-456"],
    "coordination_strategy": "consensus",
    "config": {
      "timeout": 300,
      "max_iterations": 10
    }
  }'
```

**Response:**

```json
{
  "id": "coalition-abc",
  "name": "Analysis Coalition",
  "purpose": "Data analysis and insights",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "members": [
    {
      "agent_id": "agent-123",
      "role": "coordinator",
      "joined_at": "2024-01-15T10:30:00Z"
    },
    {
      "agent_id": "agent-456",
      "role": "participant",
      "joined_at": "2024-01-15T10:30:00Z"
    }
  ],
  "coordination_strategy": "consensus",
  "performance": {
    "success_rate": 1.0,
    "average_completion_time": 45.6,
    "tasks_completed": 0
  }
}
```

#### Execute Coalition Task

```bash
# Execute task with coalition
curl -X POST https://api.freeagentics.io/api/v1/coalitions/coalition-abc/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "task": "collaborative_analysis",
    "input": {
      "dataset": "financial_data_2024",
      "analysis_type": "trend_analysis",
      "parameters": {
        "time_period": "Q1_2024",
        "granularity": "daily"
      }
    }
  }'
```

## Monitoring and Health API

### 1. Metrics API

#### Get System Metrics

```bash
# Get system metrics
curl -X GET https://api.freeagentics.io/api/v1/monitoring/metrics \
  -H "Authorization: Bearer your-jwt-token"
```

**Response:**

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "metrics": {
    "system": {
      "cpu_usage": 0.65,
      "memory_usage": 0.78,
      "disk_usage": 0.42,
      "network_io": {
        "rx_bytes_per_sec": 1048576,
        "tx_bytes_per_sec": 2097152
      }
    },
    "application": {
      "requests_per_second": 245,
      "error_rate": 0.003,
      "average_response_time": 0.156,
      "active_connections": 45
    },
    "agents": {
      "active_agents": 23,
      "total_agents": 30,
      "success_rate": 0.987,
      "average_task_time": 2.34
    }
  }
}
```

#### Get Agent Metrics

```bash
# Get agent-specific metrics
curl -X GET https://api.freeagentics.io/api/v1/monitoring/agents/agent-123/metrics \
  -H "Authorization: Bearer your-jwt-token"
```

### 2. Alerts API

#### Get Active Alerts

```bash
# Get active alerts
curl -X GET https://api.freeagentics.io/api/v1/monitoring/alerts \
  -H "Authorization: Bearer your-jwt-token"
```

**Response:**

```json
{
  "alerts": [
    {
      "id": "alert-123",
      "name": "High Memory Usage",
      "severity": "warning",
      "status": "active",
      "triggered_at": "2024-01-15T10:25:00Z",
      "description": "Memory usage above 80%",
      "source": "system",
      "value": 0.85,
      "threshold": 0.8
    }
  ],
  "summary": {
    "total_alerts": 1,
    "critical": 0,
    "warning": 1,
    "info": 0
  }
}
```

## WebSocket API

### 1. WebSocket Connection

#### Connect to WebSocket

```javascript
// JavaScript WebSocket client
const ws = new WebSocket('wss://api.freeagentics.io/ws');

ws.onopen = function(event) {
  console.log('Connected to WebSocket');

  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-jwt-token'
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.onclose = function(event) {
  console.log('WebSocket closed');
};
```

#### WebSocket Message Types

```javascript
// Subscribe to agent events
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'agent_events',
  agent_id: 'agent-123'
}));

// Subscribe to coalition events
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'coalition_events',
  coalition_id: 'coalition-abc'
}));

// Subscribe to system metrics
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'system_metrics',
  interval: 5000
}));
```

### 2. Real-time Updates

#### Agent Status Updates

```json
{
  "type": "agent_status",
  "timestamp": "2024-01-15T10:30:00Z",
  "agent_id": "agent-123",
  "status": "active",
  "data": {
    "task_count": 5,
    "success_rate": 0.96,
    "last_activity": "2024-01-15T10:29:45Z"
  }
}
```

#### Task Progress Updates

```json
{
  "type": "task_progress",
  "timestamp": "2024-01-15T10:30:00Z",
  "task_id": "task-789",
  "agent_id": "agent-123",
  "progress": 0.65,
  "estimated_completion": "2024-01-15T10:30:15Z"
}
```

## Error Handling

### 1. Error Response Format

#### Standard Error Response

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "temperature",
      "reason": "Value must be between 0 and 1"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req-123456"
  }
}
```

### 2. Common Error Codes

#### HTTP Status Codes

- **200 OK**: Success
- **201 Created**: Resource created successfully
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Access denied
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflict
- **422 Unprocessable Entity**: Validation error
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error

#### Application Error Codes

```json
{
  "AUTHENTICATION_ERROR": "Invalid credentials",
  "AUTHORIZATION_ERROR": "Insufficient permissions",
  "VALIDATION_ERROR": "Invalid input data",
  "RESOURCE_NOT_FOUND": "Requested resource not found",
  "RESOURCE_CONFLICT": "Resource already exists",
  "RATE_LIMIT_EXCEEDED": "Too many requests",
  "AGENT_NOT_AVAILABLE": "Agent is not available",
  "COALITION_ERROR": "Coalition operation failed",
  "TASK_TIMEOUT": "Task execution timeout",
  "SYSTEM_ERROR": "Internal system error"
}
```

## Rate Limiting

### 1. Rate Limit Headers

#### Response Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642258800
X-RateLimit-Window: 3600
```

### 2. Rate Limit Policies

#### Default Limits

- **Authenticated Users**: 1000 requests/hour
- **API Keys**: 5000 requests/hour
- **Premium Users**: 10000 requests/hour
- **WebSocket**: 100 messages/minute

#### Custom Rate Limits

```bash
# Check current rate limit
curl -X GET https://api.freeagentics.io/api/v1/auth/rate-limit \
  -H "Authorization: Bearer your-jwt-token"
```

**Response:**

```json
{
  "limit": 1000,
  "remaining": 999,
  "reset": 1642258800,
  "window": 3600,
  "policy": "standard"
}
```

## Testing and Examples

### 1. Code Examples

#### Python SDK Example

```python
# Python SDK usage
import freeagentics

# Initialize client
client = freeagentics.Client(
    api_key='your-api-key',
    base_url='https://api.freeagentics.io'
)

# Create agent
agent = client.agents.create(
    name='Test Agent',
    type='inference',
    config={
        'model': 'gpt-4',
        'temperature': 0.7
    }
)

# Execute task
result = client.agents.execute_task(
    agent_id=agent.id,
    task='analyze_data',
    input={'data': 'sample data'}
)

print(f"Task result: {result}")
```

#### Node.js SDK Example

```javascript
// Node.js SDK usage
const FreeAgentics = require('freeagentics-sdk');

const client = new FreeAgentics({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.freeagentics.io'
});

async function example() {
  // Create agent
  const agent = await client.agents.create({
    name: 'Test Agent',
    type: 'inference',
    config: {
      model: 'gpt-4',
      temperature: 0.7
    }
  });

  // Execute task
  const result = await client.agents.executeTask(agent.id, {
    task: 'analyze_data',
    input: { data: 'sample data' }
  });

  console.log('Task result:', result);
}

example().catch(console.error);
```

### 2. Postman Collection

#### Collection Setup

```json
{
  "info": {
    "name": "FreeAgentics API",
    "description": "Interactive API collection for FreeAgentics",
    "version": "1.0.0"
  },
  "auth": {
    "type": "bearer",
    "bearer": [
      {
        "key": "token",
        "value": "{{jwt_token}}",
        "type": "string"
      }
    ]
  },
  "variable": [
    {
      "key": "base_url",
      "value": "https://api.freeagentics.io",
      "type": "string"
    },
    {
      "key": "jwt_token",
      "value": "",
      "type": "string"
    }
  ]
}
```

### 3. OpenAPI Specification

#### Swagger/OpenAPI Documentation

```yaml
openapi: 3.0.0
info:
  title: FreeAgentics API
  description: Interactive API for FreeAgentics system
  version: 1.0.0
  contact:
    name: API Support
    url: https://docs.freeagentics.io
    email: api-support@freeagentics.io

servers:
  - url: https://api.freeagentics.io/api/v1
    description: Production server
  - url: https://staging-api.freeagentics.io/api/v1
    description: Staging server

security:
  - bearerAuth: []
  - apiKeyAuth: []

paths:
  /health:
    get:
      summary: Health check
      description: Get system health status
      responses:
        '200':
          description: System is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

  /agents:
    get:
      summary: List agents
      description: Get list of all agents
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: per_page
          in: query
          schema:
            type: integer
            default: 20
      responses:
        '200':
          description: List of agents
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AgentListResponse'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    apiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

  schemas:
    HealthResponse:
      type: object
      properties:
        status:
          type: string
          example: healthy
        timestamp:
          type: string
          format: date-time
        version:
          type: string
          example: 1.0.0
```

### 4. Testing Tools

#### API Testing Script

```bash
#!/bin/bash
# API testing script

BASE_URL="https://api.freeagentics.io/api/v1"
JWT_TOKEN="your-jwt-token"

# Test health endpoint
echo "Testing health endpoint..."
curl -s -X GET "$BASE_URL/health" | jq .

# Test authentication
echo "Testing authentication..."
curl -s -X POST "$BASE_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}' | jq .

# Test agents endpoint
echo "Testing agents endpoint..."
curl -s -X GET "$BASE_URL/agents" \
  -H "Authorization: Bearer $JWT_TOKEN" | jq .

# Test agent creation
echo "Testing agent creation..."
curl -s -X POST "$BASE_URL/agents" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "name": "Test Agent",
    "type": "inference",
    "config": {"model": "gpt-4", "temperature": 0.7}
  }' | jq .

echo "API testing complete!"
```

## Interactive Documentation Tools

### 1. Swagger UI Integration

#### Swagger UI Setup

```html
<!DOCTYPE html>
<html>
<head>
  <title>FreeAgentics API Documentation</title>
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui.css" />
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({
      url: '/api/v1/openapi.json',
      dom_id: '#swagger-ui',
      presets: [
        SwaggerUIBundle.presets.apis,
        SwaggerUIBundle.presets.standalone
      ],
      plugins: [
        SwaggerUIBundle.plugins.DownloadUrl
      ]
    });
  </script>
</body>
</html>
```

### 2. API Explorer

#### Interactive API Explorer

```javascript
// Interactive API explorer component
class APIExplorer {
  constructor(baseUrl, authToken) {
    this.baseUrl = baseUrl;
    this.authToken = authToken;
  }

  async makeRequest(endpoint, method = 'GET', data = null) {
    const config = {
      method: method,
      headers: {
        'Authorization': `Bearer ${this.authToken}`,
        'Content-Type': 'application/json'
      }
    };

    if (data) {
      config.body = JSON.stringify(data);
    }

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, config);
      const result = await response.json();
      return { status: response.status, data: result };
    } catch (error) {
      return { status: 'error', data: error.message };
    }
  }

  // Example usage
  async testEndpoint(endpoint, method, data) {
    console.log(`Testing ${method} ${endpoint}`);
    const result = await this.makeRequest(endpoint, method, data);
    console.log('Response:', result);
    return result;
  }
}
```

______________________________________________________________________

**Document Information:**

- **Version**: 1.0
- **Last Updated**: January 2024
- **Next Review**: April 2024
- **Owner**: API Team
- **Approved By**: Engineering Lead

**API Endpoints:**

- **Base URL**: https://api.freeagentics.io/api/v1
- **Documentation**: https://docs.freeagentics.io
- **Support**: api-support@freeagentics.io
