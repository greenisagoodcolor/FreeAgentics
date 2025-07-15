# FreeAgentics API Reference

## Overview

The FreeAgentics API provides a comprehensive interface for managing multi-agent AI systems with Active Inference capabilities. This reference documents all available endpoints, authentication procedures, and integration patterns.

**Base URL**: `https://api.freeagentics.com`  
**API Version**: `v1`  
**Protocol**: HTTPS  

## Table of Contents

1. [Authentication](#authentication)
2. [Rate Limiting](#rate-limiting)
3. [Error Handling](#error-handling)
4. [Core Endpoints](#core-endpoints)
   - [Authentication](#authentication-endpoints)
   - [Agents](#agents-endpoints)
   - [Inference](#inference-endpoints)
   - [Knowledge Graph](#knowledge-graph-endpoints)
   - [System](#system-endpoints)
   - [Monitoring](#monitoring-endpoints)
   - [Security](#security-endpoints)
5. [WebSocket APIs](#websocket-apis)
6. [GraphQL API](#graphql-api)
7. [Webhooks](#webhooks)

## Authentication

FreeAgentics uses JWT (JSON Web Token) based authentication with refresh token rotation for enhanced security.

### Authentication Flow

1. **Registration/Login**: Obtain access and refresh tokens
2. **API Requests**: Include access token in Authorization header
3. **Token Refresh**: Use refresh token to obtain new access token when expired
4. **Logout**: Invalidate refresh token

### Headers

All authenticated requests must include:
```
Authorization: Bearer <access_token>
```

Optional security headers:
```
X-Client-Fingerprint: <unique_client_identifier>
X-Client-Version: <client_version>
```

### Token Lifecycle

- **Access Token**: Valid for 15 minutes
- **Refresh Token**: Valid for 7 days (rotates on each refresh)
- **Token Rotation**: Old refresh tokens are invalidated after use

## Rate Limiting

Rate limits vary by environment and endpoint category:

### Production Limits

| Category | Per Minute | Per Hour | Burst | Block Duration |
|----------|------------|----------|-------|----------------|
| Auth     | 3          | 50       | 2     | 15 minutes     |
| API      | 60         | 1000     | 10    | 10 minutes     |
| WebSocket| 100        | 2000     | 20    | 5 minutes      |
| Static   | 200        | 5000     | 50    | 2 minutes      |

### Rate Limit Headers

Response headers include:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1673890800
```

### DDoS Protection

- Threshold-based blocking for suspected DDoS attempts
- Automatic IP blocking with configurable duration
- Whitelist support for trusted IPs

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request was invalid",
    "details": {
      "field": "name",
      "reason": "Required field missing"
    }
  },
  "request_id": "req_12345"
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | INVALID_REQUEST | Request validation failed |
| 401 | UNAUTHORIZED | Missing or invalid authentication |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | NOT_FOUND | Resource not found |
| 429 | RATE_LIMITED | Rate limit exceeded |
| 500 | INTERNAL_ERROR | Server error |

## Core Endpoints

### Authentication Endpoints

#### POST /api/v1/register
Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "user@example.com",
  "password": "string",
  "role": "OBSERVER" // OBSERVER, CONTRIBUTOR, OPERATOR, ADMIN
}
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "user": {
    "user_id": "uuid",
    "username": "string",
    "role": "OBSERVER",
    "permissions": ["VIEW_AGENTS", "VIEW_METRICS"]
  }
}
```

**Rate Limit:** 5 requests per 10 minutes

#### POST /api/v1/login
Authenticate and obtain tokens.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:** Same as registration

**Rate Limit:** 10 requests per 5 minutes

#### POST /api/v1/refresh
Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJ..."
}
```

**Response:** New token pair

**Rate Limit:** 10 requests per 5 minutes

#### GET /api/v1/me
Get current user information.

**Response:**
```json
{
  "user_id": "uuid",
  "username": "string",
  "role": "CONTRIBUTOR",
  "permissions": ["VIEW_AGENTS", "CREATE_AGENT", "VIEW_METRICS"],
  "exp": "2024-01-15T10:30:00Z"
}
```

#### POST /api/v1/logout
Invalidate refresh token and logout.

**Response:**
```json
{
  "message": "Successfully logged out"
}
```

#### GET /api/v1/permissions
Get user permissions for UI rendering.

**Response:**
```json
{
  "permissions": ["VIEW_AGENTS", "CREATE_AGENT"],
  "role": "CONTRIBUTOR",
  "can_create_agents": true,
  "can_delete_agents": false,
  "can_view_metrics": true,
  "can_admin_system": false
}
```

### Agents Endpoints

#### GET /api/v1/agents
List all agents accessible to the user.

**Query Parameters:**
- `skip`: Number of items to skip (default: 0)
- `limit`: Maximum items to return (default: 100, max: 1000)
- `status`: Filter by status (active, paused, stopped)
- `template`: Filter by template ID

**Response:**
```json
{
  "agents": [
    {
      "id": "agent_123",
      "name": "Research Agent",
      "template": "research_v2",
      "status": "active",
      "created_at": "2024-01-10T08:00:00Z",
      "last_active": "2024-01-15T09:30:00Z",
      "inference_count": 1543,
      "parameters": {
        "temperature": 0.7,
        "max_tokens": 2048
      }
    }
  ],
  "total": 25,
  "skip": 0,
  "limit": 100
}
```

**Required Permission:** VIEW_AGENTS

#### POST /api/v1/agents
Create a new agent.

**Request Body:**
```json
{
  "name": "My Agent",
  "template": "research_v2",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 2048
  },
  "gmn_spec": "optional GMN specification",
  "use_pymdp": true,
  "planning_horizon": 3
}
```

**Response:**
```json
{
  "id": "agent_456",
  "name": "My Agent",
  "template": "research_v2",
  "status": "pending",
  "created_at": "2024-01-15T10:00:00Z",
  "parameters": {...}
}
```

**Required Permission:** CREATE_AGENT

#### GET /api/v1/agents/{agent_id}
Get detailed agent information.

**Response:** Single agent object with extended metrics

**Required Permission:** VIEW_AGENTS

#### PUT /api/v1/agents/{agent_id}
Update agent configuration.

**Request Body:** Partial agent config
**Required Permission:** UPDATE_AGENT

#### DELETE /api/v1/agents/{agent_id}
Delete an agent.

**Required Permission:** DELETE_AGENT

#### POST /api/v1/agents/{agent_id}/start
Start/resume agent processing.

**Required Permission:** CONTROL_AGENT

#### POST /api/v1/agents/{agent_id}/stop
Stop agent processing.

**Required Permission:** CONTROL_AGENT

#### GET /api/v1/agents/{agent_id}/metrics
Get agent performance metrics.

**Response:**
```json
{
  "agent_id": "agent_123",
  "total_inferences": 1543,
  "avg_response_time": 0.235,
  "success_rate": 0.98,
  "error_count": 31,
  "last_24h": {
    "inferences": 245,
    "avg_response_time": 0.212
  }
}
```

### Inference Endpoints

#### POST /api/v1/inference
Submit inference request.

**Request Body:**
```json
{
  "agent_id": "agent_123",
  "query": "Analyze market trends",
  "context": {
    "data_source": "financial_reports",
    "time_range": "2024-Q1"
  },
  "parameters": {
    "temperature": 0.8,
    "max_tokens": 1000
  }
}
```

**Response:**
```json
{
  "inference_id": "inf_789",
  "agent_id": "agent_123",
  "status": "processing",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:30:45Z"
}
```

#### GET /api/v1/inference/{inference_id}
Get inference result.

**Response:**
```json
{
  "inference_id": "inf_789",
  "status": "completed",
  "result": {
    "analysis": "Market shows bullish trends...",
    "confidence": 0.87,
    "metadata": {...}
  },
  "processing_time": 0.732,
  "completed_at": "2024-01-15T10:30:43Z"
}
```

#### POST /api/v1/batch-inference
Submit multiple inference requests.

**Request Body:**
```json
{
  "requests": [
    {
      "agent_id": "agent_123",
      "query": "Query 1"
    },
    {
      "agent_id": "agent_456",
      "query": "Query 2"
    }
  ],
  "parallel": true
}
```

### Knowledge Graph Endpoints

#### GET /api/v1/knowledge/search
Search knowledge graph.

**Query Parameters:**
- `q`: Search query
- `limit`: Max results (default: 20)
- `type`: Entity type filter

**Response:**
```json
{
  "results": [
    {
      "id": "entity_123",
      "type": "concept",
      "label": "Market Analysis",
      "properties": {...},
      "score": 0.95
    }
  ],
  "total": 15
}
```

#### POST /api/v1/knowledge/entities
Create knowledge entity.

**Request Body:**
```json
{
  "type": "concept",
  "label": "New Concept",
  "properties": {
    "description": "...",
    "category": "finance"
  }
}
```

#### PUT /api/v1/knowledge/entities/{entity_id}
Update knowledge entity.

#### DELETE /api/v1/knowledge/entities/{entity_id}
Delete knowledge entity.

#### POST /api/v1/knowledge/relationships
Create entity relationship.

**Request Body:**
```json
{
  "source_id": "entity_123",
  "target_id": "entity_456",
  "relationship_type": "related_to",
  "properties": {...}
}
```

### System Endpoints

#### GET /api/v1/system/status
Get system health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "inference_engine": "healthy"
  },
  "uptime": 864000,
  "last_check": "2024-01-15T10:35:00Z"
}
```

#### GET /api/v1/system/metrics
Get system-wide metrics.

**Response:**
```json
{
  "cpu_usage": 45.2,
  "memory_usage": 62.8,
  "active_agents": 12,
  "total_inferences_24h": 15432,
  "avg_response_time": 0.287,
  "queue_length": 23
}
```

**Required Permission:** VIEW_METRICS

#### GET /api/v1/system/config
Get system configuration (admin only).

**Required Permission:** ADMIN_SYSTEM

### Monitoring Endpoints

#### GET /api/v1/monitoring/metrics
Get real-time metrics stream.

**Query Parameters:**
- `metrics`: Comma-separated metric types
- `agents`: Comma-separated agent IDs
- `interval`: Update interval in seconds

#### GET /api/v1/monitoring/alerts
Get active system alerts.

**Response:**
```json
{
  "alerts": [
    {
      "id": "alert_123",
      "severity": "warning",
      "type": "high_memory_usage",
      "message": "Memory usage above 80%",
      "timestamp": "2024-01-15T10:30:00Z",
      "affected_components": ["agent_456"]
    }
  ]
}
```

### Security Endpoints

#### GET /api/v1/security/audit-log
Get security audit log.

**Query Parameters:**
- `start_date`: ISO 8601 date
- `end_date`: ISO 8601 date
- `event_type`: Filter by event type
- `user_id`: Filter by user

**Response:**
```json
{
  "events": [
    {
      "id": "event_123",
      "type": "LOGIN_SUCCESS",
      "severity": "INFO",
      "timestamp": "2024-01-15T09:00:00Z",
      "user_id": "user_123",
      "ip_address": "192.168.1.1",
      "details": {...}
    }
  ],
  "total": 1543,
  "page": 1
}
```

**Required Permission:** VIEW_AUDIT_LOG

#### GET /api/v1/security/active-sessions
Get active user sessions.

**Required Permission:** ADMIN_SYSTEM

#### POST /api/v1/security/revoke-session
Revoke a user session.

**Request Body:**
```json
{
  "session_id": "session_123"
}
```

**Required Permission:** ADMIN_SYSTEM

## WebSocket APIs

WebSocket connections provide real-time bidirectional communication for agent monitoring and control.

### Connection URL
```
wss://api.freeagentics.com/api/v1/ws
```

### Authentication
Include token as query parameter:
```
wss://api.freeagentics.com/api/v1/ws?token=<access_token>
```

### Connection Lifecycle

1. **Connection**: Client connects with valid token
2. **Acknowledgment**: Server sends connection confirmation
3. **Subscription**: Client subscribes to events
4. **Message Exchange**: Bidirectional message flow
5. **Disconnection**: Clean disconnect or timeout

### Message Format

All WebSocket messages follow this format:
```json
{
  "type": "message_type",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    // Message-specific data
  }
}
```

### Message Types

#### Client → Server

**subscribe**
```json
{
  "type": "subscribe",
  "data": {
    "events": ["agent_status", "metrics"],
    "agents": ["agent_123", "agent_456"]
  }
}
```

**unsubscribe**
```json
{
  "type": "unsubscribe",
  "data": {
    "events": ["metrics"]
  }
}
```

**command**
```json
{
  "type": "command",
  "data": {
    "agent_id": "agent_123",
    "command": "stop"
  }
}
```

#### Server → Client

**connection_established**
```json
{
  "type": "connection_established",
  "data": {
    "client_id": "client_789",
    "pooled": true
  }
}
```

**agent_status**
```json
{
  "type": "agent_status",
  "data": {
    "agent_id": "agent_123",
    "status": "active",
    "last_inference": "2024-01-15T10:29:00Z"
  }
}
```

**metrics_update**
```json
{
  "type": "metrics_update",
  "data": {
    "cpu_usage": 45.2,
    "memory_usage": 62.8,
    "agent_metrics": {
      "agent_123": {
        "inference_rate": 2.5,
        "avg_response_time": 0.234
      }
    }
  }
}
```

**error**
```json
{
  "type": "error",
  "data": {
    "code": "INVALID_COMMAND",
    "message": "Unknown command type"
  }
}
```

### Connection Pooling

WebSocket connections are pooled for efficiency:
- Maximum connections per client: 5
- Idle timeout: 5 minutes
- Reconnection backoff: Exponential

### Rate Limiting

WebSocket-specific limits:
- Messages per minute: 100
- Subscriptions per connection: 20
- Maximum message size: 64KB

## GraphQL API

GraphQL endpoint for flexible data queries.

### Endpoint
```
POST /api/v1/graphql
```

### Authentication
Same as REST API - include Authorization header.

### Schema

```graphql
type Query {
  # Agent queries
  agent(id: ID!): Agent
  agents(
    status: AgentStatus
    template: String
    limit: Int = 100
    offset: Int = 0
  ): AgentConnection!
  
  # Metrics queries
  systemMetrics: SystemMetrics!
  agentMetrics(agentId: ID!): AgentMetrics
  
  # Knowledge graph queries
  searchKnowledge(
    query: String!
    limit: Int = 20
  ): [KnowledgeEntity!]!
}

type Mutation {
  # Agent mutations
  createAgent(input: CreateAgentInput!): Agent!
  updateAgent(id: ID!, input: UpdateAgentInput!): Agent!
  deleteAgent(id: ID!): Boolean!
  
  # Agent control
  startAgent(id: ID!): Agent!
  stopAgent(id: ID!): Agent!
  
  # Knowledge mutations
  createEntity(input: CreateEntityInput!): KnowledgeEntity!
  createRelationship(input: CreateRelationshipInput!): Relationship!
}

type Subscription {
  # Real-time subscriptions
  agentStatusChanged(agentId: ID): AgentStatusUpdate!
  metricsUpdate(interval: Int = 1000): MetricsUpdate!
  systemAlert: SystemAlert!
}
```

### Example Queries

**Get agent with metrics:**
```graphql
query GetAgentDetails($id: ID!) {
  agent(id: $id) {
    id
    name
    status
    template
    metrics {
      totalInferences
      avgResponseTime
      successRate
    }
    recentInferences(limit: 10) {
      id
      query
      completedAt
      processingTime
    }
  }
}
```

**Create agent with subscription:**
```graphql
mutation CreateAgent($input: CreateAgentInput!) {
  createAgent(input: $input) {
    id
    name
    status
  }
}

subscription WatchAgent($agentId: ID!) {
  agentStatusChanged(agentId: $agentId) {
    agentId
    oldStatus
    newStatus
    timestamp
  }
}
```

## Webhooks

Configure webhooks to receive real-time notifications about system events.

### Webhook Configuration

Register webhooks via API or dashboard:

```json
POST /api/v1/webhooks
{
  "url": "https://your-domain.com/webhook",
  "events": ["agent.created", "agent.status_changed", "inference.completed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Security

- **Signature Verification**: All webhook requests include `X-FreeAgentics-Signature` header
- **Timestamp**: `X-FreeAgentics-Timestamp` prevents replay attacks
- **Secret**: HMAC-SHA256 signature using your webhook secret

### Signature Verification Example

```python
import hmac
import hashlib

def verify_webhook(payload, signature, timestamp, secret):
    message = f"{timestamp}.{payload}"
    expected = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

### Event Types

#### agent.created
Triggered when a new agent is created.
```json
{
  "event": "agent.created",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "agent_id": "agent_123",
    "name": "New Agent",
    "template": "research_v2",
    "created_by": "user_456"
  }
}
```

#### agent.status_changed
Triggered when agent status changes.
```json
{
  "event": "agent.status_changed",
  "timestamp": "2024-01-15T10:31:00Z",
  "data": {
    "agent_id": "agent_123",
    "old_status": "pending",
    "new_status": "active"
  }
}
```

#### inference.completed
Triggered when inference completes.
```json
{
  "event": "inference.completed",
  "timestamp": "2024-01-15T10:32:00Z",
  "data": {
    "inference_id": "inf_789",
    "agent_id": "agent_123",
    "processing_time": 0.732,
    "success": true
  }
}
```

#### system.alert
Triggered for system alerts.
```json
{
  "event": "system.alert",
  "timestamp": "2024-01-15T10:33:00Z",
  "data": {
    "alert_id": "alert_123",
    "severity": "warning",
    "type": "high_memory_usage",
    "message": "Memory usage above 80%"
  }
}
```

### Webhook Retry Policy

- Initial retry: 5 seconds
- Max retries: 5
- Backoff: Exponential (5s, 10s, 20s, 40s, 80s)
- Timeout: 30 seconds per request

### Webhook Management

#### List webhooks
```
GET /api/v1/webhooks
```

#### Update webhook
```
PUT /api/v1/webhooks/{webhook_id}
```

#### Delete webhook
```
DELETE /api/v1/webhooks/{webhook_id}
```

#### Test webhook
```
POST /api/v1/webhooks/{webhook_id}/test
```

## SDK Support

Official SDKs are available for:
- Python
- JavaScript/TypeScript
- Go
- Java
- .NET

See the [Developer Guide](DEVELOPER_GUIDE.md) for SDK usage examples.

## API Versioning

- Current version: `v1`
- Version in URL path: `/api/v1/`
- Deprecation notice: 6 months
- Sunset period: 12 months after deprecation

## Support

- Documentation: https://docs.freeagentics.com
- API Status: https://status.freeagentics.com
- Support: support@freeagentics.com
- Community: https://community.freeagentics.com