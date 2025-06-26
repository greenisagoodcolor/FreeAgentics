# Agent API Documentation

## Overview

The Agent API provides comprehensive endpoints for creating, managing, and interacting with AI agents in the FreeAgentics system. All endpoints require authentication and implement rate limiting.

## Authentication

All API endpoints require session-based authentication. Include a valid session cookie with each request.

## Rate Limiting

The API implements rate limiting to prevent abuse:

- General read operations: 20 requests per minute
- Create/update operations: 10 requests per minute
- Delete operations: 5 requests per minute
- Command execution: 10 requests per minute

## Base URL

```
/api/agents
```

## Endpoints

### List Agents

Get a paginated list of agents with optional filtering.

```
GET /api/agents
```

#### Query Parameters

| Parameter  | Type   | Description                                                                                       | Default    |
| ---------- | ------ | ------------------------------------------------------------------------------------------------- | ---------- |
| status     | string | Filter by agent status (idle, moving, interacting, planning, executing, learning, error, offline) | -          |
| capability | string | Filter by capability                                                                              | -          |
| tag        | string | Filter by tag                                                                                     | -          |
| limit      | number | Number of results per page (1-100)                                                                | 20         |
| offset     | number | Pagination offset                                                                                 | 0          |
| sortBy     | string | Sort field (created_at, updated_at, name, status)                                                 | created_at |
| sortOrder  | string | Sort order (asc, desc)                                                                            | desc       |

#### Response

```json
{
  "agents": [
    {
      "id": "agent-1",
      "name": "Explorer Alpha",
      "status": "idle",
      "personality": {
        "openness": 0.8,
        "conscientiousness": 0.7,
        "extraversion": 0.6,
        "agreeableness": 0.75,
        "neuroticism": 0.3
      },
      "capabilities": ["movement", "perception", "communication", "planning"],
      "position": { "x": 10, "y": 20, "z": 0 },
      "resources": {
        "energy": 85,
        "health": 100,
        "memory_used": 2048,
        "memory_capacity": 8192
      },
      "created_at": "2025-06-18T12:00:00Z",
      "updated_at": "2025-06-18T14:30:00Z"
    }
  ],
  "pagination": {
    "total": 50,
    "limit": 20,
    "offset": 0,
    "hasMore": true
  }
}
```

### Create Agent

Create a new agent with specified personality and capabilities.

```
POST /api/agents
```

#### Request Body

```json
{
  "name": "Explorer Beta",
  "personality": {
    "openness": 0.8,
    "conscientiousness": 0.7,
    "extraversion": 0.6,
    "agreeableness": 0.75,
    "neuroticism": 0.3
  },
  "capabilities": ["movement", "perception", "communication"],
  "initialPosition": {
    "x": 0,
    "y": 0,
    "z": 0
  },
  "tags": ["explorer", "autonomous"],
  "metadata": {
    "purpose": "exploration",
    "team": "alpha"
  }
}
```

#### Response

```json
{
  "agent": {
    "id": "agent-123456789",
    "name": "Explorer Beta",
    "status": "idle",
    "personality": { ... },
    "capabilities": [ ... ],
    "position": { ... },
    "resources": { ... },
    "tags": [ ... ],
    "metadata": { ... },
    "created_at": "2025-06-18T15:00:00Z",
    "updated_at": "2025-06-18T15:00:00Z"
  }
}
```

### Get Agent

Get detailed information about a specific agent.

```
GET /api/agents/{agentId}
```

#### Response

```json
{
  "agent": {
    "id": "agent-1",
    "name": "Explorer Alpha",
    "status": "idle",
    "personality": { ... },
    "capabilities": [ ... ],
    "position": { ... },
    "resources": { ... },
    "beliefs": [
      {
        "id": "belief-1",
        "content": "Resource at location (15, 25)",
        "confidence": 0.8
      }
    ],
    "goals": [
      {
        "id": "goal-1",
        "description": "Explore unknown areas",
        "priority": 0.7,
        "deadline": null
      }
    ],
    "relationships": [
      {
        "agent_id": "agent-2",
        "trust_level": 0.8,
        "last_interaction": "2025-06-18T12:00:00Z"
      }
    ],
    "tags": [ ... ],
    "metadata": { ... },
    "created_at": "2025-06-18T12:00:00Z",
    "updated_at": "2025-06-18T14:30:00Z"
  }
}
```

### Update Agent

Update agent properties.

```
PUT /api/agents/{agentId}
```

#### Request Body

```json
{
  "name": "Explorer Alpha v2",
  "status": "moving",
  "position": {
    "x": 15,
    "y": 25,
    "z": 0
  },
  "resources": {
    "energy": 75,
    "health": 95
  },
  "tags": ["explorer", "veteran"],
  "metadata": {
    "version": "2.0"
  }
}
```

### Delete Agent

Delete an agent and all associated data.

```
DELETE /api/agents/{agentId}
```

#### Response

```json
{
  "message": "Agent agent-1 deleted successfully",
  "deleted_at": "2025-06-18T15:30:00Z"
}
```

## Agent State Management

### Get State History

Get the state transition history for an agent.

```
GET /api/agents/{agentId}/state
```

#### Query Parameters

| Parameter | Type   | Description       | Default |
| --------- | ------ | ----------------- | ------- |
| limit     | number | Number of results | 10      |
| offset    | number | Pagination offset | 0       |

#### Response

```json
{
  "agent_id": "agent-1",
  "current_state": "idle",
  "state_history": [
    {
      "timestamp": "2025-06-18T14:00:00Z",
      "from_state": "idle",
      "to_state": "moving",
      "reason": "Task assigned: Explore sector 5",
      "metadata": {
        "task_id": "task-123"
      }
    }
  ],
  "pagination": { ... }
}
```

### Update State

Update the agent's state with validation.

```
PUT /api/agents/{agentId}/state
```

#### Request Body

```json
{
  "status": "moving",
  "force": false
}
```

## Command Execution

### Get Command History

Get the command execution history for an agent.

```
GET /api/agents/{agentId}/commands
```

#### Query Parameters

| Parameter | Type   | Description                                             | Default |
| --------- | ------ | ------------------------------------------------------- | ------- |
| status    | string | Filter by status (queued, executing, completed, failed) | -       |
| limit     | number | Number of results                                       | 10      |
| offset    | number | Pagination offset                                       | 0       |

### Execute Command

Execute a command on the agent.

```
POST /api/agents/{agentId}/commands
```

#### Request Body

```json
{
  "command": "move",
  "parameters": {
    "target": { "x": 20, "y": 30 },
    "speed": "normal"
  },
  "async": true
}
```

#### Available Commands

- `move`: Move to a location
- `interact`: Interact with another agent or object
- `observe`: Scan the surrounding area
- `plan`: Create a plan for achieving goals
- `learn`: Process experiences and update knowledge
- `rest`: Conserve energy and restore resources

#### Response (Async)

```json
{
  "command": {
    "id": "cmd-123456",
    "agent_id": "agent-1",
    "command": "move",
    "parameters": { ... },
    "status": "queued",
    "issued_at": "2025-06-18T15:00:00Z"
  },
  "async": true,
  "status_url": "/api/agents/agent-1/commands/cmd-123456"
}
```

### Get Command Status

Get the status of a specific command.

```
GET /api/agents/{agentId}/commands/{commandId}
```

## Memory Operations

### Query Memories

Query the agent's memory system.

```
GET /api/agents/{agentId}/memory
```

#### Query Parameters

| Parameter      | Type   | Description                                                  | Default |
| -------------- | ------ | ------------------------------------------------------------ | ------- |
| type           | string | Memory type (event, interaction, location, pattern, general) | -       |
| query          | string | Search query                                                 | -       |
| tags           | array  | Filter by tags                                               | -       |
| min_importance | number | Minimum importance (0-1)                                     | -       |
| limit          | number | Number of results                                            | 20      |
| offset         | number | Pagination offset                                            | 0       |

### Add Memory

Add a new memory to the agent.

```
POST /api/agents/{agentId}/memory
```

#### Request Body

```json
{
  "type": "location",
  "content": "Found abundant resources at coordinates (45, 67)",
  "importance": 0.9,
  "tags": ["resources", "exploration"],
  "metadata": {
    "location": { "x": 45, "y": 67 },
    "resource_type": "energy_crystal"
  }
}
```

### Delete Memory

Delete a specific memory.

```
DELETE /api/agents/{agentId}/memory/{memoryId}
```

### Consolidate Memories

Trigger memory consolidation to optimize storage.

```
POST /api/agents/{agentId}/memory/consolidate
```

## Export/Import

### Export Agent

Export an agent for deployment to edge devices.

```
POST /api/agents/{agentId}/export
```

#### Request Body

```json
{
  "target": "raspberry_pi_4b"
}
```

### Evaluate Readiness

Check if an agent is ready for deployment.

```
GET /api/agents/{agentId}/readiness
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Error message",
  "details": {
    // Additional error context
  }
}
```

### Common Error Codes

- `400`: Bad Request - Invalid parameters or request body
- `401`: Unauthorized - Missing or invalid authentication
- `404`: Not Found - Agent or resource not found
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server-side error

## WebSocket Support

For real-time agent monitoring, connect to:

```
ws://[host]/api/agents/ws
```

Send agent IDs to subscribe:

```json
{
  "action": "subscribe",
  "agent_ids": ["agent-1", "agent-2"]
}
```

Receive real-time updates:

```json
{
  "type": "state_change",
  "agent_id": "agent-1",
  "data": {
    "previous_state": "idle",
    "current_state": "moving",
    "timestamp": "2025-06-18T15:00:00Z"
  }
}
```

## Examples

### Create and Command an Agent

```bash
# Create agent
curl -X POST http://localhost:3000/api/agents \
  -H "Content-Type: application/json" \
  -H "Cookie: session=..." \
  -d '{
    "name": "Explorer Bot",
    "personality": {
      "openness": 0.8,
      "conscientiousness": 0.7,
      "extraversion": 0.6,
      "agreeableness": 0.75,
      "neuroticism": 0.3
    }
  }'

# Execute move command
curl -X POST http://localhost:3000/api/agents/agent-123/commands \
  -H "Content-Type: application/json" \
  -H "Cookie: session=..." \
  -d '{
    "command": "move",
    "parameters": {
      "target": { "x": 50, "y": 50 }
    }
  }'
```

### Monitor Agent State

```javascript
// JavaScript/TypeScript example
const eventSource = new EventSource("/api/agents/agent-123/state/stream");

eventSource.onmessage = (event) => {
  const stateUpdate = JSON.parse(event.data);
  console.log("State changed:", stateUpdate);
};
```

## SDK Support

Official SDKs are available for:

- TypeScript/JavaScript
- Python
- Go

See the respective SDK documentation for language-specific examples.
