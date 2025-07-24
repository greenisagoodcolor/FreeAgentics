# FreeAgentics API Reference

## Overview

The FreeAgentics API provides endpoints for creating and managing Active Inference agents through natural language prompts. This reference covers the prompt processing pipeline and related endpoints.

**Base URL**: `http://localhost:8000/api/v1`
**Protocol**: HTTP/HTTPS
**Authentication**: JWT Bearer tokens

## Table of Contents

1. [Authentication](#authentication)
2. [Prompt Processing](#prompt-processing)
3. [WebSocket Events](#websocket-events)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)

## Authentication

All API requests require authentication using JWT tokens.

### Headers

```
Authorization: Bearer <access_token>
```

### Token Management

- Access tokens expire after 15 minutes
- Use refresh tokens to obtain new access tokens
- Refresh tokens are rotated on each use

## Prompt Processing

### POST /api/v1/prompts

Process a natural language prompt to create or modify an agent.

This endpoint orchestrates the full pipeline:

1. Converts natural language to GMN specification via LLM
2. Validates and parses GMN into PyMDP model
3. Creates active inference agent from model
4. Updates knowledge graph with agent beliefs
5. Returns suggestions for next actions

#### Request

```http
POST /api/v1/prompts
Content-Type: application/json
Authorization: Bearer <token>

{
  "prompt": "Create an explorer agent for a 5x5 grid world",
  "conversation_id": "optional-uuid",
  "iteration_count": 1
}
```

#### Request Body

| Field             | Type    | Required | Description                                  |
| ----------------- | ------- | -------- | -------------------------------------------- |
| `prompt`          | string  | Yes      | Natural language prompt (1-2000 chars)       |
| `conversation_id` | string  | No       | Continue existing conversation               |
| `iteration_count` | integer | No       | GMN refinement iterations (1-10, default: 1) |

#### Response

```json
{
  "agent_id": "agent_123e4567-e89b-12d3-a456-426614174000",
  "gmn_specification": "WORLD grid_world_5x5\n  STATES locations[25]\n  ...",
  "knowledge_graph_updates": [
    {
      "node_id": "node_001",
      "type": "belief",
      "properties": {
        "agent_id": "agent_123...",
        "belief_type": "location",
        "uncertainty": 0.8
      }
    }
  ],
  "next_suggestions": [
    "Add curiosity-driven exploration to reduce uncertainty",
    "Define goal states for directed behavior",
    "Consider adding obstacles to the grid world"
  ],
  "status": "success",
  "processing_time_ms": 782.5,
  "iteration_context": {
    "iteration_number": 1,
    "total_agents": 1,
    "kg_nodes": 5,
    "conversation_summary": "Created basic explorer agent"
  }
}
```

#### Response Fields

| Field                     | Type   | Description                                         |
| ------------------------- | ------ | --------------------------------------------------- |
| `agent_id`                | string | UUID of created/modified agent                      |
| `gmn_specification`       | string | Generated GMN specification                         |
| `knowledge_graph_updates` | array  | List of KG nodes added/modified                     |
| `next_suggestions`        | array  | Suggested refinements or actions                    |
| `status`                  | string | Processing status: success, partial_success, failed |
| `warnings`                | array  | Any warnings during processing                      |
| `processing_time_ms`      | number | Total processing time in milliseconds               |
| `iteration_context`       | object | Conversation iteration details                      |

#### Status Codes

| Code | Description                                           |
| ---- | ----------------------------------------------------- |
| 200  | Success - Agent created/modified                      |
| 400  | Bad Request - Invalid prompt or GMN validation failed |
| 401  | Unauthorized - Invalid or missing token               |
| 403  | Forbidden - Insufficient permissions                  |
| 422  | Unprocessable Entity - Request validation failed      |
| 500  | Internal Server Error - Agent creation failed         |

### GET /api/v1/prompts/templates

Get available prompt templates for common agent types.

#### Request

```http
GET /api/v1/prompts/templates?category=explorer
Authorization: Bearer <token>
```

#### Query Parameters

| Parameter  | Type   | Description                 |
| ---------- | ------ | --------------------------- |
| `category` | string | Filter by template category |

#### Response

```json
[
  {
    "id": "explorer-basic",
    "name": "Basic Explorer Agent",
    "category": "explorer",
    "description": "Simple grid world exploration agent",
    "example_prompt": "Create an explorer agent for a 5x5 grid world",
    "suggested_parameters": {
      "grid_size": [5, 5],
      "planning_horizon": 3
    }
  }
]
```

### GET /api/v1/prompts/suggestions

Get contextual suggestions for an existing agent.

#### Request

```http
GET /api/v1/prompts/suggestions?agent_id=agent_123
Authorization: Bearer <token>
```

#### Response

```json
{
  "agent_id": "agent_123",
  "current_state": {
    "belief_entropy": 0.75,
    "exploration_coverage": 0.3,
    "goal_progress": 0.5
  },
  "suggestions": [
    "Add curiosity-driven exploration to reduce uncertainty",
    "Define clearer goal states for directed behavior",
    "Consider forming a coalition for complex tasks"
  ],
  "recommended_prompts": [
    "Make the agent more curious about unexplored areas",
    "Add a specific goal location to the agent's preferences",
    "Create a coordinator agent to work with this explorer"
  ]
}
```

## WebSocket Events

The prompt processing pipeline sends real-time updates via WebSocket.

### Connection

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/client123?token=<access_token>");
```

### Event Types

#### Pipeline Events

**pipeline_started**

```json
{
  "type": "pipeline_started",
  "timestamp": "2024-01-18T10:30:00Z",
  "data": {
    "prompt_id": "prompt_123",
    "prompt_text": "Create an explorer...",
    "conversation_id": "conv_123",
    "stage": "initialization",
    "total_stages": 6,
    "iteration_number": 1
  }
}
```

**pipeline_progress**

```json
{
  "type": "pipeline_progress",
  "timestamp": "2024-01-18T10:30:01Z",
  "data": {
    "prompt_id": "prompt_123",
    "stage": "gmn_generation",
    "stage_number": 1,
    "message": "Generating GMN specification..."
  }
}
```

**gmn_generated**

```json
{
  "type": "gmn_generated",
  "timestamp": "2024-01-18T10:30:02Z",
  "data": {
    "prompt_id": "prompt_123",
    "gmn_preview": "WORLD grid_world_5x5...",
    "gmn_length": 1248
  }
}
```

**validation_success**

```json
{
  "type": "validation_success",
  "timestamp": "2024-01-18T10:30:03Z",
  "data": {
    "prompt_id": "prompt_123",
    "model_dimensions": {
      "num_states": [25],
      "num_obs": [25],
      "num_actions": [4]
    }
  }
}
```

**agent_created**

```json
{
  "type": "agent_created",
  "timestamp": "2024-01-18T10:30:04Z",
  "data": {
    "prompt_id": "prompt_123",
    "agent_id": "agent_123",
    "agent_type": "explorer"
  }
}
```

**knowledge_graph_updated**

```json
{
  "type": "knowledge_graph_updated",
  "timestamp": "2024-01-18T10:30:05Z",
  "data": {
    "prompt_id": "prompt_123",
    "updates_count": 5,
    "nodes_added": 5
  }
}
```

**pipeline_completed**

```json
{
  "type": "pipeline_completed",
  "timestamp": "2024-01-18T10:30:06Z",
  "data": {
    "prompt_id": "prompt_123",
    "agent_id": "agent_123",
    "processing_time_ms": 782.5,
    "suggestions": [...],
    "status": "success"
  }
}
```

#### Error Events

**validation_failed**

```json
{
  "type": "validation_failed",
  "timestamp": "2024-01-18T10:30:03Z",
  "data": {
    "prompt_id": "prompt_123",
    "errors": ["Invalid state space dimensions"],
    "stage": "gmn_validation"
  }
}
```

**pipeline_failed**

```json
{
  "type": "pipeline_failed",
  "timestamp": "2024-01-18T10:30:04Z",
  "data": {
    "prompt_id": "prompt_123",
    "stage": "agent_creation",
    "error": "Failed to initialize PyMDP agent",
    "error_type": "runtime_error"
  }
}
```

### Subscribing to Events

```javascript
// Subscribe to pipeline events
ws.send(
  JSON.stringify({
    type: "subscribe",
    event_types: ["pipeline:*", "agent:*", "knowledge_graph:*"],
  }),
);

// Handle events
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log("Event:", message.type, message.data);
};
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "GMN validation failed: Invalid state space dimensions",
    "details": {
      "field": "num_states",
      "expected": "array of positive integers",
      "received": "[-1, 5]"
    }
  },
  "request_id": "req_123"
}
```

### Common Error Codes

| Code               | HTTP Status | Description                       |
| ------------------ | ----------- | --------------------------------- |
| `VALIDATION_ERROR` | 400         | Request or GMN validation failed  |
| `UNAUTHORIZED`     | 401         | Missing or invalid authentication |
| `FORBIDDEN`        | 403         | Insufficient permissions          |
| `NOT_FOUND`        | 404         | Resource not found                |
| `RATE_LIMITED`     | 429         | Rate limit exceeded               |
| `INTERNAL_ERROR`   | 500         | Server error during processing    |

## Rate Limiting

### Limits

| Endpoint                      | Limit | Window     |
| ----------------------------- | ----- | ---------- |
| `/api/v1/prompts`             | 60    | per minute |
| `/api/v1/prompts/templates`   | 100   | per minute |
| `/api/v1/prompts/suggestions` | 100   | per minute |

### Rate Limit Headers

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1673890800
```

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded. Please retry after 1673890800",
    "retry_after": 1673890800
  }
}
```

## Example Integration

### Python Example

```python
import requests
import json

# Configuration
API_URL = "http://localhost:8000/api/v1"
TOKEN = "your_jwt_token"

# Headers
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Create agent from prompt
prompt_data = {
    "prompt": "Create an explorer agent for a 5x5 grid world with obstacles",
    "iteration_count": 2
}

response = requests.post(
    f"{API_URL}/prompts",
    headers=headers,
    json=prompt_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Agent created: {result['agent_id']}")
    print(f"Suggestions: {result['next_suggestions']}")
else:
    print(f"Error: {response.json()}")
```

### JavaScript Example

```javascript
// Using fetch API
const API_URL = "http://localhost:8000/api/v1";
const TOKEN = "your_jwt_token";

async function createAgent(prompt) {
  const response = await fetch(`${API_URL}/prompts`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt: prompt,
      iteration_count: 1,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const result = await response.json();
  return result;
}

// Usage
createAgent("Create a trader agent for market simulation")
  .then((result) => {
    console.log("Agent ID:", result.agent_id);
    console.log("Suggestions:", result.next_suggestions);
  })
  .catch((error) => {
    console.error("Error:", error);
  });
```

### WebSocket Example

```javascript
class PromptPipeline {
  constructor(wsUrl, token) {
    this.ws = new WebSocket(`${wsUrl}?token=${token}`);
    this.setupEventHandlers();
  }

  setupEventHandlers() {
    this.ws.onopen = () => {
      // Subscribe to pipeline events
      this.ws.send(
        JSON.stringify({
          type: "subscribe",
          event_types: ["pipeline:*"],
        }),
      );
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handlePipelineEvent(message);
    };
  }

  handlePipelineEvent(message) {
    switch (message.type) {
      case "pipeline_started":
        console.log("Pipeline started:", message.data.prompt_id);
        break;
      case "pipeline_progress":
        console.log(`Stage ${message.data.stage_number}: ${message.data.message}`);
        break;
      case "pipeline_completed":
        console.log("Agent created:", message.data.agent_id);
        console.log("Suggestions:", message.data.suggestions);
        break;
      case "pipeline_failed":
        console.error("Pipeline failed:", message.data.error);
        break;
    }
  }
}

// Usage
const pipeline = new PromptPipeline("ws://localhost:8000/ws/client1", "your_token");
```

## Support

- GitHub Issues: https://github.com/freeagentics/freeagentics
- Documentation: https://docs.freeagentics.com
- Community: https://community.freeagentics.com
