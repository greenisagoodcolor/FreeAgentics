# Agent API

## Overview

The Agent API provides a comprehensive RESTful interface for managing AI agents in the FreeAgentics system. This API enables full lifecycle management of agents, from creation and configuration to real-time monitoring and control.

## Architecture

### Directory Structure

```
app/api/agents/
├── route.ts                    # Main endpoints (list, create)
├── [agentId]/
│   ├── route.ts               # Individual agent operations (get, update, delete)
│   ├── state/
│   │   └── route.ts           # State management endpoints
│   ├── commands/
│   │   └── route.ts           # Command execution endpoints
│   ├── memory/
│   │   └── route.ts           # Memory operations endpoints
│   ├── export/
│   │   └── route.ts           # Export agent for deployment
│   ├── evaluate/
│   │   └── route.ts           # Evaluate agent performance
│   └── readiness/
│       └── route.ts           # Check deployment readiness
└── README.md                   # This file
```

## Key Features

### 1. Agent Management

- **Create**: Instantiate new agents with custom personalities and capabilities
- **List**: Query agents with filtering, pagination, and sorting
- **Get**: Retrieve detailed information about specific agents
- **Update**: Modify agent properties and configurations
- **Delete**: Remove agents and clean up resources

### 2. State Management

- Track agent state transitions with full history
- Validate state changes based on allowed transitions
- Force state changes when necessary
- Real-time state monitoring

### 3. Command Execution

- Execute commands synchronously or asynchronously
- Track command history and results
- Support for various command types (move, interact, observe, etc.)
- Command status polling for async operations

### 4. Memory Operations

- Query agent memories with filtering
- Add new memories with importance scoring
- Delete specific memories
- Trigger memory consolidation
- Support for different memory types (event, interaction, location, pattern)

### 5. Security & Rate Limiting

- Session-based authentication required for all endpoints
- Rate limiting to prevent abuse:
  - Read operations: 20/minute
  - Write operations: 10/minute
  - Delete operations: 5/minute

## Implementation Details

### Authentication

All endpoints use session-based authentication through the `validateSession` function:

```typescript
const sessionId = request.cookies.get("session")?.value;
const isValid = sessionId ? await validateSession("session", sessionId) : false;
```

### Rate Limiting

Rate limiting is implemented using the `rateLimit` utility:

```typescript
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500,
});
```

### Request Validation

All requests are validated using Zod schemas:

```typescript
const CreateAgentSchema = z.object({
  name: z.string().min(1).max(100),
  personality: z.object({
    openness: z.number().min(0).max(1),
    // ... other personality traits
  }),
  // ... other fields
});
```

### Error Handling

Consistent error responses across all endpoints:

```typescript
// Validation errors
if (error instanceof z.ZodError) {
  return NextResponse.json(
    { error: "Invalid request", details: error.errors },
    { status: 400 },
  );
}

// General errors
return NextResponse.json(
  { error: "Failed to perform operation" },
  { status: 500 },
);
```

## Usage Examples

### Create an Agent

```bash
curl -X POST http://localhost:3000/api/agents \
  -H "Content-Type: application/json" \
  -H "Cookie: session=..." \
  -d '{
    "name": "Explorer Alpha",
    "personality": {
      "openness": 0.8,
      "conscientiousness": 0.7,
      "extraversion": 0.6,
      "agreeableness": 0.75,
      "neuroticism": 0.3
    },
    "capabilities": ["movement", "perception", "communication"]
  }'
```

### Execute a Command

```bash
curl -X POST http://localhost:3000/api/agents/agent-123/commands \
  -H "Content-Type: application/json" \
  -H "Cookie: session=..." \
  -d '{
    "command": "move",
    "parameters": {
      "target": { "x": 50, "y": 50 },
      "speed": "normal"
    },
    "async": true
  }'
```

### Query Agent Memory

```bash
curl -X GET "http://localhost:3000/api/agents/agent-123/memory?type=location&min_importance=0.7" \
  -H "Cookie: session=..."
```

## Integration Points

### Database Integration

Currently using mock data. In production, integrate with:

- PostgreSQL/MySQL for persistent storage
- Redis for caching and real-time data
- Vector database for memory similarity search

### Agent System Integration

Connect to the actual agent system:

- Agent state manager
- Command executor
- Memory system
- Active inference engine

### WebSocket Support

For real-time updates, implement WebSocket endpoints:

- State change notifications
- Command execution progress
- Resource updates
- Memory consolidation status

## Future Enhancements

1. **Batch Operations**

   - Create multiple agents
   - Execute commands on agent groups
   - Bulk memory operations

2. **Advanced Querying**

   - Full-text search on agent properties
   - Geospatial queries for agent positions
   - Complex filtering with AND/OR conditions

3. **Performance Optimization**

   - Response caching
   - Query optimization
   - Connection pooling

4. **Monitoring & Analytics**

   - Agent performance metrics
   - Command execution analytics
   - Memory usage statistics

5. **Import/Export**
   - Agent configuration templates
   - Batch import from CSV/JSON
   - Export for backup and migration

## Testing

### Unit Tests

Test individual route handlers:

```typescript
describe("Agent API", () => {
  test("should create agent with valid data", async () => {
    const response = await POST(mockRequest);
    expect(response.status).toBe(201);
  });
});
```

### Integration Tests

Test full API workflows:

```typescript
describe("Agent Lifecycle", () => {
  test("should create, update, and delete agent", async () => {
    // Create agent
    // Update properties
    // Execute commands
    // Delete agent
  });
});
```

### Load Testing

Use tools like k6 or Artillery:

```javascript
import http from "k6/http";
import { check } from "k6";

export default function () {
  const res = http.get("http://localhost:3000/api/agents");
  check(res, { "status is 200": (r) => r.status === 200 });
}
```

## Documentation

- [API Reference](../../../docs/api/agents-api.md) - Detailed endpoint documentation
- [TypeScript Types](../../../web/lib/types/agent-api.ts) - Type definitions
- [Architecture Diagrams](../../../docs/architecture/agent-api-architecture.md) - System design

## Contributing

When adding new endpoints:

1. Follow the existing pattern for route handlers
2. Add proper authentication and rate limiting
3. Include request validation with Zod schemas
4. Write comprehensive tests
5. Update documentation
6. Add TypeScript types

## Support

For questions or issues:

- Check the [main documentation](../../../docs/README.md)
- Review [troubleshooting guide](../../../docs/troubleshooting.md)
- Contact the development team
