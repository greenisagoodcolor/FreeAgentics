# ADR-008: API and Interface Layer Architecture

## Status

Accepted

## Context

FreeAgentics requires a comprehensive API and interface layer that provides clean, well-documented access to the agent system. The interface layer must support multiple interaction patterns (REST, WebSocket, GraphQL) while maintaining strict adherence to the canonical directory structure and dependency rules.

## Decision

We will implement a layered API architecture with the following components:

### 1. RESTful API Layer (`api/rest/`)

- **Endpoints**: Agent management, coalition operations, simulation control, world state queries
- **Framework**: FastAPI for performance and automatic OpenAPI generation
- **Authentication**: JWT-based with API key fallback
- **Versioning**: URL-based versioning (e.g., `/api/v1/`)

### 2. WebSocket Layer (`api/websocket/`)

- **Real-time Updates**: Agent state changes, coalition events, world updates
- **Connection Management**: Automatic reconnection, heartbeat monitoring
- **Message Format**: JSON with structured event types
- **Scaling**: Redis pub/sub for multi-instance deployments

### 3. GraphQL Layer (`api/graphql/`)

- **Schema**: Unified graph of agents, coalitions, world state
- **Subscriptions**: Real-time updates via WebSocket transport
- **Optimization**: DataLoader pattern for N+1 query prevention
- **Introspection**: Full schema introspection enabled for development

### 4. Client SDKs (`api/clients/`)

- **Languages**: Python, TypeScript/JavaScript, eventually Java/C#
- **Generation**: Automated from OpenAPI/GraphQL schemas
- **Examples**: Comprehensive usage examples for each SDK
- **Documentation**: API reference with live examples

### 5. Security Layer (`api/security/`)

- **Authentication**: OAuth 2.0/JWT for user auth, API keys for service auth
- **Authorization**: Role-based access control (RBAC)
- **Rate Limiting**: Token bucket algorithm with Redis storage
- **CORS**: Configurable cross-origin resource sharing
- **Input Validation**: Strict validation with sanitization

## Implementation Details

### Directory Structure

```
api/
├── rest/
│   ├── endpoints/
│   │   ├── agents.py
│   │   ├── coalitions.py
│   │   └── world.py
│   ├── models/
│   │   ├── requests.py
│   │   └── responses.py
│   └── middleware/
├── websocket/
│   ├── handlers/
│   ├── managers/
│   └── events/
├── graphql/
│   ├── schema/
│   ├── resolvers/
│   └── subscriptions/
├── clients/
│   ├── python/
│   ├── javascript/
│   └── typescript/
├── security/
│   ├── auth/
│   ├── middleware/
│   └── utils/
└── docs/
    ├── openapi.yml
    ├── examples/
    └── guides/
```

### API Response Format

```json
{
  "success": true,
  "data": {
    "agent": {
      "id": "agent_7f3a9c",
      "name": "Explorer Alice",
      "type": "Explorer",
      "status": "active",
      "energy": 85,
      "location": "8928308280fffff"
    }
  },
  "meta": {
    "timestamp": "2025-06-20T11:15:00.000Z",
    "request_id": "req_123456"
  }
}
```

### WebSocket Message Format

```json
{
  "event": "agent.belief_updated",
  "timestamp": "2025-06-20T11:15:00.000Z",
  "data": {
    "agent_id": "agent_7f3a9c",
    "belief_state": [0.3, 0.7, 0.0],
    "confidence": 0.85
  }
}
```

## Architectural Compliance

### Dependency Rules (ADR-003)

- API layer depends on core domain through interfaces only
- No core domain dependencies on API layer
- Infrastructure components (Redis, database) accessed via dependency injection

### Directory Structure (ADR-002)

- All API components strictly placed in `api/` directory
- Clear separation between REST, WebSocket, and GraphQL implementations
- Client SDKs isolated in `api/clients/`

### Naming Conventions (ADR-004)

- RESTful endpoints use kebab-case: `/api/v1/agent-types`
- GraphQL fields use camelCase: `getAgentById`
- File names follow kebab-case: `agent-management.py`
- Class names use PascalCase: `AgentController`

## Performance Considerations

### Caching Strategy

- Redis for session storage and rate limiting
- Application-level caching for expensive operations
- HTTP caching headers for static resources

### Scaling Strategy

- Horizontal scaling with load balancer
- Stateless design for easy replication
- WebSocket sticky sessions via Redis

### Monitoring

- Request/response time metrics
- Error rate tracking
- WebSocket connection monitoring
- API usage analytics

## Security Measures

### Authentication Flow

1. User authenticates with OAuth 2.0
2. JWT token issued with expiration
3. Token validated on each request
4. Refresh token for seamless renewal

### API Key Management

- Service-to-service authentication
- Scoped permissions per key
- Usage tracking and quotas
- Automatic rotation capabilities

## Testing Strategy

### Unit Tests

- Controller logic testing
- Middleware validation
- Schema validation

### Integration Tests

- Full API flow testing
- WebSocket connection lifecycle
- Authentication/authorization

### Load Tests

- Concurrent user simulation
- WebSocket connection limits
- Rate limiting validation

## Documentation Requirements

### API Documentation

- OpenAPI 3.0 specification
- Interactive Swagger UI
- Code examples in multiple languages
- Error code reference

### Developer Guides

- Getting started tutorial
- Authentication setup guide
- WebSocket integration guide
- Best practices document

## Consequences

### Positive

- Clean separation of concerns
- Multiple interaction patterns supported
- Scalable architecture
- Comprehensive documentation
- Strong security foundation

### Negative

- Increased complexity with multiple API styles
- Additional maintenance overhead
- Learning curve for full feature set

### Risks and Mitigations

- **Risk**: API versioning complexity
  - **Mitigation**: Clear versioning strategy with deprecation timeline
- **Risk**: WebSocket scaling challenges
  - **Mitigation**: Redis pub/sub for multi-instance support
- **Risk**: Security vulnerabilities
  - **Mitigation**: Regular security audits and penetration testing

## Related Decisions

- ADR-002: Canonical Directory Structure
- ADR-003: Dependency Rules
- ADR-004: Naming Conventions
- ADR-007: Testing Strategy Architecture

## Implementation Timeline

1. **Phase 1**: Core REST API with authentication
2. **Phase 2**: WebSocket layer for real-time updates
3. **Phase 3**: GraphQL implementation
4. **Phase 4**: Client SDK generation
5. **Phase 5**: Advanced security features

This ADR ensures the API and interface layer provides a robust, scalable, and developer-friendly gateway to the FreeAgentics agent system while maintaining architectural integrity.
