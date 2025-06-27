# FreeAgentics API Documentation

Welcome to the FreeAgentics API documentation. This documentation provides comprehensive information about the APIs available in the FreeAgentics platform.

## Available API Documentation

| API                                  | Description                                               | Format   |
| ------------------------------------ | --------------------------------------------------------- | -------- |
| [Agents API](agents-api.md)          | Documentation for the Agent management API endpoints      | Markdown |
| [GNN API](gnn-api.md)                | Documentation for the Graph Neural Network processing API | Markdown |
| [REST API](rest-api.md)              | Complete REST API documentation for all platform features | Markdown |
| [OpenAPI Specification](openapi.yml) | OpenAPI 3.1.0 specification for all APIs                  | YAML     |

## API Overview

The FreeAgentics API is organized into several key areas:

### Authentication

Secure access to the API using JWT tokens or API keys.

### Agents

Create, manage, and interact with autonomous agents in the system.

### World

Manage the world environment, including hexes, resources, and terrain.

### Simulation

Control the simulation, including starting, pausing, and stepping through time.

### Knowledge

Access and manipulate knowledge graphs and information sharing between agents.

### Conversations

Manage communications between agents.

### GNN Processing

Process graph data using various Graph Neural Network architectures.

## Using the API

All APIs require authentication and implement rate limiting. See the [REST API](rest-api.md) documentation for detailed information on authentication and general usage patterns.

## OpenAPI Specification

The [OpenAPI specification](openapi.yml) provides a machine-readable description of the API that can be used to generate client libraries, documentation, and test cases. You can import this specification into tools like Swagger UI, Postman, or OpenAPI Generator.

## SDK Support

Official SDKs are available for:

- Python: `pip install freeagentics-client`
- JavaScript/TypeScript: `npm install @freeagentics/client`
- Java: Maven package available
- Go: `go get github.com/freeagentics/client-go`

## API Versioning

The API uses versioning to ensure backward compatibility. The current version is v1, accessed via the `/api` base path.
