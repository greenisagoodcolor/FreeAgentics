# FreeAgentics Architecture Documentation

Welcome to the FreeAgentics architecture documentation. This documentation provides comprehensive information about the architecture of the FreeAgentics platform.

## Architecture Overview

FreeAgentics follows a Clean Architecture approach with distinct layers:

1. **Core Domain Layer**: The heart of the system, containing the business logic and domain models
   - `agents/`: Agent implementations and behaviors
   - `inference/`: Active Inference engine and calculations
   - `coalitions/`: Coalition formation and management
   - `world/`: World simulation and physics

2. **Interface Layer**: The entry points for external systems to interact with FreeAgentics
   - `api/`: REST, WebSocket, GraphQL endpoints
   - `web/`: React frontend application

3. **Infrastructure Layer**: The technical foundations that support the application
   - `infrastructure/`: Database, external services, deployment

## Architecture Documentation

| Document                                                      | Description                                                     |
| ------------------------------------------------------------- | --------------------------------------------------------------- |
| [Architecture Diagrams](diagrams.md)                          | Visual representations of the system architecture using Mermaid |
| [Dependency Validation Guide](dependency-validation-guide.md) | Guide for validating architectural dependencies                 |
| [Developer Quick Reference](developer-quick-reference.md)     | Quick reference for developers working with the architecture    |

## Architecture Decision Records (ADRs)

Architecture Decision Records document the architectural decisions made during the development of FreeAgentics.

| ADR                                                                                      | Description                                                 |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [ADR-001: Migration Structure](decisions/001-migration-structure.md)                     | Initial migration structure from legacy codebase            |
| [ADR-002: Canonical Directory Structure](decisions/002-canonical-directory-structure.md) | Definition of the canonical directory structure             |
| [ADR-003: Dependency Rules](decisions/003-dependency-rules.md)                           | Rules governing dependencies between components             |
| [ADR-004: Naming Conventions](decisions/004-naming-conventions.md)                       | Naming conventions for files, classes, and other components |

## Key Architectural Principles

1. **Clean Architecture**: Dependencies flow inward toward the core domain
2. **Domain-Driven Design**: The structure of the codebase reflects the business domain
3. **Separation of Concerns**: Each component has a single responsibility
4. **Dependency Inversion**: High-level modules do not depend on low-level modules
5. **Testability**: The architecture supports comprehensive testing at all levels

## Validating Architectural Compliance

To validate that your code complies with the architectural rules:

1. Run the dependency validation tool: `python scripts/validate-dependencies.py`
2. Check for any violations of the dependency rules
3. Fix any violations before committing your changes

## Contributing to the Architecture

When proposing architectural changes:

1. Create a new ADR in the `docs/architecture/decisions/` directory
2. Follow the ADR template format
3. Discuss the ADR with the team
4. Once approved, update the relevant documentation

## Further Reading

- [Active Inference Guide](../active-inference-guide.md): Understanding the Active Inference principles
- [GNN Model Format](../gnn-model-format.md): Details about the GNN model format
- [API Documentation](../api/index.md): API reference documentation
