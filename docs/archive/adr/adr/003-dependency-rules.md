# ADR-003: Architectural Dependency Rules

- **Status**: Accepted
- **Date**: 2025-06-20
- **Deciders**: Expert Committee (Martin, Fowler, Cockcroft, et al.)
- **Technical Story**: Task 3: Design New Directory Structure

## Context and Problem Statement

For an architecture to remain "clean" and maintainable over time, the relationships and dependencies between its components must be strictly managed. An unmanaged dependency graph leads to a "big ball of mud" where changes in one part of the system have unpredictable effects on others. This makes development slow, risky, and expensive. The FreeAgentics project requires a clear set of rules to govern how its architectural layers and modules interact, ensuring the domain core remains independent and the overall system stays decoupled and testable.

## Decision Drivers

- **Maintainability**: The primary driver is to ensure the long-term health of the codebase. A change in a volatile area (like the UI framework or a specific database) should not require changes in the stable, core business logic.
- **Testability**: The core business logic (the domain) must be testable in isolation, without dependencies on UI, databases, or external services.
- **Independence**: The domain logic should be independent of frameworks and infrastructure. This allows these external elements to be changed or upgraded with minimal impact on the core system.
- **Clarity**: The dependency flow should be simple, unidirectional, and easy for any developer to understand.

## Considered Options

1. **Unrestricted Dependencies**: Allow any module to import any other module.
   - Pro: Fast in the very short term for trivial changes.
   - Con: Leads directly to a highly coupled, unmaintainable system. This is the anti-pattern we are fixing. Completely unacceptable.
2. **Strict Layering**: Define a strict, linear hierarchy of layers where each layer can only depend on the layer directly below it.
   - Pro: Simple to understand.
   - Con: Can be too rigid. Sometimes a UI component might need a type defined in a shared/domain layer, and forcing it to go through an intermediate application layer adds unnecessary boilerplate.
3. **The Dependency Inversion Principle (Clean Architecture)**: All dependencies must flow inwards, towards the central domain logic.
   - Pro: Maximizes the independence of the core domain.
   - Pro: Aligns perfectly with the principles of Clean Architecture and Domain-Driven Design.
   - Pro: It is flexible yet powerful, providing a clear rule that handles most situations gracefully.
   - Con: Requires developers to understand the principle of dependency inversion.

## Decision Outcome

Chosen option: **The Dependency Inversion Principle**.

All dependencies in the FreeAgentics architecture must flow **inwards**. The modules are organized into conceptual layers, and dependencies may only cross layer boundaries in one direction: towards the center.

### The Dependency Rule Visualized

```
----------------------------------------------------------------------> Dependency Flow
+----------------+   +----------------+   +----------------+   +----------------+
|                |   |                |   |                |   |                |
| Infrastructure |-->|   Interfaces   |-->|   Application  |-->|      Domain    |
| (Docker, k8s)  |   |   (API, Web)   |   | (Use Cases)    |   | (Agents, Rules)|
|                |   |                |   |                |   |                |
+----------------+   +----------------+   +----------------+   +----------------+
```

_(Note: This is a conceptual model. The Application layer is an implicit set of use cases that orchestrate the domain, not necessarily a top-level directory)_

### Specific Directory-Level Rules

1. **`agents/`, `inference/`, `coalitions/`, `world/` (The Domain Core)**:
   - These directories represent the center of the architecture.
   - They **MUST NOT** depend on any other layer. They cannot import from `api`, `web`, `infrastructure`, `config`, etc.
   - They can depend on each other (e.g., `coalitions` can depend on `agents`), but this should be modeled carefully.
   - They must not have any knowledge of databases, UI frameworks, or specific external services.

2. **`api/`, `web/` (The Interface Layer)**:
   - This layer is the entry point for external systems (users, other services).
   - It **CAN** depend on the Domain Core (`agents`, etc.). It orchestrates the domain logic to fulfill requests.
   - It **MUST NOT** be depended upon by the Domain Core.
   - It contains framework-specific code (e.g., FastAPI, Next.js). The domain core knows nothing of these frameworks.

3. **`infrastructure/`, `config/`, `data/` (The Infrastructure & Configuration Layer)**:
   - This layer contains concrete implementations and configurations (e.g., Dockerfiles, database connection details, environment variables).
   - It is the most volatile and outermost layer.
   - It "depends" on the rest of the application in the sense that it provides the environment and tools to run it.
   - No part of the application logic (Domain or Interface) should depend directly on this layer. Instead, interfaces defined in the domain are implemented by classes that might use configuration from this layer.

### Positive Consequences

- **A Stable Core**: The most important business logic is protected from changes in volatile, external components.
- **Component Swapping**: The database, UI framework, or any external service can be replaced with minimal impact on the core system.
- **Enhanced Testability**: The domain logic can be tested without a database or UI, leading to faster and more reliable tests.
- **Parallel Development**: Teams can work on the UI and the domain core simultaneously, as their dependency relationship is a one-way street.

### Negative Consequences

- Requires discipline. The rules must be enforced through code reviews and, ideally, automated tooling, to prevent architectural drift.
