# ADR-002: Canonical Directory Structure

## Status

Accepted

## Context

Following the initial migration outlined in [ADR-001](001-structure.md), a deeper analysis revealed that the initial documentation did not fully capture the comprehensive, domain-driven structure envisioned by the expert committee in the project's PRD. The committee's intent was to establish a "screaming architecture" where the repository structure itself explains the system's purpose and boundaries.

This document serves as the single, immutable source of truth for the FreeAgentics repository structure. It is derived directly from the "PHASE 2: FINAL ARCHITECTURE" section of the PRD. All future development, file creation, and refactoring must strictly adhere to this structure.

## Decision

The official directory structure for FreeAgentics is a modular, domain-driven monolith, organized into four primary layers: **Core Domain**, **Interface**, **Infrastructure**, and **Supporting**. This structure is designed for clarity, separation of concerns, and evolutionary scalability.

## Canonical Structure Definition

The repository root `freeagentics/` shall contain the following directories, listed alphabetically.

---

### 1. `.github/`

- **Purpose**: Contains all GitHub-specific configurations, including CI/CD workflows, issue templates, and pull request templates. This centralizes repository automation and contribution management.
- **Layer**: Supporting
- **Key Contents**:
  - `workflows/`: Houses all GitHub Actions workflows (`ci.yml`, `security.yml`, `release.yml`).
  - `ISSUE_TEMPLATE/`: Defines templates for bug reports, feature requests, etc.
  - `PULL_REQUEST_TEMPLATE.md`: A template to guide contributors in creating informative pull requests.

### 2. `agents/`

- **Purpose**: The absolute core domain of the application. Contains all logic, definitions, and behaviors for the different types of autonomous agents. Each agent type is a distinct sub-domain.
- **Layer**: Core Domain
- **Key Contents**:
  - `base/`: Defines the foundational `Agent` class, shared behaviors, and common interfaces that all agents implement.
  - `explorer/`, `merchant/`, `scholar/`, `guardian/`: Individual directories for each specific agent type, containing their unique implementation (`.py`), behavior specifications, unit tests (`test_*.py`), and documentation (`README.md`).

### 3. `api/`

- **Purpose**: The primary entry point for external systems to interact with FreeAgentics. This layer is strictly for handling transport-level concerns (HTTP, WebSockets) and translating external requests into internal commands. It must remain thin and contain no business logic.
- **Layer**: Interface
- **Key Contents**:
  - `rest/`: Endpoints for the RESTful API, organized by resource (e.g., `agents/`, `coalitions/`).
  - `websocket/`: Logic for real-time communication channels.
  - `graphql/`: An optional GraphQL schema and resolvers for more complex queries.

### 4. `coalitions/`

- **Purpose**: A core domain responsible for the logic of how agents form, manage, and dissolve coalitions. This includes preference matching, contract negotiation, and resource sharing.
- **Layer**: Core Domain
- **Key Contents**:
  - `formation/`: Algorithms for preference matching, coalition building, and stability analysis.
  - `contracts/`: Logic defining the agreements and rules that govern a coalition.
  - `deployment/`: Code for packaging a coalition and its agents for future edge deployment as an independent entity.

### 5. `config/`

- **Purpose**: Centralized application configuration. This directory separates configuration from code, allowing for different settings across various environments without code changes.
- **Layer**: Supporting
- **Key Contents**:
  - `environments/`: Specific configuration files for each deployment environment (`development.yml`, `testing.yml`, etc.).
  - `database/`, `logging/`: Configuration for cross-cutting concerns.

### 6. `data/`

- **Purpose**: Contains all data-related assets that are not code. This includes database schemas, migration files, and test data fixtures.
- **Layer**: Supporting
- **Key Contents**:
  - `schemas/`: Data structure definitions (e.g., JSON schemas).
  - `migrations/`: Database migration scripts (e.g., for Alembic or Django).
  - `fixtures/`: Seed data for testing and development.

### 7. `docs/`

- **Purpose**: Houses all project documentation. Documentation is treated as a first-class citizen, essential for maintainability, onboarding, and user understanding.
- **Layer**: Supporting
- **Key Contents**:
  - `architecture/`: High-level system design documents, including ADRs in `decisions/`.
  - `api/`: Generated or manually written API documentation (e.g., OpenAPI specs).
  - `guides/`: User-facing tutorials and guides.
  - `runbooks/`: Operational guides for each environment.

### 8. `inference/`

- **Purpose**: A core domain that encapsulates the "mind" of the agents. It contains the Active Inference engine, belief update mechanisms, and integrations with other models like GNNs and LLMs.
- **Layer**: Core Domain
- **Key Contents**:
  - `engine/`: The heart of the Active Inference implementation, including belief updating and policy selection.
  - `gnn/`: Integration point for both GMN (Generative Model Notation) mathematical models and GNN (Graph Neural Network) components.
  - `llm/`: Abstractions for interacting with Large Language Models to inform agent beliefs.

### 9. `infrastructure/`

- **Purpose**: Contains all code related to the deployment and orchestration of the application. This is often referred to as "Infrastructure as Code" (IaC).
- **Layer**: Infrastructure
- **Key Contents**:
  - `docker/`: Dockerfiles for building application containers and `docker-compose.yml` for local development orchestration.
  - `kubernetes/`, `terraform/`: Future homes for production-grade deployment manifests.

### 10. `scripts/`

- **Purpose**: Holds automation and utility scripts for development, deployment, and maintenance tasks. These are not part of the application itself but support the development lifecycle.
- **Layer**: Supporting
- **Key Contents**:
  - `setup/`: Installation and initialization scripts.
  - `development/`: Scripts for running the application locally.
  - `deployment/`: Build and deployment automation scripts.

### 11. `tests/`

- **Purpose**: A centralized location for all non-unit tests. While unit tests live alongside the code they test (e.g., `agents/explorer/test_explorer.py`), this directory holds tests that span multiple components.
- **Layer**: Supporting
- **Key Contents**:
  - `integration/`: Tests for interactions between different components.
  - `behavior/`: Behavior-Driven Development (BDD) tests that describe agent scenarios.
  - `performance/`: Benchmarking and load tests.
  - `chaos/`: Tests for system resilience under failure conditions.

### 12. `web/`

- **Purpose**: The frontend application. This is a self-contained project (likely Next.js or similar) responsible for the user interface. It interacts with the backend exclusively through the `api/` layer.
- **Layer**: Interface
- **Key Contents**:
  - ``: The source code for the frontend application.
  - `public/`: Static assets like images and fonts.
  - `package.json`: Frontend project dependencies.

### 13. `world/`

- **Purpose**: A core domain that defines the environment in which the agents operate. This includes the spatial grid (H3), available resources, and the "physics" of the simulation.
- **Layer**: Core Domain
- **Key Contents**:
  - `grid/`: The hexagonal world implementation and H3 spatial indexing logic.
  - `resources/`: Definitions of resources that can exist in the world.
  - `physics/`: The rules governing interactions within the world.

### Top-Level Files

- **Purpose**: The root directory also contains critical project-wide configuration and metadata files.
- **Key Contents**:
  - `Makefile`: A common entry point for build and automation commands.
  - `pyproject.toml`: Python project configuration (PEP 621).
  - `package.json`: Root Node.js dependencies (for root-level scripts).
  - `README.md`: The primary entry point for all users and contributors.
  - `CONTRIBUTING.md`, `LICENSE`, `SECURITY.md`, `GOVERNANCE.md`: Essential community and legal documents.

## Dependency Rules

1. **Core Domain to Core Domain**: Permitted, but should be minimized. For example, `coalitions` may depend on `agents`.
2. **Interface to Core Domain**: Permitted. The `api` layer will depend heavily on the `agents`, `coalitions`, and `inference` layers to execute commands.
3. **Core Domain to Interface**: **STRICTLY FORBIDDEN**. No file within `agents`, `inference`, `coalitions`, or `world` may import from `api` or `web`. This is the cornerstone of the clean architecture.
4. **Infrastructure to Application**: Permitted. `docker` files will reference application code.
5. **Application to Infrastructure**: **STRICTLY FORBIDDEN**. No application code should know how it is deployed.
6. **Supporting to Anything**: Permitted. `scripts` and `tests` will import from all over the codebase.
7. **Anything to Supporting**: **STRICTLY FORBIDDEN**. Application code should not depend on test code or scripts.

## Consequences

- **Clarity**: The structure is now self-documenting.
- **Enforcement**: This ADR provides a clear standard against which all pull requests can be measured.
- **Scalability**: The modular design allows for independent development and future extraction of domains into microservices if necessary.
- **Discipline**: Adherence to these rules is mandatory to prevent architectural decay.

---

_Decision based on: FreeAgentics PRD, "PHASE 2: FINAL ARCHITECTURE"_
_Authored: 2025-06-19_
_Version: 1.0_

### `inference/` - Core Inference Systems

**Purpose**: Mathematical and neural network inference capabilities

**Structure**:
```
inference/
├── engine/          # Core Active Inference engine (PyMDP integration)
├── gnn/            # GMN System: Mathematical notation for PyMDP models
│   ├── parser.py   # Parse .gmn.md files to PyMDP models  
│   ├── executor.py # Execute Active Inference
│   ├── validator.py# Validate mathematical models
│   └── layers.py   # Graph Neural Network components (ML)
├── algorithms/     # Mathematical algorithms and utilities
└── llm/           # LLM integration for cognitive processes
```

**Key Distinction**:
- `gnn/`: Contains both **GMN** (Generative Model Notation for PyMDP mathematical models) and **GNN** (Graph Neural Network components for ML)
- GMN components: parser.py, executor.py, validator.py, generator.py (mathematical)  
- GNN components: layers.py, feature_extractor.py, model_mapper.py (neural networks)
