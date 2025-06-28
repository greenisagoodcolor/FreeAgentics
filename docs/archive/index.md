# FreeAgentics Documentation

Welcome to the FreeAgentics documentation. This documentation provides comprehensive information about the FreeAgentics platform, a sophisticated multi-agent simulation platform implementing Active Inference principles for emergent intelligence and collaborative behavior.

## Getting Started

- [Installation & Development Setup](guides/development-setup.md) - Install and configure FreeAgentics
- [Quick Start Guide](../docs/QUICKSTART.md) - Get started with FreeAgentics in minutes
- [Understanding Active Inference](active-inference-guide.md) - Learn about the Active Inference principles that drive agent behavior

## Core Concepts

- [Agents](guides/agent-creation.md) - Learn about the different types of agents in FreeAgentics
- [Coalitions](tutorials/coalition-formation.md) - Understand how agents form coalitions for collaborative behavior
- [Active Inference](active-inference-guide.md) - Dive deep into the Active Inference framework
- [Knowledge Graphs](guides/knowledge-graph-integration.md) - Learn how agents build and share knowledge
- [World Simulation](guides/world-simulation.md) - Understand the world environment where agents operate

## Tutorials

Our step-by-step tutorials will help you learn how to use FreeAgentics effectively:

- [Tutorial Index](tutorials/index.md) - Browse all available tutorials
- [Creating an Agent](tutorials/creating-an-agent.md) - Learn how to create your first agent
- [Coalition Formation](tutorials/coalition-formation.md) - Guide to forming coalitions between agents
- [Edge Deployment](tutorials/edge-deployment.md) - Deploy agent coalitions to edge devices

## API Reference

Comprehensive API documentation for developers:

- [API Index](api/index.md) - Browse all API documentation
- [REST API](api/rest-api.md) - REST API documentation
- [GNN API](api/gnn-api.md) - GNN API documentation
- [Agents API](api/agents-api.md) - Agents API documentation
- [OpenAPI Specification](api/openapi.yml) - OpenAPI specification

## Architecture

Detailed information about the FreeAgentics architecture:

- [Architecture Overview](architecture/index.md) - High-level overview of the architecture
- [Architecture Diagrams](architecture/diagrams.md) - Visual representations of the system architecture
- [Dependency Rules](architecture/decisions/003-dependency-rules.md) - Rules governing dependencies between components
- [Canonical Directory Structure](architecture/decisions/002-canonical-directory-structure.md) - Definition of the canonical directory structure
- [Naming Conventions](architecture/decisions/004-naming-conventions.md) - Naming conventions for files, classes, and other components

## Guides

Detailed guides for specific topics:

- [Agent Creation](guides/agent-creation.md) - Comprehensive guide to creating agents
- [Coalition Formation](tutorials/coalition-formation.md) - Guide to forming coalitions between agents
- [Knowledge Graph Integration](guides/knowledge-graph-integration.md) - Guide to integrating knowledge graphs
- [Edge Deployment](guides/edge-deployment.md) - Guide to deploying agent coalitions to edge devices

## GNN Models

Information about the Generalized Notation Notation (GNN) models:

- [GNN Model Format](gnn-model-format.md) - Specification of the GNN model format
- [GNN Architecture](architecture/gnn-architecture-diagrams.md) - Architecture of the GNN processing core
- [GNN Processing Core](architecture/gnn-processing-core.md) - Details of the GNN processing core

## Active Inference

Detailed information about the Active Inference framework:

- [Active Inference Guide](active-inference-guide.md) - Comprehensive guide to Active Inference
- [Mathematical Framework](active-inference/mathematical-framework.md) - Mathematical foundation of Active Inference
- [Implementation Guide](active-inference/implementation-guide.md) - Guide to implementing Active Inference
- [Discrete State Space](active-inference/discrete-state-space.md) - Working with discrete state spaces in Active Inference

## Testing

Information about testing in FreeAgentics:

- [Async Testing Guide](async-testing-guide.md) - Guide to async testing in FreeAgentics
- [Test Coverage](coverage.md) - Information about test coverage

## Reference

Reference documentation:

- [Glossary](glossary.md) - Glossary of technical terms used in FreeAgentics
- [Code Quality](code-quality.md) - Information about code quality standards
- [Quality Quick Reference](guides/quality-quick-reference.md) - Quick reference for quality standards
- [Quality Troubleshooting](guides/quality-troubleshooting.md) - Troubleshooting guide for quality issues

## Validation

Documentation validation tools and reports:

- [Documentation Validation Guide](validation/documentation-validation.md) - Guide to validating documentation against ADRs
- [Compliance Report](validation/compliance-report.md) - Report on documentation compliance with ADRs

## Contributing

Information for contributors:

- [Contributing Guide](../CONTRIBUTING.md) - Guide to contributing to FreeAgentics
- [Governance](../GOVERNANCE.md) - Governance model for FreeAgentics
- [Security](../SECURITY.md) - Security policy for FreeAgentics
- [Changelog](../CHANGELOG.md) - Changelog for FreeAgentics

## Core Documentation

### Active Inference & Mathematical Models

Information about the Generative Model Notation (GMN) system for PyMDP mathematical models:

- **[GMN System Overview](gnn/README.md)** - Mathematical notation for Active Inference models
- **[GMN Model Format](gnn/gnn-model-format.md)** - Specification for `.gmn.md` files
- **[Model Examples](examples/)** - Example mathematical models and usage patterns

_Note: For Graph Neural Network components, see the neural network documentation in `inference/gnn/layers.py`_
