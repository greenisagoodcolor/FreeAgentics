# FreeAgentics Glossary

This glossary provides definitions for technical terms used throughout the FreeAgentics project documentation.

## A

### Active Inference
A theoretical framework from computational neuroscience that describes perception, learning, and decision-making as processes of minimizing variational free energy or surprise. In FreeAgentics, it's the core algorithm driving agent behavior, enabling agents to form beliefs about the world and select actions that minimize expected surprise.

### Agent
An autonomous software entity in the FreeAgentics system capable of perceiving its environment, making decisions, and taking actions. Agents are driven by Active Inference principles and can have different specializations (Explorer, Merchant, Scholar, Guardian).

### Agent Factory
A component responsible for creating and initializing agents with specific personalities and capabilities.

### API Gateway
The entry point for all external interactions with the FreeAgentics system, providing REST, WebSocket, and GraphQL interfaces.

## B

### Belief State
An agent's internal representation of the world, encoded as probability distributions over possible states. Beliefs are continuously updated through Active Inference as new observations are made.

### Belief Updating
The process by which an agent revises its beliefs based on new observations, using Bayesian inference principles within the Active Inference framework.

## C

### Coalition
A group of agents that have formed an alliance to achieve common goals or share resources. Coalitions are governed by contracts and can be deployed as independent units to edge devices.

### Coalition Formation
The process by which agents discover potential partners, evaluate their compatibility, negotiate terms, and form coalitions.

### Contract
A formal agreement between agents in a coalition, specifying the terms of their cooperation, resource sharing, and mutual obligations.

## D

### Dependency Rule
A principle from Clean Architecture (ADR-003) stating that dependencies should flow inward toward the core domain, ensuring that the domain logic remains independent of implementation details.

### Deployment
The process of packaging and distributing a coalition of agents to an edge device for independent operation.

## E

### Edge Deployment
The process of packaging a coalition of agents and deploying them to an edge device (like a Raspberry Pi or Jetson Nano) for independent operation without requiring a central server.

### Explorer Agent
A specialized agent type focused on discovering new territories and resources, mapping the environment, and sharing discoveries with other agents.

### Expected Free Energy
A measure used in Active Inference to evaluate potential actions based on their anticipated impact on reducing uncertainty and achieving desired outcomes.

## F

### Free Energy
In Active Inference, free energy is a measure of the difference between an agent's beliefs and reality. Minimizing free energy is equivalent to minimizing surprise or prediction error.

### Free Energy Principle
The theoretical foundation of Active Inference, stating that all adaptive systems work to minimize their free energy, which is equivalent to maximizing evidence for their model of the world.

## G

### Generative Model Notation (GMN)

A mathematical notation system used in FreeAgentics for defining Active Inference agent models in human-readable `.gmn.md` files. GMN bridges natural language specifications with PyMDP mathematical implementations.

**Key Components:**
- **GMNParser**: Parses `.gmn.md` files into structured mathematical models
- **GMNValidator**: Validates model structure and mathematical consistency  
- **GMNExecutor**: Executes Active Inference using the mathematical models
- **GMNGenerator**: Generates new models from templates and patterns

**Note**: This is distinct from **Graph Neural Networks (GNN)** used in the neural network components.

### Graph Neural Network
A type of neural network designed to operate on graph-structured data, used in FreeAgentics for processing knowledge graphs and agent relationships.

### Guardian Agent
A specialized agent type focused on protecting territories, maintaining security, coordinating defense, and responding to threats.

## H

### H3
A geospatial indexing system that uses hexagonal grids to represent the world. In FreeAgentics, H3 is used to create the spatial environment in which agents operate.

### Hexagonal Grid
The spatial representation of the world in FreeAgentics, using hexagonal cells to define locations, resources, and agent positions.

## I

### Inference Engine
The core component responsible for implementing Active Inference algorithms, including belief updating, policy selection, and free energy minimization.

## K

### Knowledge Graph
A structured representation of knowledge as a network of entities and relationships. In FreeAgentics, agents build and share knowledge graphs to represent their understanding of the world.

### Knowledge Sharing
The process by which agents exchange information, beliefs, and discoveries with each other, enhancing collective intelligence.

## L

### Large Language Model (LLM)
A type of AI model trained on vast amounts of text data that can generate human-like text. In FreeAgentics, LLMs are used to interpret natural language specifications of agent behaviors.

### Likelihood Mapping
In Active Inference, the mapping between hidden states and observations, representing how likely different observations are given particular states of the world.

## M

### Merchant Agent
A specialized agent type focused on facilitating resource trading, maintaining market equilibrium, building trade networks, and optimizing profit strategies.

## P

### Personality
A set of traits (openness, conscientiousness, extraversion, agreeableness, neuroticism) that influence an agent's behavior, preferences, and decision-making.

### Policy
A mapping from beliefs to actions, representing an agent's strategy for achieving its goals. In Active Inference, policies are selected to minimize expected free energy.

### Precision
In Active Inference, precision represents the confidence or reliability assigned to predictions or sensory inputs, affecting how strongly they influence belief updates.

### Prior Belief
An agent's belief before receiving new observations, representing its expectations about the world based on previous experiences.

## R

### Resource
Any valuable item or information in the FreeAgentics world that agents can discover, collect, trade, or use.

## S

### Scholar Agent
A specialized agent type focused on analyzing patterns, generating new knowledge, teaching other agents, and advancing collective intelligence.

### Simulation
The process of running the FreeAgentics system to observe how agents interact, form coalitions, and evolve over time.

## V

### Variational Free Energy
A mathematical quantity in Active Inference that provides an upper bound on surprise. Minimizing variational free energy is equivalent to maximizing model evidence.

## W

### World State
The complete representation of the environment at a given time, including the positions of agents, resources, and other relevant information.

### WebSocket API
A real-time communication interface allowing bidirectional communication between clients and the FreeAgentics server, used for monitoring agent activities and simulation events.

This glossary will be continuously updated as new terms and concepts are introduced to the FreeAgentics project.
