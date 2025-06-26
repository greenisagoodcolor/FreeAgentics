# FreeAgentics Architecture Diagrams

This document provides visual representations of the FreeAgentics architecture using Mermaid diagrams.

## System Architecture Overview

The following diagram illustrates the high-level architecture of FreeAgentics, showing the main components and their relationships.

```mermaid
graph TB
    subgraph "Interface Layer"
        API["API Gateway<br/>(api/)"]
        Web["Web Frontend<br/>(web/)"]
    end

    subgraph "Core Domain Layer"
        Agents["Agent System<br/>(agents/)"]
        Inference["Inference Engine<br/>(inference/)"]
        Coalitions["Coalition Management<br/>(coalitions/)"]
        World["World Simulation<br/>(world/)"]
    end

    subgraph "Infrastructure Layer"
        DB["Database<br/>(infrastructure/database/)"]
        Deploy["Deployment<br/>(infrastructure/deployment/)"]
        Docker["Containerization<br/>(infrastructure/docker/)"]
        Export["Export System<br/>(infrastructure/export/)"]
        Hardware["Hardware Abstraction<br/>(infrastructure/hardware/)"]
    end

    %% Interface to Core Domain connections
    API --> Agents
    API --> Inference
    API --> Coalitions
    API --> World
    Web --> API

    %% Core Domain internal connections
    Agents --> Inference
    Coalitions --> Agents
    World --> Agents

    %% Infrastructure connections
    DB --> Agents
    DB --> Coalitions
    DB --> World
    Deploy --> Coalitions
    Export --> Coalitions
    Hardware --> Agents
```

## Clean Architecture Dependency Flow

This diagram illustrates the dependency rules as defined in ADR-003, showing how dependencies flow inward toward the core domain.

```mermaid
graph LR
    subgraph "Infrastructure Layer"
        DB["Database"]
        Deploy["Deployment"]
        Docker["Containerization"]
    end

    subgraph "Interface Layer"
        API["API Gateway"]
        Web["Web Frontend"]
    end

    subgraph "Core Domain Layer"
        Agents["Agents"]
        Inference["Inference"]
        Coalitions["Coalitions"]
        World["World"]
    end

    %% Dependency flow (inward arrows)
    Web --> API
    API --> Agents
    API --> Inference
    API --> Coalitions
    API --> World

    DB -.-> Agents
    DB -.-> Coalitions
    Deploy -.-> Coalitions
    Docker -.-> API
    Docker -.-> Web

    %% Internal domain dependencies
    Agents --> Inference
    Coalitions --> Agents
    World --> Agents

    %% Legend
    classDef infra fill:#e9a3c9,stroke:#333,stroke-width:1px
    classDef interface fill:#a1d76a,stroke:#333,stroke-width:1px
    classDef domain fill:#7fc7ff,stroke:#333,stroke-width:1px

    class DB,Deploy,Docker infra
    class API,Web interface
    class Agents,Inference,Coalitions,World domain

    %% Note: Dotted lines represent implementation details, not direct code dependencies
```

## Component Diagram

The following diagram shows the main components of the system and their interactions.

```mermaid
graph TB
    subgraph "Web Application"
        Dashboard["Dashboard"]
        AgentCreator["Agent Creator"]
        SimControl["Simulation Controls"]
        WorldView["World Visualization"]
    end

    subgraph "API Layer"
        REST["REST API"]
        WebSocket["WebSocket API"]
        GraphQL["GraphQL API"]
    end

    subgraph "Agent System"
        AgentFactory["Agent Factory"]
        Explorer["Explorer Agents"]
        Merchant["Merchant Agents"]
        Scholar["Scholar Agents"]
        Guardian["Guardian Agents"]
    end

    subgraph "Inference Engine"
        ActiveInference["Active Inference"]
        GNN["Graph Neural Networks"]
        LLM["LLM Integration"]
    end

    subgraph "World System"
        HexGrid["H3 Hexagonal Grid"]
        Resources["Resource Management"]
        Physics["World Physics"]
    end

    subgraph "Coalition System"
        Formation["Coalition Formation"]
        Contracts["Coalition Contracts"]
        Deployment["Edge Deployment"]
    end

    %% Web to API connections
    Dashboard --> REST
    Dashboard --> WebSocket
    AgentCreator --> REST
    SimControl --> REST
    SimControl --> WebSocket
    WorldView --> WebSocket
    WorldView --> GraphQL

    %% API to Core connections
    REST --> AgentFactory
    REST --> ActiveInference
    REST --> Formation
    REST --> HexGrid
    WebSocket --> Explorer
    WebSocket --> Merchant
    WebSocket --> Scholar
    WebSocket --> Guardian
    GraphQL --> HexGrid
    GraphQL --> Resources

    %% Core internal connections
    AgentFactory --> Explorer
    AgentFactory --> Merchant
    AgentFactory --> Scholar
    AgentFactory --> Guardian

    Explorer --> ActiveInference
    Merchant --> ActiveInference
    Scholar --> ActiveInference
    Guardian --> ActiveInference

    ActiveInference --> GNN
    ActiveInference --> LLM

    Formation --> Explorer
    Formation --> Merchant
    Formation --> Scholar
    Formation --> Guardian
    Formation --> Contracts
    Contracts --> Deployment

    Explorer --> HexGrid
    Explorer --> Resources
    Merchant --> Resources
    HexGrid --> Physics
    Resources --> Physics
```

## Agent Interaction Flow

This diagram illustrates how agents interact with each other and the environment.

```mermaid
sequenceDiagram
    participant Explorer as Explorer Agent
    participant World as World System
    participant Inference as Inference Engine
    participant Merchant as Merchant Agent
    participant Coalition as Coalition System

    Explorer->>Inference: Update beliefs
    Inference->>Explorer: Return policy
    Explorer->>World: Move to new location
    World->>Explorer: Return observations
    Explorer->>Inference: Process observations
    Inference->>Explorer: Update beliefs
    Explorer->>World: Discover resource

    Explorer->>Merchant: Share resource location
    Merchant->>Inference: Update beliefs
    Inference->>Merchant: Return policy
    Merchant->>World: Move to resource
    World->>Merchant: Return observations

    Merchant->>Explorer: Propose coalition
    Explorer->>Inference: Evaluate proposal
    Inference->>Explorer: Decision (accept)
    Explorer->>Merchant: Accept proposal

    Merchant->>Coalition: Form coalition
    Coalition->>Merchant: Return contract
    Coalition->>Explorer: Return contract
```

## Data Flow Diagram

This diagram shows how data flows through the system.

```mermaid
graph LR
    %% External entities
    User["User"]
    Edge["Edge Device"]

    %% Processes
    WebUI["Web UI"]
    API["API Gateway"]
    AgentSystem["Agent System"]
    InferenceEngine["Inference Engine"]
    WorldSim["World Simulation"]
    CoalitionMgr["Coalition Manager"]

    %% Data stores
    AgentDB[("Agent Database")]
    WorldDB[("World State DB")]
    KnowledgeGraph[("Knowledge Graph")]
    CoalitionDB[("Coalition DB")]

    %% Data flows
    User -- "Commands, Queries" --> WebUI
    WebUI -- "API Requests" --> API
    API -- "Agent Commands" --> AgentSystem
    API -- "World Queries" --> WorldSim
    API -- "Coalition Requests" --> CoalitionMgr

    AgentSystem -- "Store Agent State" --> AgentDB
    AgentSystem -- "Belief Updates" --> InferenceEngine
    AgentSystem -- "World Interactions" --> WorldSim
    AgentSystem -- "Store Knowledge" --> KnowledgeGraph

    InferenceEngine -- "Retrieve Knowledge" --> KnowledgeGraph
    InferenceEngine -- "Return Policies" --> AgentSystem

    WorldSim -- "Store World State" --> WorldDB
    WorldSim -- "Return Observations" --> AgentSystem

    CoalitionMgr -- "Query Agents" --> AgentSystem
    CoalitionMgr -- "Store Contracts" --> CoalitionDB
    CoalitionMgr -- "Deploy Coalition" --> Edge

    %% Legend
    classDef external fill:#f9f,stroke:#333,stroke-width:1px
    classDef process fill:#bbf,stroke:#333,stroke-width:1px
    classDef datastore fill:#dfd,stroke:#333,stroke-width:1px

    class User,Edge external
    class WebUI,API,AgentSystem,InferenceEngine,WorldSim,CoalitionMgr process
    class AgentDB,WorldDB,KnowledgeGraph,CoalitionDB datastore
```

## Active Inference Process

This diagram illustrates the Active Inference process that drives agent behavior, now powered by the validated PyMDP library.

```mermaid
graph TD
    Prior["Prior Beliefs (D)<br/>(Initial state expectations)"]
    Observation["Observations<br/>(Sensory input)"]
    Likelihood["Observation Model (A)<br/>(How states generate observations)"]
    Transition["Transition Model (B)<br/>(How actions change states)"]
    Preferences["Preferences (C)<br/>(Goal states and utilities)"]

    PyMDPEngine["PyMDP Inference Engine<br/>(Validated algorithms)"]
    BeliefUpdate["Belief Update<br/>(Variational message passing)"]
    PolicySelection["Policy Selection<br/>(Expected free energy minimization)"]

    Action["Action Execution"]
    NewObservation["New Observations"]
    UpdatedBeliefs["Updated Beliefs<br/>(Posterior distribution)"]

    %% PyMDP Integration Flow
    Prior --> PyMDPEngine
    Observation --> PyMDPEngine
    Likelihood --> PyMDPEngine
    Transition --> PyMDPEngine
    Preferences --> PyMDPEngine

    PyMDPEngine --> BeliefUpdate
    PyMDPEngine --> PolicySelection

    BeliefUpdate --> UpdatedBeliefs
    PolicySelection --> Action
    Action --> NewObservation
    NewObservation --> Observation
    UpdatedBeliefs --> Prior

    %% Fallback Mechanism
    PyMDPFallback["Non-PyMDP Fallback<br/>(Robust error handling)"]
    PyMDPEngine -.-> PyMDPFallback
    PyMDPFallback -.-> PolicySelection

    %% Style
    classDef pymdp fill:#e6f3ff,stroke:#0066cc,stroke-width:2px
    classDef model fill:#ffcccc,stroke:#333,stroke-width:1px
    classDef process fill:#ccffcc,stroke:#333,stroke-width:1px
    classDef action fill:#ccccff,stroke:#333,stroke-width:1px
    classDef fallback fill:#fff2cc,stroke:#d6b656,stroke-width:1px,stroke-dasharray: 5 5

    class PyMDPEngine pymdp
    class Prior,UpdatedBeliefs,Likelihood,Transition,Preferences model
    class BeliefUpdate,PolicySelection process
    class Action,NewObservation,Observation action
    class PyMDPFallback fallback
```

## PyMDP Integration Architecture

This diagram shows the detailed architecture of our PyMDP integration with robust fallback mechanisms.

```mermaid
graph TB
    subgraph "Agent Interface Layer"
        AgentAPI["Agent API<br/>(Backward Compatible)"]
        PolicyAdapter["PyMDPPolicyAdapter<br/>(Compatibility Layer)"]
    end

    subgraph "PyMDP Core Integration"
        PyMDPAgent["PyMDP Agent<br/>(Cached Instances)"]
        PyMDPGenerativeModel["PyMDPGenerativeModel<br/>(A, B, C, D matrices)"]
        PyMDPPolicySelector["PyMDPPolicySelector<br/>(Validated algorithms)"]
    end

    subgraph "Robust Fallback System"
        ErrorDetection["Error Detection<br/>(Einstein summation, matrix issues)"]
        FallbackCalculator["Non-PyMDP Calculator<br/>(Functional alternatives)"]
        AgentCache["Agent Cache<br/>(Performance optimization)"]
    end

    subgraph "Legacy Compatibility"
        DiscreteExpectedFreeEnergy["DiscreteExpectedFreeEnergy<br/>(Alias to PyMDPPolicyAdapter)"]
        BackwardCompatAPI["Backward Compatibility API<br/>(Maintains existing interfaces)"]
    end

    %% Main flow
    AgentAPI --> PolicyAdapter
    PolicyAdapter --> PyMDPAgent
    PolicyAdapter --> PyMDPGenerativeModel
    PolicyAdapter --> PyMDPPolicySelector

    %% Caching and optimization
    PyMDPPolicySelector --> AgentCache
    AgentCache --> PyMDPAgent

    %% Error handling and fallback
    PyMDPPolicySelector --> ErrorDetection
    ErrorDetection --> FallbackCalculator
    FallbackCalculator --> PolicyAdapter

    %% Legacy support
    DiscreteExpectedFreeEnergy --> PolicyAdapter
    BackwardCompatAPI --> AgentAPI

    %% Style
    classDef interface fill:#e6f3ff,stroke:#0066cc,stroke-width:2px
    classDef pymdp fill:#ccffcc,stroke:#009900,stroke-width:2px
    classDef fallback fill:#fff2cc,stroke:#d6b656,stroke-width:2px
    classDef legacy fill:#ffcccc,stroke:#cc0000,stroke-width:1px,stroke-dasharray: 5 5

    class AgentAPI,PolicyAdapter interface
    class PyMDPAgent,PyMDPGenerativeModel,PyMDPPolicySelector pymdp
    class ErrorDetection,FallbackCalculator,AgentCache fallback
    class DiscreteExpectedFreeEnergy,BackwardCompatAPI legacy
```

## Deployment Architecture

This diagram shows the deployment architecture of the system.

```mermaid
graph TB
    subgraph "Cloud Infrastructure"
        API["API Server"]
        WebServer["Web Server"]
        DB["Database Cluster"]
        Queue["Message Queue"]
        Storage["Object Storage"]
    end

    subgraph "Edge Devices"
        RPi["Raspberry Pi"]
        Jetson["Jetson Nano"]
        Custom["Custom Hardware"]
    end

    subgraph "Client Devices"
        Browser["Web Browser"]
        Mobile["Mobile App"]
        Desktop["Desktop App"]
    end

    %% Connections
    Browser --> WebServer
    Mobile --> API
    Desktop --> API

    WebServer --> API
    API --> DB
    API --> Queue
    API --> Storage

    Queue --> RPi
    Queue --> Jetson
    Queue --> Custom

    RPi --> API
    Jetson --> API
    Custom --> API

    %% Style
    classDef cloud fill:#b3e6ff,stroke:#333,stroke-width:1px
    classDef edge fill:#ffcccc,stroke:#333,stroke-width:1px
    classDef client fill:#ccffcc,stroke:#333,stroke-width:1px

    class API,WebServer,DB,Queue,Storage cloud
    class RPi,Jetson,Custom edge
    class Browser,Mobile,Desktop client
```

## Coalition Formation Process

This diagram illustrates the coalition formation process.

```mermaid
graph TD
    Discovery["Agent Discovery"]
    Evaluation["Preference Matching & Evaluation"]
    Negotiation["Contract Negotiation"]
    Formation["Coalition Formation"]
    Operation["Collaborative Operation"]
    Deployment["Edge Deployment"]

    Discovery --> Evaluation
    Evaluation --> Negotiation
    Negotiation --> Formation
    Formation --> Operation
    Operation --> Deployment

    %% Style
    classDef process fill:#f9f9f9,stroke:#333,stroke-width:1px

    class Discovery,Evaluation,Negotiation,Formation,Operation,Deployment process
```

## Knowledge Graph Structure

This diagram shows the structure of the knowledge graph used by agents.

```mermaid
graph TD
    subgraph "Knowledge Graph"
        Agent1["Agent Node"]
        Agent2["Agent Node"]
        Location1["Location Node"]
        Location2["Location Node"]
        Resource1["Resource Node"]
        Resource2["Resource Node"]
        Pattern1["Pattern Node"]
        Pattern2["Pattern Node"]
    end

    Agent1 -- "KNOWS" --> Agent2
    Agent1 -- "VISITED" --> Location1
    Agent1 -- "DISCOVERED" --> Resource1
    Agent2 -- "VISITED" --> Location2
    Agent2 -- "DISCOVERED" --> Resource2

    Location1 -- "CONTAINS" --> Resource1
    Location2 -- "CONTAINS" --> Resource2

    Agent1 -- "LEARNED" --> Pattern1
    Agent2 -- "LEARNED" --> Pattern2

    Pattern1 -- "APPLIES_TO" --> Resource1
    Pattern2 -- "APPLIES_TO" --> Resource2

    %% Style
    classDef agent fill:#ffcccc,stroke:#333,stroke-width:1px
    classDef location fill:#ccffcc,stroke:#333,stroke-width:1px
    classDef resource fill:#ccccff,stroke:#333,stroke-width:1px
    classDef pattern fill:#ffffcc,stroke:#333,stroke-width:1px

    class Agent1,Agent2 agent
    class Location1,Location2 location
    class Resource1,Resource2 resource
    class Pattern1,Pattern2 pattern
```

## Module Dependencies

This diagram shows the dependencies between the main modules of the system, adhering to ADR-003.

```mermaid
graph TD
    %% Core Domain modules
    Agents["agents/"]
    Inference["inference/"]
    Coalitions["coalitions/"]
    World["world/"]

    %% Interface modules
    API["api/"]
    Web["web/"]

    %% Infrastructure modules
    Infrastructure["infrastructure/"]
    Config["config/"]
    Data["data/"]

    %% Supporting modules
    Scripts["scripts/"]
    Tests["tests/"]
    Docs["docs/"]

    %% Interface to Core Domain dependencies
    API --> Agents
    API --> Inference
    API --> Coalitions
    API --> World
    Web --> API

    %% Core Domain internal dependencies
    Coalitions --> Agents
    Coalitions --> World
    Agents --> Inference
    Agents --> World

    %% Infrastructure dependencies
    Infrastructure -.-> Agents
    Infrastructure -.-> Inference
    Infrastructure -.-> Coalitions
    Infrastructure -.-> World
    Infrastructure -.-> API
    Infrastructure -.-> Web

    Config -.-> Agents
    Config -.-> Inference
    Config -.-> Coalitions
    Config -.-> World
    Config -.-> API
    Config -.-> Web

    %% Supporting dependencies
    Scripts -.-> Agents
    Scripts -.-> Inference
    Scripts -.-> Coalitions
    Scripts -.-> World
    Scripts -.-> API
    Scripts -.-> Web

    Tests -.-> Agents
    Tests -.-> Inference
    Tests -.-> Coalitions
    Tests -.-> World
    Tests -.-> API

    %% Style
    classDef core fill:#ffcccc,stroke:#333,stroke-width:1px
    classDef interface fill:#ccffcc,stroke:#333,stroke-width:1px
    classDef infra fill:#ccccff,stroke:#333,stroke-width:1px
    classDef support fill:#ffffcc,stroke:#333,stroke-width:1px

    class Agents,Inference,Coalitions,World core
    class API,Web interface
    class Infrastructure,Config,Data infra
    class Scripts,Tests,Docs support

    %% Note: Dotted lines represent implementation details, not direct code dependencies
```

## Class Diagram: Agent System

This diagram shows the class structure of the agent system.

```mermaid
classDiagram
    class Agent {
        +UUID id
        +String name
        +AgentState state
        +Personality personality
        +Position position
        +Resources resources
        +initialize()
        +updateBeliefs()
        +selectAction()
        +executeAction()
        +perceive()
        +communicate()
    }

    class BaseAgent {
        +BeliefState beliefs
        +KnowledgeGraph knowledge
        +updateBeliefs(Observation)
        +selectAction() Policy
        +executeAction(Action)
    }

    class ExplorerAgent {
        +exploreTerritory()
        +mapEnvironment()
        +shareDiscoveries()
    }

    class MerchantAgent {
        +evaluateResources()
        +negotiateTrade()
        +optimizeProfit()
    }

    class ScholarAgent {
        +analyzePatterns()
        +generateKnowledge()
        +teachOthers()
    }

    class GuardianAgent {
        +protectTerritory()
        +assessThreats()
        +coordinateDefense()
    }

    class AgentFactory {
        +createAgent(AgentType, Personality) Agent
        +loadAgent(UUID) Agent
        +saveAgent(Agent)
    }

    class Personality {
        +float openness
        +float conscientiousness
        +float extraversion
        +float agreeableness
        +float neuroticism
    }

    Agent <|-- BaseAgent
    BaseAgent <|-- ExplorerAgent
    BaseAgent <|-- MerchantAgent
    BaseAgent <|-- ScholarAgent
    BaseAgent <|-- GuardianAgent

    AgentFactory --> Agent : creates
    Agent *-- Personality
```

These diagrams provide a comprehensive visual representation of the FreeAgentics architecture, showing how the different components interact and how data flows through the system.
