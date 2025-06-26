# GMN Processing Core - Architecture Diagrams

> **Important**: This documentation covers **Generative Model Notation (GMN)** for PyMDP mathematical model processing. For Graph Neural Network architecture diagrams, see the neural network documentation.

This document contains detailed architecture diagrams for the GMN Processing Core components used to parse, validate, and execute mathematical Active Inference models.

## GMN Component Interaction Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        WebUI[Web UI]
        CLI[CLI Tool]
        SDK[Python SDK]
        API_Client[API Client]
    end

    subgraph "API Gateway"
        Auth[Authentication]
        RateLimit[Rate Limiter]
        Router[Request Router]
        LoadBalancer[Load Balancer]
    end

    subgraph "GMN Core Services"
        subgraph "GMN Parser Service"
            Lexer[GMN Lexer]
            Parser[GMN Parser]
            AST[AST Builder]
            SyntaxValidator[Syntax Validator]
        end

        subgraph "GMN Validation Service"
            SchemaValidator[Schema Validator]
            ConstraintChecker[Constraint Checker]
            DependencyResolver[Dependency Resolver]
        end

        subgraph "PyMDP Mapping Service"
            MathProcessor[Mathematical Processor]
            ParameterExtractor[Parameter Extractor]
            PyMDPMapper[PyMDP Mapper]
        end

        subgraph "Active Inference Generation Service"
            BeliefBuilder[Belief Builder]
            PolicyBuilder[Policy Builder]
            ModelAssembler[Active Inference Assembler]
        end
    end

    WebUI --> Auth
    CLI --> Auth
    SDK --> Auth
    API_Client --> Auth

    Auth --> RateLimit
    RateLimit --> Router
    Router --> LoadBalancer

    LoadBalancer --> Lexer
    Lexer --> Parser
    Parser --> AST
    AST --> SyntaxValidator

    SyntaxValidator --> SchemaValidator
    SchemaValidator --> ConstraintChecker
    ConstraintChecker --> DependencyResolver

    DependencyResolver --> MathProcessor
    MathProcessor --> ParameterExtractor
    ParameterExtractor --> PyMDPMapper

    PyMDPMapper --> BeliefBuilder
    BeliefBuilder --> PolicyBuilder
    PolicyBuilder --> ModelAssembler
```

## GMN Data Processing Pipeline

```mermaid
graph LR
    subgraph "Input Stage"
        Input[.gmn.md File]
        Raw[Raw Mathematical Text]
        Tokens[Token Stream]
    end

    subgraph "Parsing Stage"
        AST[Abstract Syntax Tree]
        Validated[Validated AST]
        Normalized[Normalized Model]
    end

    subgraph "Mathematical Mapping Stage"
        MathFeatures[Mathematical Features]
        Parameters[PyMDP Parameters]
        ModelSpec[Mathematical Specification]
    end

    subgraph "Active Inference Generation Stage"
        Architecture[Active Inference Architecture]
        Components[PyMDP Components]
        Model[Complete Active Inference Model]
    end

    subgraph "Output Stage"
        Serialized[Serialized Model]
        Versioned[Versioned Model]
        Deployed[Deployment Package]
    end

    Input --> Raw
    Raw --> Tokens
    Tokens --> AST
    AST --> Validated
    Validated --> Normalized
    Normalized --> MathFeatures
    MathFeatures --> Parameters
    Parameters --> ModelSpec
    ModelSpec --> Architecture
    Architecture --> Components
    Components --> Model
    Model --> Serialized
    Serialized --> Versioned
    Versioned --> Deployed
```

## Layer Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        REST[REST API]
        GraphQL[GraphQL API]
        WebSocket[WebSocket API]
    end

    subgraph "Service Layer"
        ModelService[Model Service]
        ProcessingService[Processing Service]
        VersionService[Version Service]
        MonitoringService[Monitoring Service]
    end

    subgraph "Business Logic Layer"
        ParserLogic[Parser Logic]
        ValidationLogic[Validation Logic]
        MappingLogic[Mapping Logic]
        GenerationLogic[Generation Logic]
    end

    subgraph "Data Access Layer"
        ModelRepo[Model Repository]
        CacheManager[Cache Manager]
        StorageAdapter[Storage Adapter]
        DatabaseAdapter[Database Adapter]
    end

    subgraph "Infrastructure Layer"
        PostgreSQL[(PostgreSQL)]
        Redis[(Redis)]
        S3[S3/GCS Storage]
        MessageQueue[Message Queue]
    end

    REST --> ModelService
    GraphQL --> ProcessingService
    WebSocket --> MonitoringService

    ModelService --> ParserLogic
    ProcessingService --> ValidationLogic
    ProcessingService --> MappingLogic
    VersionService --> GenerationLogic

    ParserLogic --> ModelRepo
    ValidationLogic --> CacheManager
    MappingLogic --> StorageAdapter
    GenerationLogic --> DatabaseAdapter

    ModelRepo --> PostgreSQL
    CacheManager --> Redis
    StorageAdapter --> S3
    DatabaseAdapter --> PostgreSQL
    ModelService --> MessageQueue
```

## GMN Model Processing Flow

```mermaid
stateDiagram-v2
    [*] --> Submitted: User submits .gmn.md

    Submitted --> Parsing: Start parsing
    Parsing --> SyntaxError: Invalid syntax
    Parsing --> Parsed: Valid syntax

    SyntaxError --> [*]: Return error

    Parsed --> Validating: Validate model
    Validating --> ValidationError: Invalid model
    Validating --> Validated: Valid model

    ValidationError --> [*]: Return error

    Validated --> Mapping: Map to AI parameters
    Mapping --> MappingError: Mapping failed
    Mapping --> Mapped: Successfully mapped

    MappingError --> [*]: Return error

    Mapped --> Generating: Generate Active Inference model
    Generating --> GenerationError: Generation failed
    Generating --> Generated: Model generated

    GenerationError --> [*]: Return error

    Generated --> Optimizing: Optimize model
    Optimizing --> Optimized: Model optimized

    Optimized --> Versioning: Create version
    Versioning --> Versioned: Version created

    Versioned --> Storing: Store model
    Storing --> Stored: Model stored

    Stored --> [*]: Return success
```

## Deployment Topology

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX/HAProxy]
    end

    subgraph "API Tier"
        API1[API Instance 1]
        API2[API Instance 2]
        API3[API Instance 3]
    end

    subgraph "Processing Tier"
        subgraph "CPU Pool"
            CPU1[CPU Worker 1]
            CPU2[CPU Worker 2]
            CPU3[CPU Worker 3]
        end

        subgraph "GPU Pool"
            GPU1[GPU Worker 1]
            GPU2[GPU Worker 2]
        end
    end

    subgraph "Cache Tier"
        Redis1[Redis Primary]
        Redis2[Redis Replica 1]
        Redis3[Redis Replica 2]
    end

    subgraph "Data Tier"
        subgraph "Database Cluster"
            PG1[PostgreSQL Primary]
            PG2[PostgreSQL Replica]
        end

        subgraph "Object Storage"
            S3[S3 Bucket]
        end
    end

    LB --> API1
    LB --> API2
    LB --> API3

    API1 --> CPU1
    API1 --> GPU1
    API2 --> CPU2
    API2 --> GPU2
    API3 --> CPU3

    CPU1 --> Redis1
    CPU2 --> Redis1
    CPU3 --> Redis1
    GPU1 --> Redis1
    GPU2 --> Redis1

    Redis1 --> Redis2
    Redis1 --> Redis3

    CPU1 --> PG1
    CPU2 --> PG1
    CPU3 --> PG1

    PG1 --> PG2

    GPU1 --> S3
    GPU2 --> S3
```

## Security Architecture

```mermaid
graph TB
    subgraph "External"
        Client[Client Application]
        Attacker[Potential Attacker]
    end

    subgraph "Perimeter Security"
        WAF[Web Application Firewall]
        DDoS[DDoS Protection]
        TLS[TLS Termination]
    end

    subgraph "Application Security"
        AuthN[Authentication Service]
        AuthZ[Authorization Service]
        Validator[Input Validator]
        Sanitizer[Output Sanitizer]
    end

    subgraph "Internal Security"
        Encryption[Encryption Service]
        Secrets[Secrets Manager]
        Audit[Audit Logger]
        SIEM[SIEM Integration]
    end

    subgraph "Data Security"
        EncryptedDB[(Encrypted Database)]
        EncryptedStorage[Encrypted Storage]
        BackupVault[Backup Vault]
    end

    Client --> WAF
    Attacker --> WAF
    WAF --> DDoS
    DDoS --> TLS

    TLS --> AuthN
    AuthN --> AuthZ
    AuthZ --> Validator
    Validator --> Sanitizer

    Sanitizer --> Encryption
    Encryption --> Secrets
    Secrets --> Audit
    Audit --> SIEM

    Encryption --> EncryptedDB
    Encryption --> EncryptedStorage
    EncryptedDB --> BackupVault
    EncryptedStorage --> BackupVault
```

## Model Versioning System

```mermaid
graph LR
    subgraph "Version Tree"
        V1[Version 1.0]
        V11[Version 1.1]
        V12[Version 1.2]
        V2[Version 2.0]
        V21[Version 2.1]
        V3[Version 3.0]
    end

    subgraph "Version Metadata"
        Meta1[Metadata 1.0<br/>- Created: 2024-01-01<br/>- Author: System<br/>- Changes: Initial]
        Meta2[Metadata 2.0<br/>- Created: 2024-02-01<br/>- Author: User1<br/>- Changes: Major update]
        Meta3[Metadata 3.0<br/>- Created: 2024-03-01<br/>- Author: User2<br/>- Changes: Architecture change]
    end

    subgraph "Model Storage"
        Store1[Model Binary 1.x]
        Store2[Model Binary 2.x]
        Store3[Model Binary 3.x]
    end

    V1 --> V11
    V11 --> V12
    V1 --> V2
    V2 --> V21
    V2 --> V3

    V1 --> Meta1
    V2 --> Meta2
    V3 --> Meta3

    Meta1 --> Store1
    Meta2 --> Store2
    Meta3 --> Store3
```

## Performance Monitoring Dashboard

```mermaid
graph TB
    subgraph "Metrics Collection"
        App[Application Metrics]
        Sys[System Metrics]
        Bus[Business Metrics]
    end

    subgraph "Processing"
        Aggregator[Metric Aggregator]
        Calculator[Rate Calculator]
        Analyzer[Anomaly Analyzer]
    end

    subgraph "Storage"
        TSDB[Time Series DB]
        LogStore[Log Storage]
        TraceStore[Trace Storage]
    end

    subgraph "Visualization"
        Dashboard[Grafana Dashboard]
        Alerts[Alert Manager]
        Reports[Report Generator]
    end

    App --> Aggregator
    Sys --> Aggregator
    Bus --> Aggregator

    Aggregator --> Calculator
    Calculator --> Analyzer

    Analyzer --> TSDB
    Analyzer --> LogStore
    Analyzer --> TraceStore

    TSDB --> Dashboard
    LogStore --> Dashboard
    TraceStore --> Dashboard

    Dashboard --> Alerts
    Dashboard --> Reports
```

## Error Handling Flow

```mermaid
flowchart TB
    Start([Request Received]) --> Validate{Valid Request?}

    Validate -->|No| ValidationError[Validation Error]
    ValidationError --> LogError1[Log Error]
    LogError1 --> ReturnError1[Return 400 Bad Request]

    Validate -->|Yes| Process[Process Request]
    Process --> CheckError{Processing Error?}

    CheckError -->|Parse Error| ParseError[Parse Error]
    ParseError --> LogError2[Log Error]
    LogError2 --> ReturnError2[Return 422 Unprocessable]

    CheckError -->|System Error| SystemError[System Error]
    SystemError --> LogError3[Log Error]
    LogError3 --> Retry{Retryable?}

    Retry -->|Yes| RetryQueue[Add to Retry Queue]
    RetryQueue --> Process

    Retry -->|No| ReturnError3[Return 500 Internal Error]

    CheckError -->|No Error| Success[Process Success]
    Success --> LogSuccess[Log Success]
    LogSuccess --> ReturnSuccess[Return 200 OK]

    ReturnError1 --> End([End])
    ReturnError2 --> End
    ReturnError3 --> End
    ReturnSuccess --> End
```

## Cache Strategy

```mermaid
graph TB
    subgraph "Cache Layers"
        L1[L1: Application Cache<br/>TTL: 5 min]
        L2[L2: Redis Cache<br/>TTL: 1 hour]
        L3[L3: CDN Cache<br/>TTL: 24 hours]
    end

    subgraph "Cache Keys"
        ModelKey[model:{id}:{version}]
        ResultKey[result:{model}:{graph_hash}]
        MetaKey[meta:{model}:{type}]
    end

    subgraph "Cache Operations"
        Get[Cache Get]
        Set[Cache Set]
        Invalidate[Cache Invalidate]
        Warm[Cache Warm]
    end

    Request[Incoming Request] --> Get
    Get --> L1
    L1 -->|Miss| L2
    L2 -->|Miss| L3
    L3 -->|Miss| Database[(Database)]

    Database --> Set
    Set --> L3
    Set --> L2
    Set --> L1

    Update[Model Update] --> Invalidate
    Invalidate --> L1
    Invalidate --> L2
    Invalidate --> L3

    Schedule[Scheduled Job] --> Warm
    Warm --> Database
    Warm --> Set
```

---

These diagrams provide visual representations of the GNN Processing Core architecture, helping developers and stakeholders understand the system's structure and behavior.
