# ADR-006: Coalition Formation Architecture

- **Status**: Accepted
- **Date**: 2025-06-20
- **Deciders**: Expert Committee (Fowler, Beck, Martin, et al.)
- **Category**: Core Domain Architecture
- **Impact**: High
- **Technical Story**: Task 7: Develop Agent System Architecture

## Context and Problem Statement

FreeAgentics agents must be able to autonomously form coalitions, negotiate agreements, and operate collaborative businesses. This requires a sophisticated system for evaluating mutual benefit, managing shared resources, coordinating actions, and eventually deploying as independent edge businesses.

The coalition formation system must support:

- Autonomous discovery and evaluation of potential collaboration opportunities
- Preference-based matching using Active Inference principles
- Dynamic coalition membership with joining and leaving capabilities
- Resource sharing and economic transaction management
- Business model implementation and profit distribution
- Edge deployment packaging for independent operation

## Decision Drivers

- **Autonomy**: Agents must form coalitions without centralized control
- **Economic Viability**: Coalitions must create real economic value
- **Scalability**: Support formation of multiple concurrent coalitions
- **Fairness**: Equitable benefit distribution among coalition members
- **Flexibility**: Support different coalition types and business models
- **Independence**: Coalitions must be deployable as standalone edge businesses
- **Mathematical Foundation**: Use Active Inference for preference matching

## Considered Options

### Option 1: Centralized Coalition Manager

- **Pros**:
  - Simple to implement and control
  - Global optimization possible
  - Easy to monitor and debug
- **Cons**:
  - Single point of failure
  - Not truly autonomous
  - Doesn't scale to thousands of agents
  - Goes against distributed agent philosophy
- **Implementation Effort**: Low

### Option 2: Market-Based Coalition Formation

- **Pros**:
  - Economically grounded
  - Self-organizing through price signals
  - Well-understood mechanism design
- **Cons**:
  - Requires complex auction mechanisms
  - May not capture all forms of collaboration value
  - Vulnerable to market manipulation
  - Doesn't leverage Active Inference naturally
- **Implementation Effort**: High

### Option 3: Active Inference Preference Matching

- **Pros**:
  - Consistent with agent cognitive architecture
  - Naturally handles uncertainty in partner evaluation
  - Agents can express complex preference structures
  - Supports dynamic belief updating about partners
  - Mathematically principled decision making
- **Cons**:
  - Complex implementation
  - Requires deep understanding of multi-agent Active Inference
- **Implementation Effort**: Medium

### Option 4: Graph-Based Coalition Discovery

- **Pros**:
  - Clear structural representation
  - Efficient algorithms available
  - Easy to visualize and understand
- **Cons**:
  - Static representation doesn't capture dynamic preferences
  - No natural uncertainty handling
  - Requires external mechanism for preference expression
- **Implementation Effort**: Medium

## Decision Outcome

**Chosen option**: "Active Inference Preference Matching" because it provides a unified mathematical framework consistent with agent cognition while supporting sophisticated preference expression and uncertainty handling.

### Implementation Strategy

1. **Core Components**:
   - PreferenceMatcher: Evaluates agent compatibility using Active Inference
   - CoalitionBuilder: Orchestrates formation process
   - ResourceManager: Handles shared resource allocation
   - BusinessModel: Implements collaborative value creation
   - ContractManager: Manages agreements and profit sharing

2. **Mathematical Foundation**:

   ```
   Coalition Value: V(C) = Σ V_i(C) + Synergy(C)
   Where:
   - V_i(C): Individual agent value in coalition C
   - Synergy(C): Emergent value from collaboration

   Preference Matching: P(i,j) = exp(-F(beliefs_i, observations_j))
   Where F is the free energy of agent i given observations about agent j
   ```

3. **Formation Process**:
   - Discovery: Agents explore and observe potential partners
   - Evaluation: Active Inference calculates expected coalition value
   - Negotiation: Iterative belief updates based on partner responses
   - Formation: Coalition creation when mutual benefit exceeds threshold
   - Operation: Collaborative business execution with profit sharing

4. **Architecture Integration**:
   - Coalition logic in `coalitions/` directory (Core Domain)
   - Business models as domain entities with clear value calculation
   - Resource management through domain abstractions
   - External world interactions via dependency-inverted interfaces

### Validation Criteria

- Mathematical verification of Pareto improvement (all members benefit)
- Performance benchmarks: Coalition formation in <500ms for 10 agents
- Economic validation: Coalitions generate measurable value over individual operation
- Stability testing: Coalitions remain stable over extended operation periods
- Edge deployment: Successful packaging and deployment as independent units

### Positive Consequences

- **Autonomous Operation**: No centralized control required for coalition formation
- **Economic Efficiency**: Coalitions form only when mutually beneficial
- **Adaptability**: Dynamic membership based on changing circumstances
- **Scalability**: Distributed formation scales to large numbers of agents
- **Mathematical Rigor**: Preference matching based on principled inference
- **Edge Deployment**: Coalitions can operate as independent businesses
- **Composability**: Different business models can be easily implemented

### Negative Consequences

- **Complexity**: Multi-agent Active Inference is mathematically complex
- **Computational Overhead**: Preference calculations require significant computation
- **Convergence Issues**: No guarantee of optimal coalition structures
- **Coordination Overhead**: Managing shared resources and decisions requires protocols

## Compliance and Enforcement

- **Validation**: Property-based tests verify Pareto improvement invariant
- **Monitoring**: Coalition health metrics track value creation and member satisfaction
- **Violations**: Failed coalitions automatically dissolve with asset redistribution

## Implementation Details

### Core Components Structure

```
coalitions/
├── core/
│   ├── coalition.py              # Main coalition entity
│   ├── preference_matcher.py     # Active Inference-based matching
│   ├── coalition_builder.py      # Formation orchestration
│   ├── resource_manager.py       # Shared resource handling
│   └── contract_manager.py       # Agreement management
├── business/
│   ├── business_model.py         # Abstract business model base
│   ├── resource_optimization.py  # Resource optimization business
│   ├── data_aggregation.py       # Data collection and analysis
│   └── service_provision.py      # Service-based businesses
├── formation/
│   ├── discovery.py              # Partner discovery algorithms
│   ├── evaluation.py             # Coalition value calculation
│   ├── negotiation.py            # Multi-agent negotiation protocols
│   └── stability_analysis.py     # Coalition stability prediction
└── deployment/
    ├── edge_packager.py          # Edge deployment packaging
    ├── deployment_manifest.py    # Deployment configuration
    └── business_launcher.py      # Independent business startup
```

### Coalition Lifecycle

1. **Discovery Phase**:
   - Agents broadcast capabilities and interests
   - Potential partners identified through spatial proximity or complementary skills
   - Initial preference calculations based on observed agent behaviors

2. **Evaluation Phase**:
   - Detailed preference matching using Active Inference
   - Expected value calculations for potential coalitions
   - Risk assessment and uncertainty quantification

3. **Negotiation Phase**:
   - Iterative communication between potential coalition members
   - Belief updates based on partner responses and proposals
   - Contract term negotiation (profit sharing, responsibilities, etc.)

4. **Formation Phase**:
   - Formal coalition creation when all parties agree
   - Initial resource contribution and business model selection
   - Legal/contractual framework establishment

5. **Operation Phase**:
   - Collaborative business execution
   - Profit generation and distribution
   - Continuous evaluation of coalition health and member satisfaction

6. **Evolution/Dissolution Phase**:
   - Dynamic membership changes (new joiners, departures)
   - Business model adaptation based on performance
   - Graceful dissolution with asset redistribution if needed

### Business Model Framework

Each coalition implements a business model that defines:

- **Value Creation**: How the coalition generates economic value
- **Resource Requirements**: What inputs are needed for operation
- **Capability Matching**: Which agent types contribute specific skills
- **Profit Distribution**: How generated value is shared among members
- **Success Metrics**: How to measure business performance

### Edge Deployment Strategy

Successful coalitions can be packaged for independent edge deployment:

- **Containerization**: Docker containers with all necessary dependencies
- **Configuration**: Environment-specific deployment parameters
- **Monitoring**: Health checks and performance monitoring
- **Updates**: Mechanism for receiving software updates
- **Independence**: Self-contained operation without central infrastructure

## Links and References

- [Shapley Value in Cooperative Game Theory](https://en.wikipedia.org/wiki/Shapley_value)
- [Multi-Agent Active Inference](https://www.frontiersin.org/articles/10.3389/fncom.2020.00020/full)
- [Task 7: Develop Agent System Architecture](../../../.taskmaster/tasks/task_007.txt)
- [ADR-005: Active Inference Architecture](005-active-inference-architecture.md)
- [ADR-002: Canonical Directory Structure](002-canonical-directory-structure.md)
- [ADR-003: Dependency Rules](003-dependency-rules.md)

---

**Economic Note**: Coalition formation is modeled as a cooperative game where agents seek to maximize their expected utility while ensuring all coalition members benefit. The Active Inference framework provides a natural mechanism for expressing and updating preferences about potential coalition partners under uncertainty.
