# ADR-013: Active Inference UI and Real-time Visualization Architecture

- **Status**: Accepted
- **Date**: 2025-06-24
- **Deciders**: Active Inference UI Implementation Team
- **Category**: User Interface & Visualization Architecture
- **Impact**: High
- **Precedence**: Extends ADR-005 (Active Inference Architecture) with UI layer

## Context and Problem Statement

The FreeAgentics platform requires sophisticated user interfaces for creating, configuring, and monitoring Active Inference agents. The UI must provide:

1. **Mathematical Agent Configuration**: Template selection with real-time parameter validation
2. **Real-time Belief Visualization**: Live D3.js visualization of belief states q(s) and entropy
3. **Free Energy Monitoring**: Dynamic visualization of variational free energy F = Accuracy + Complexity
4. **Production-Ready Components**: Enterprise-grade UI components for $2M seed pitch MVP

**Technical Challenges:**

- Real-time mathematical visualization with D3.js integration
- Complex form validation for Active Inference parameters (stochastic matrices, probability constraints)
- Mathematical accuracy in UI calculations (Shannon entropy, free energy decomposition)
- Seamless API integration with comprehensive error handling

## Decision Outcome

**Chosen Architecture**: Multi-Layer Active Inference UI with Mathematical Rigor

### Core Implementation Strategy

#### 1. Agent Template Selector Architecture

**Component**: `web/components/ui/agent-template-selector.tsx`

- Template-based agent creation (Explorer, Guardian, Merchant, Scholar)
- Mathematical parameter preview with complexity indicators
- Real-time validation of precision parameters (γ, β, α)
- Interactive template comparison with mathematical foundations display

#### 2. Mathematical Configuration Form

**Component**: `web/components/ui/agent-configuration-form.tsx`

- Multi-section form with mathematical parameter configuration
- Real-time Zod validation of stochastic matrices and probability constraints
- Belief state normalization validation (Σ q(s) = 1.0 ± 1e-10)
- Precision parameter configuration with mathematical bounds

#### 3. Guided Agent Creation Wizard

**Component**: `web/components/ui/agent-creation-wizard.tsx`

- 3-step guided process: Template → Configuration → Review
- Progress tracking with mathematical foundation summary
- Integration with API service layer for seamless agent instantiation
- Comprehensive error handling and success feedback

#### 4. Real-time D3.js Visualization System

##### Belief State Visualization

**Component**: `web/components/ui/belief-state-visualization.tsx`

- Real-time q(s) probability distribution visualization
- Shannon entropy calculation: H[q(s)] = -Σ q(s) log q(s)
- Confidence metrics with probability validation
- Interactive controls with mathematical annotations

##### Free Energy Visualization

**Component**: `web/components/ui/free-energy-visualization.tsx`

- Dynamic Variational Free Energy F = Accuracy + Complexity
- Component breakdown: -E_q[ln p(o|s)] (accuracy) and D_KL[q(s)||p(s)] (complexity)
- Expected Free Energy G(π) for policy planning
- Temporal windowing with multi-tab interface

##### Integrated Dashboard

**Component**: `web/components/ui/active-inference-dashboard.tsx`

- Combined belief states, free energy, and precision parameters
- Real-time monitoring with mathematical alerts
- Performance metrics and multi-agent support
- Mathematical constraint violation notifications

#### 5. API Integration Layer

**Service**: `web/lib/api/agents-api.ts`

- Type-safe AgentsApi class following ADR-008 patterns
- Mathematical validation methods for API payloads
- Template integration with `createAgentFromTemplate()`
- Comprehensive error handling with mathematical constraint feedback

### Extended API Schema (ADR-008 Extension)

#### Active Inference Specific Endpoints

```typescript
// Belief State Management
GET / api / agents / { id } / belief - state; // Real-time belief monitoring
PUT / api / agents / { id } / belief - state; // Belief state updates

// Free Energy Monitoring
GET / api / agents / { id } / free - energy; // Variational free energy history

// Precision Parameter Management
GET / api / agents / { id } / precision; // Current precision parameters
PUT / api / agents / { id } / precision; // Precision parameter updates
```

#### Mathematical Schema Definitions

```typescript
interface BeliefState {
  beliefs: number[]; // q(s) - normalized probability distribution
  entropy: number; // H[q(s)] = -Σ q(s) log q(s)
  confidence: number; // 1 - normalized entropy
  mostLikelyState: number; // argmax q(s)
  timestamp: number; // Unix timestamp
}

interface GenerativeModel {
  A: number[][]; // Observation model (numObs × numStates)
  B: number[][][]; // Transition model (numActions × numStates × numStates)
  C: number[]; // Prior preferences over observations
  D: number[]; // Initial beliefs over states
}

interface PrecisionParameters {
  sensory: number; // γ - sensory precision [0.1, 100]
  policy: number; // β - policy precision [0.1, 100]
  state: number; // α - state precision [0.1, 100]
}
```

## Mathematical Accuracy Requirements

### Probability Constraint Validation

- **Belief States**: q(s) ∈ Δ^|S| with validation Σ q(s) = 1.0 ± 1e-10
- **Stochastic Matrices**: Row-wise probability constraint validation for A and B matrices
- **Shannon Entropy**: Accurate computation H[q(s)] = -Σ q(s) log q(s)
- **Free Energy**: Proper decomposition F = Accuracy + Complexity

### Real-time Performance

- **Visualization Update Rate**: 60 FPS for belief state animations
- **Mathematical Calculation**: <1ms for entropy and free energy computation
- **Form Validation**: Real-time constraint checking without UI blocking
- **API Response Time**: <100ms for belief state updates

## Architecture Compliance

### ADR-002: Canonical Directory Structure

- All UI components placed in `web/components/ui/`
- API services in `web/lib/api/`
- Clear separation of concerns within UI layer

### ADR-003: Clean Architecture Dependency Rules

- UI components depend on API service layer, not directly on backend
- Mathematical validation logic abstracted into reusable functions
- No business logic in UI components - proper separation of concerns

### ADR-005: Active Inference Architecture Integration

- UI layer integrates with core Active Inference engine through API
- Mathematical parameters preserved with full precision
- pymdp compatibility maintained through proper schema mapping

### ADR-008: API Interface Layer Architecture

- Extends REST API with Active Inference specific endpoints
- Maintains consistent error handling and response format
- Full OpenAPI documentation for new endpoints

## Implementation Details

### Technology Stack

- **Frontend Framework**: React 18 with TypeScript
- **UI Library**: Tailwind CSS with shadcn/ui components
- **Visualization**: D3.js v7 for mathematical visualizations
- **Form Validation**: Zod for mathematical constraint validation
- **API Client**: Custom type-safe API service layer
- **State Management**: React hooks with proper mathematical state handling

### Component Architecture

```
web/components/ui/
├── agent-template-selector.tsx      # Template selection with math preview
├── agent-configuration-form.tsx     # Mathematical parameter configuration
├── agent-creation-wizard.tsx        # Guided creation process
├── belief-state-visualization.tsx   # Real-time belief state display
├── free-energy-visualization.tsx    # Free energy monitoring
└── active-inference-dashboard.tsx   # Integrated monitoring dashboard
```

### Mathematical Validation Pipeline

1. **Client-side Validation**: Real-time Zod schema validation
2. **API Validation**: Server-side mathematical constraint checking
3. **Backend Validation**: Core engine mathematical verification
4. **Error Propagation**: Clear mathematical error messages throughout stack

## Consequences

### Positive Consequences

- **Mathematical Rigor**: All Active Inference mathematics properly implemented and validated
- **Real-time Monitoring**: Live visualization enables deep insights into agent cognition
- **Developer Experience**: Comprehensive UI tools for agent creation and monitoring
- **MVP Ready**: Production-grade components suitable for investor demonstrations
- **Extensibility**: Modular architecture supports additional mathematical visualizations

### Negative Consequences

- **Complexity**: Mathematical UI components require specialized knowledge
- **Performance**: Real-time D3.js visualizations may impact performance on low-end devices
- **Maintenance**: Mathematical accuracy requires ongoing validation and testing

### Risk Mitigation

- **Mathematical Errors**: Comprehensive test coverage for all mathematical calculations
- **Performance**: Optimization through efficient D3.js patterns and virtualization
- **Usability**: Clear documentation and intuitive mathematical parameter presentation

## Future Considerations

### Planned Enhancements

- **3D Belief Visualizations**: WebGL-based 3D visualization for high-dimensional belief spaces
- **Multi-Agent Visualization**: Coordinated belief state visualization for agent coalitions
- **Historical Analysis**: Time-series analysis tools for belief evolution
- **Export Capabilities**: Export mathematical data and visualizations for research

### Integration Points

- **pymdp Integration**: Direct integration with pymdp for advanced mathematical operations
- **Research Tools**: Integration with Jupyter notebooks for mathematical analysis
- **Edge Deployment**: UI tools for monitoring edge-deployed agents

## Validation Criteria

### Mathematical Accuracy

- All probability distributions properly normalized
- Shannon entropy calculations mathematically correct
- Free energy decomposition accurate to theoretical formulation
- Stochastic matrix constraints properly enforced

### User Experience

- Intuitive mathematical parameter configuration
- Clear visualization of complex mathematical concepts
- Responsive real-time updates without performance degradation
- Comprehensive error messages for mathematical constraint violations

### Performance

- 60 FPS visualization performance on modern browsers
- <100ms API response times for real-time updates
- Efficient memory usage for long-running visualizations
- Graceful degradation on lower-performance devices

## Related ADRs

- [ADR-002: Canonical Directory Structure](002-canonical-directory-structure.md)
- [ADR-003: Dependency Rules](003-dependency-rules.md)
- [ADR-005: Active Inference Architecture](005-active-inference-architecture.md)
- [ADR-008: API Interface Layer Architecture](008-api-interface-layer-architecture.md)

---

**Status**: This ADR documents the completed implementation of the Active Inference UI and visualization architecture, providing mathematical rigor and real-time monitoring capabilities essential for the FreeAgentics MVP demonstration.
