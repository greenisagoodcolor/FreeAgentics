# FreeAgentics Testing Coverage Improvement Plan

## Executive Summary
- **Current Python Coverage**: 33%
- **Current Frontend Coverage**: 0.66%
- **Target Coverage**: 80% for both
- **Timeline**: 4-6 weeks
- **Approach**: Incremental, high-impact testing focusing on critical paths

## Python Backend Testing Plan (33% → 80%)

### Phase 1: Core Infrastructure (Week 1)
**Target: 33% → 50%**

#### 1.1 Active Inference Engine (0% → 90%)
- [ ] `inference/engine/active_inference.py` - Core inference logic
- [ ] `inference/engine/generative_model.py` - Generative model implementation
- [ ] `inference/engine/hierarchical_inference.py` - Hierarchical processing
- [ ] Test belief updates, free energy calculations, action selection

#### 1.2 Agent Base Components (19% → 80%)
- [ ] `agents/base/agent.py` - Core agent functionality
- [ ] `agents/base/behaviors.py` - Behavior implementations
- [ ] `agents/base/memory.py` - Memory management
- [ ] `agents/base/perception.py` - Perception system
- [ ] Test agent lifecycle, state management, behavior execution

#### 1.3 Communication System (0% → 80%)
- [ ] `agents/base/communication.py` - Inter-agent communication
- [ ] `agents/base/interaction.py` - Interaction protocols
- [ ] Test message passing, protocol adherence, error handling

### Phase 2: Coalition & Collaboration (Week 2)
**Target: 50% → 65%**

#### 2.1 Coalition Formation (0% → 85%)
- [ ] `coalitions/formation/coalition_builder.py` - Coalition construction
- [ ] `coalitions/formation/coalition_formation_algorithms.py` - Formation algorithms
- [ ] `coalitions/formation/stability_analysis.py` - Stability checks
- [ ] Test formation strategies, stability metrics, constraint handling

#### 2.2 World Integration (0% → 80%)
- [ ] `agents/base/world_integration.py` - World model integration
- [ ] `coalitions/formation/preference_matching.py` - Preference algorithms
- [ ] Test environmental interactions, preference calculations

### Phase 3: Advanced Features (Week 3)
**Target: 65% → 80%**

#### 3.1 GNN Components (0% → 75%)
- [ ] `inference/gnn/layers.py` - GNN layer implementations
- [ ] `inference/gnn/parser.py` - Graph parsing logic
- [ ] `inference/gnn/model_mapper.py` - Model mapping functionality
- [ ] Test graph operations, layer computations, mapping accuracy

#### 3.2 Optimization & Monitoring (25% → 75%)
- [ ] `inference/engine/computational_optimization.py` - Performance optimization
- [ ] `inference/gnn/monitoring.py` - System monitoring
- [ ] Test optimization strategies, monitoring accuracy

## Frontend Testing Plan (0.66% → 80%)

### Phase 1: Core Application Structure (Week 1)
**Target: 0.66% → 25%**

#### 1.1 Application Pages & Layout
- [ ] `app/layout.tsx` - Test layout rendering, theme switching
- [ ] `app/page.tsx` - Test landing page interactions
- [ ] `app/dashboard/page.tsx` - Test dashboard data loading
- [ ] `app/agents/page.tsx` - Test agent list and creation
- [ ] `app/conversations/page.tsx` - Test conversation display
- [ ] Create integration tests for navigation flow

#### 1.2 Core Components
- [ ] `components/AgentList.tsx` - Test agent display, filtering, actions
- [ ] `components/chat-window.tsx` - Test message sending, receiving
- [ ] `components/GlobalKnowledgeGraph.tsx` - Test graph rendering
- [ ] `components/KnowledgeGraph.tsx` - Test knowledge visualization
- [ ] Mock WebSocket connections for testing

### Phase 2: API Integration & State Management (Week 2)
**Target: 25% → 50%**

#### 2.1 API Clients
- [ ] `lib/api/agents-api.ts` - Test CRUD operations
- [ ] `lib/api/dashboard-api.ts` - Test data fetching
- [ ] `lib/api/knowledge-graph.ts` - Test graph operations
- [ ] Mock API responses, test error handling

#### 2.2 LLM Integration
- [ ] `lib/llm-client.ts` - Test LLM communication
- [ ] `lib/llm-secure-client.ts` - Test security features
- [ ] `lib/conversation-orchestrator.ts` - Test orchestration logic
- [ ] Test rate limiting, error recovery, response handling

### Phase 3: Real-time Features & Hooks (Week 3)
**Target: 50% → 80%**

#### 3.1 WebSocket Hooks
- [ ] `hooks/useConversationWebSocket.ts` - Test real-time messaging
- [ ] `hooks/useKnowledgeGraphWebSocket.ts` - Test graph updates
- [ ] `hooks/useMarkovBlanketWebSocket.ts` - Test Markov updates
- [ ] Mock WebSocket server, test reconnection logic

#### 3.2 UI Components & Utilities
- [ ] `components/conversation/*.tsx` - Test conversation components
- [ ] `components/agent-dashboard.tsx` - Test dashboard features
- [ ] `lib/utils.ts` - Complete remaining 5% coverage
- [ ] Test responsive design, accessibility features

## Testing Infrastructure Setup

### Python Testing Setup
```bash
# Install testing dependencies
pip install pytest-cov pytest-mock pytest-asyncio pytest-timeout

# Create test fixtures
mkdir -p tests/fixtures
touch tests/conftest.py

# Setup coverage configuration
cat > .coveragerc << EOF
[run]
source = agents,coalitions,inference
omit = */tests/*,*/venv/*,*/migrations/*

[report]
precision = 2
show_missing = True
skip_covered = False
EOF
```

### Frontend Testing Setup
```bash
# Install testing dependencies
cd web
npm install -D @testing-library/react @testing-library/jest-dom
npm install -D @testing-library/user-event jest-websocket-mock
npm install -D msw @types/jest

# Configure Jest for better coverage
# Update jest.config.js with coverage thresholds
```

## Testing Guidelines

### Python Testing Best Practices
1. **Use pytest fixtures** for reusable test data
2. **Mock external dependencies** (LLMs, APIs, file systems)
3. **Test edge cases** and error conditions
4. **Use parametrized tests** for multiple scenarios
5. **Async testing** for WebSocket and concurrent operations

### Frontend Testing Best Practices
1. **Use React Testing Library** for component tests
2. **Mock API calls** with MSW (Mock Service Worker)
3. **Test user interactions** not implementation details
4. **Use data-testid** for reliable element selection
5. **Test accessibility** with jest-axe

## Incremental Testing Strategy

### Week 1: Foundation (0% → 25%)
- Set up testing infrastructure
- Write tests for core modules
- Establish testing patterns

### Week 2: Integration (25% → 50%)
- Test module interactions
- Add integration tests
- Mock external services

### Week 3: Features (50% → 70%)
- Test complex features
- Add end-to-end tests
- Performance testing

### Week 4: Polish (70% → 80%)
- Fill coverage gaps
- Add edge case tests
- Documentation tests

## Success Metrics

### Coverage Targets by Module Type
- **Core Logic**: 90%+ coverage
- **UI Components**: 80%+ coverage
- **Utilities**: 95%+ coverage
- **API Clients**: 85%+ coverage
- **Integration Points**: 75%+ coverage

### Quality Metrics
- All tests pass in CI/CD
- Test execution time < 5 minutes
- No flaky tests
- Clear test documentation

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Test Coverage
on: [push, pull_request]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Python Tests
        run: |
          pip install -r requirements.txt
          pytest --cov --cov-report=xml
      - name: Upload Coverage
        uses: codecov/codecov-action@v3

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Frontend Tests
        run: |
          cd web
          npm install
          npm run test:coverage
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
```

## Risk Mitigation

### Potential Challenges
1. **Complex mocking requirements** for AI/LLM interactions
2. **WebSocket testing complexity**
3. **State management in frontend tests
4. **Performance impact of comprehensive tests

### Mitigation Strategies
1. Create reusable mock utilities
2. Use jest-websocket-mock for WebSocket testing
3. Use React Testing Library's Provider patterns
4. Parallelize test execution

## Next Steps

1. **Immediate Actions**:
   - Set up testing infrastructure
   - Create base test fixtures
   - Start with highest-impact modules

2. **Week 1 Goals**:
   - Achieve 25% coverage overall
   - Establish testing patterns
   - Document testing approach

3. **Long-term Maintenance**:
   - Enforce coverage requirements in CI
   - Regular coverage reviews
   - Update tests with new features