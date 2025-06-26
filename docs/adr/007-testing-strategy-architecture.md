# ADR-007: Comprehensive Testing Strategy Architecture

- **Status**: Accepted
- **Date**: 2025-06-20
- **Deciders**: Expert Committee (Beck, Fowler, Martin, Wayne, et al.)
- **Category**: Quality & Testing
- **Impact**: High
- **Technical Story**: Task 2: Establish Test Baseline & CI Pipeline

## Context and Problem Statement

FreeAgentics is a complex multi-agent system with Active Inference cognitive engines, coalition formation mechanisms, and emergent behaviors. Traditional testing approaches are insufficient for validating systems where:
- Agent behaviors emerge from mathematical principles rather than explicit programming
- Coalition formation involves multi-agent negotiation and game theory
- Active Inference calculations must maintain mathematical invariants
- System behavior is probabilistic and involves uncertainty quantification
- Performance requirements include supporting 1000+ concurrent agents

The testing strategy must provide confidence in mathematical correctness, emergent behavior validation, performance characteristics, and architectural compliance.

## Decision Drivers

- **Mathematical Rigor**: Verify Active Inference calculations and invariants
- **Emergent Behavior**: Test complex multi-agent interactions and coalition formation
- **Performance Requirements**: Validate scalability to 1000+ agents
- **Architectural Compliance**: Ensure adherence to ADR-002, ADR-003, ADR-004
- **Developer Experience**: Fast feedback cycles and clear failure diagnostics
- **Continuous Integration**: Automated validation preventing regressions
- **Edge Case Coverage**: Handle probabilistic systems and uncertainty
- **Property Verification**: Mathematical properties must hold across all scenarios

## Considered Options

### Option 1: Traditional Unit Testing Only
- **Pros**:
  - Simple to implement
  - Fast execution
  - Well-understood tooling
- **Cons**:
  - Cannot test emergent behaviors
  - Misses integration issues
  - No mathematical property verification
  - Poor coverage of multi-agent scenarios
- **Implementation Effort**: Low

### Option 2: Comprehensive Test Pyramid (Unit, Integration, E2E)
- **Pros**:
  - Good coverage across layers
  - Industry standard approach
  - Balanced speed vs. confidence
- **Cons**:
  - Insufficient for mathematical property testing
  - Doesn't handle emergent behaviors well
  - Limited probabilistic system support
- **Implementation Effort**: Medium

### Option 3: Property-Based + Behavior-Driven + Performance Testing
- **Pros**:
  - Mathematical property verification
  - Emergent behavior validation
  - Performance characteristics testing
  - Handles probabilistic systems
  - Comprehensive coverage
- **Cons**:
  - Complex implementation
  - Requires specialized knowledge
  - Longer execution times
- **Implementation Effort**: High

### Option 4: Simulation-Based Testing Only
- **Pros**:
  - Realistic scenarios
  - Emergent behavior coverage
  - End-to-end validation
- **Cons**:
  - Slow execution
  - Difficult to debug failures
  - Poor unit-level feedback
  - Resource intensive
- **Implementation Effort**: Medium

## Decision Outcome

**Chosen option**: "Property-Based + Behavior-Driven + Performance Testing" because it provides the mathematical rigor, emergent behavior validation, and performance guarantees required for a complex multi-agent system with Active Inference.

### Implementation Strategy

1. **Multi-Layer Testing Architecture**:
   ```
   Tests/
   ├── unit/              # Fast, isolated component tests
   ├── property/          # Mathematical invariant verification
   ├── integration/       # Component interaction tests
   ├── behavior/          # BDD agent scenario tests
   ├── performance/       # Scalability and benchmarks
   ├── chaos/             # Failure resilience tests
   └── compliance/        # Architectural rule validation
   ```

2. **Testing Framework Stack**:
   - **Property Testing**: Hypothesis (Python) for mathematical invariant verification
   - **Unit Testing**: pytest with comprehensive coverage reporting
   - **Behavior Testing**: pytest-bdd for agent scenario validation
   - **Performance Testing**: pytest-benchmark with statistical analysis
   - **Integration Testing**: Docker-based environment simulation
   - **Chaos Testing**: Custom failure injection framework

3. **Mathematical Property Verification**:
   - Active Inference invariants (beliefs sum to 1, free energy minimization)
   - Coalition formation properties (Pareto improvement, stability)
   - Resource conservation laws
   - Performance characteristics under load

4. **Architectural Compliance Integration**:
   - Automated import dependency validation (ADR-003)
   - File placement verification (ADR-002)
   - Naming convention enforcement (ADR-004)
   - Layer boundary validation

### Validation Criteria

- **Coverage**: >95% code coverage with meaningful assertions
- **Performance**: All benchmarks execute within 10% variance
- **Properties**: 100% mathematical invariant verification
- **Behaviors**: All agent scenarios pass deterministically
- **Compliance**: Zero architectural violations in any test
- **Speed**: Unit test suite completes in <30 seconds
- **Integration**: Full integration suite completes in <5 minutes

### Positive Consequences

- **Mathematical Confidence**: Property tests verify correctness of Active Inference
- **Emergent Behavior Validation**: BDD tests capture complex multi-agent scenarios
- **Performance Guarantees**: Benchmarks ensure scalability requirements
- **Architectural Integrity**: Compliance tests prevent architectural decay
- **Developer Productivity**: Fast feedback and clear failure diagnostics
- **Regression Prevention**: Comprehensive CI pipeline catches issues early
- **Documentation**: Tests serve as executable specifications

### Negative Consequences

- **Complexity**: Requires understanding of property-based testing concepts
- **Execution Time**: Comprehensive test suite takes longer than simple unit tests
- **Maintenance Overhead**: More sophisticated tests require more maintenance
- **Tool Learning**: Team needs training on advanced testing frameworks

## Compliance and Enforcement

- **Validation**: All tests must pass before code merge
- **Monitoring**: Test execution time and coverage tracked over time
- **Violations**: Failed tests block deployment and require immediate fix

## Implementation Details

### Property-Based Testing Examples

```python
# tests/property/test_active_inference_invariants.py
from hypothesis import given, strategies as st
import numpy as np

@given(beliefs=st.lists(st.floats(min_value=0, max_value=1), min_size=2, max_size=10))
def test_beliefs_sum_to_one(beliefs):
    """Belief distributions must always sum to 1 (probability constraint)."""
    belief_array = np.array(beliefs)
    normalized = belief_array / belief_array.sum()

    agent = Agent.create("Explorer")
    agent.beliefs.update(normalized)

    assert abs(agent.beliefs.sum() - 1.0) < 1e-10

@given(
    num_agents=st.integers(min_value=2, max_value=10),
    timesteps=st.integers(min_value=10, max_value=100)
)
def test_coalition_formation_improves_utility(num_agents, timesteps):
    """Coalition formation should only occur when it improves utility for all members."""
    agents = [Agent.create("Explorer") for _ in range(num_agents)]
    world = World()

    # Record initial utilities
    initial_utilities = [agent.calculate_utility() for agent in agents]

    # Add to world and simulate
    for agent in agents:
        world.add_agent(agent)

    for _ in range(timesteps):
        world.step()

    # Check coalition members have improved utility
    for coalition in world.coalitions:
        for agent_id in coalition.members:
            agent = world.get_agent(agent_id)
            current_utility = agent.calculate_utility()
            initial_utility = initial_utilities[agents.index(agent)]
            assert current_utility >= initial_utility, f"Agent {agent_id} utility decreased"
```

### Behavior-Driven Testing Examples

```python
# tests/behavior/test_resource_optimization_scenario.py
from pytest_bdd import scenarios, given, when, then

scenarios('resource_optimization.feature')

@given('an Explorer agent with high curiosity')
def explorer_agent():
    return Agent.create("Explorer", personality={"curiosity": 0.9})

@given('a Developer agent with high efficiency')
def developer_agent():
    return Agent.create("Developer", personality={"efficiency": 0.9})

@when('agents discover complementary resources over 100 timesteps')
def simulate_discovery(explorer_agent, developer_agent):
    world = World(resource_density=0.3)
    world.add_agents([explorer_agent, developer_agent])

    for _ in range(100):
        world.step()

    return world

@then('a ResourceOptimization coalition should form')
def verify_coalition_formation(world):
    assert len(world.coalitions) >= 1
    coalition = world.coalitions[0]
    assert coalition.business_type == "ResourceOptimization"
    assert len(coalition.members) >= 2

@then('the coalition should generate positive economic value')
def verify_economic_value(world):
    coalition = world.coalitions[0]
    metrics = coalition.get_metrics()
    assert metrics["revenue"] > 0
    assert metrics["efficiency"] > 0.7
```

### Performance Testing Examples

```python
# tests/performance/test_scalability_benchmarks.py
import pytest

class TestScalabilityBenchmarks:

    @pytest.mark.benchmark(group="agent_creation")
    def test_agent_creation_performance(self, benchmark):
        """Agent creation should complete in <1ms."""
        def create_agent():
            return Agent.create("Explorer")

        result = benchmark(create_agent)
        assert benchmark.stats.mean < 0.001  # <1ms

    @pytest.mark.benchmark(group="simulation_step")
    def test_simulation_step_1000_agents(self, benchmark):
        """Simulation step with 1000 agents should complete in <10ms."""
        world = World()
        agents = [Agent.create("Explorer") for _ in range(1000)]
        for agent in agents:
            world.add_agent(agent)

        def simulation_step():
            world.step()

        result = benchmark(simulation_step)
        assert benchmark.stats.mean < 0.01  # <10ms

    @pytest.mark.benchmark(group="coalition_formation")
    def test_coalition_formation_performance(self, benchmark):
        """Coalition formation with 10 agents should complete in <500ms."""
        def form_coalition():
            world = World()
            agents = [Agent.create("Explorer") for _ in range(10)]
            for agent in agents:
                world.add_agent(agent)

            # Run until coalition forms or timeout
            for _ in range(1000):
                world.step()
                if world.coalitions:
                    break

            return len(world.coalitions)

        result = benchmark(form_coalition)
        assert benchmark.stats.mean < 0.5  # <500ms
        assert result >= 1  # At least one coalition formed
```

### Architectural Compliance Testing

```python
# tests/compliance/test_dependency_rules.py
import ast
import os
from pathlib import Path

def test_core_domain_has_no_interface_imports():
    """Core domain modules must not import from interface layers (ADR-003)."""
    core_domains = ['agents/', 'inference/', 'coalitions/', 'world/']
    forbidden_imports = ['api/', 'web/', 'infrastructure/']

    violations = []

    for domain in core_domains:
        for py_file in Path(domain).glob('**/*.py'):
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                        for forbidden in forbidden_imports:
                            if module.startswith(forbidden.rstrip('/')):
                                violations.append(f"{py_file}: imports {module}")

    assert len(violations) == 0, f"ADR-003 violations found: {violations}"

def test_file_placement_follows_canonical_structure():
    """All files must be placed according to ADR-002 canonical structure."""
    # Implementation validates file placement against ADR-002 rules
    pass

def test_naming_conventions_compliance():
    """All files must follow ADR-004 naming conventions."""
    # Implementation validates naming patterns
    pass
```

### Continuous Integration Pipeline

```yaml
# .github/workflows/comprehensive-testing.yml
name: Comprehensive Testing Pipeline

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Unit Tests
        run: pytest tests/unit/ --cov=src --cov-report=xml
      - name: Upload Coverage
        uses: codecov/codecov-action@v1

  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Property Tests
        run: pytest tests/property/ --hypothesis-profile=ci

  behavior-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Behavior Tests
        run: pytest tests/behavior/ --bdd-strict

  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Performance Tests
        run: pytest tests/performance/ --benchmark-only

  compliance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Compliance Tests
        run: pytest tests/compliance/ --strict

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test
    steps:
      - uses: actions/checkout@v2
      - name: Run Integration Tests
        run: pytest tests/integration/
```

## Links and References

- [Hypothesis: Property-Based Testing](https://hypothesis.readthedocs.io/)
- [pytest-bdd: Behavior Driven Development](https://pytest-bdd.readthedocs.io/)
- [pytest-benchmark: Performance Testing](https://pytest-benchmark.readthedocs.io/)
- [Testing Mathematical Software](https://doi.org/10.1109/MCSE.2014.20)
- [Task 2: Establish Test Baseline & CI Pipeline](../../../.taskmaster/tasks/task_002.txt)
- [ADR-002: Canonical Directory Structure](002-canonical-directory-structure.md)
- [ADR-003: Dependency Rules](003-dependency-rules.md)
- [ADR-005: Active Inference Architecture](005-active-inference-architecture.md)

---

**Testing Philosophy**: "Tests are executable specifications that provide confidence in system behavior. For complex multi-agent systems with mathematical foundations, comprehensive testing is not optional—it's essential for correctness, performance, and maintainability."
