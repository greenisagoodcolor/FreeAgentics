# Test Data Management and Fixtures System

This directory contains a comprehensive test data management system for FreeAgentics, providing type-safe factories, builders, and fixtures for all domain objects.

## Overview

The test data management system consists of several key components:

- **Schemas** (`schemas.py`): Pydantic models for validating test data
- **Builders** (`builders.py`): Builder pattern implementations for flexible object construction
- **Factories** (`factories.py`): High-level factories for creating test data with database persistence
- **Fixtures** (`fixtures.py`): Reusable pytest fixtures for common test scenarios
- **Generators** (`generators.py`): Large-scale data generators for performance testing

## Quick Start

### Using Builders

```python
from tests.fixtures import AgentBuilder

# Create a custom agent
agent = (AgentBuilder()
         .with_name("TestAgent")
         .as_resource_collector()
         .with_position(10.0, 20.0)
         .active()
         .build())
```

### Using Factories

```python
from tests.fixtures import AgentFactory, CoalitionFactory

# Create agent with database persistence
agent = AgentFactory.create(
    session=db_session,
    name="TestAgent",
    template="explorer"
)

# Create coalition with agents
coalition, agents = CoalitionFactory.create_with_agents(
    session=db_session,
    num_agents=5,
    name="TestCoalition"
)
```

### Using Pytest Fixtures

```python
def test_agent_behavior(active_agent, coalition_with_agents):
    """Test using pre-configured fixtures."""
    assert active_agent.status == AgentStatus.ACTIVE
    assert len(coalition_with_agents.agents) > 0
```

### Using Generators for Performance Testing

```python
from tests.fixtures import generate_performance_dataset
from tests.fixtures.schemas import PerformanceTestConfigSchema

# Generate large dataset
config = PerformanceTestConfigSchema(
    num_agents=1000,
    num_coalitions=50,
    num_knowledge_nodes=5000
)
dataset = generate_performance_dataset(config)
```

## Schemas

All test data is validated using Pydantic schemas that match production models:

- `AgentSchema`: Complete agent data with beliefs, preferences, and metrics
- `CoalitionSchema`: Coalition data with objectives and member relationships
- `KnowledgeNodeSchema`: Knowledge graph nodes with versioning
- `KnowledgeEdgeSchema`: Relationships between knowledge nodes

### Schema Features

- **Type Safety**: All fields are strongly typed
- **Validation**: Automatic validation of constraints and relationships
- **Defaults**: Sensible defaults for all optional fields
- **Enums**: Type-safe enumerations for status values

## Builders

Builders provide a fluent interface for constructing test objects:

### AgentBuilder

```python
agent = (AgentBuilder()
         .with_name("Explorer001")
         .as_explorer()  # Pre-configured explorer template
         .with_grid_world_config(grid_size=10)
         .with_uniform_beliefs(num_states=5)
         .with_random_metrics()
         .active()
         .build())
```

### CoalitionBuilder

```python
coalition = (CoalitionBuilder()
             .with_name("Strategic Alliance")
             .as_resource_coalition()
             .with_resource_optimization_objective()
             .with_required_capabilities("planning", "coordination")
             .with_random_scores()
             .active()
             .build())
```

### KnowledgeNodeBuilder

```python
node = (KnowledgeNodeBuilder()
        .as_concept("Navigation Strategy")
        .with_confidence(0.85)
        .with_embedding(dim=256)
        .with_creator_agent(agent.id)
        .build())
```

## Factories

Factories provide high-level methods for creating complex test scenarios:

### AgentFactory

- `create()`: Create single agent with optional database persistence
- `create_batch()`: Create multiple agents efficiently
- `create_with_history()`: Create agent with realistic activity history

### CoalitionFactory

- `create()`: Create coalition with optional member agents
- `create_with_agents()`: Create coalition and agents together
- `create_coalition_network()`: Create network of interconnected coalitions

### KnowledgeGraphFactory

- `create_node()`: Create single knowledge node
- `create_edge()`: Create relationship between nodes
- `create_knowledge_graph()`: Create complete connected graph
- `create_agent_knowledge_scenario()`: Create knowledge for specific agent

## Fixtures

The system provides comprehensive pytest fixtures:

### Database Fixtures

- `test_engine`: In-memory SQLite engine for testing
- `db_session`: Database session with automatic rollback
- `clean_database`: Ensures clean database state

### Agent Fixtures

- `agent_fixture`: Single test agent
- `active_agent`: Fully configured active agent
- `agent_batch`: Batch of 10 test agents
- `resource_collector_agent`: Pre-configured resource collector
- `explorer_agent`: Pre-configured explorer
- `coordinator_agent`: Pre-configured coordinator

### Coalition Fixtures

- `coalition_fixture`: Single test coalition
- `coalition_with_agents`: Coalition with 5 member agents
- `resource_coalition`: Resource-focused coalition
- `exploration_coalition`: Exploration-focused coalition
- `coalition_network`: Network of 3 interconnected coalitions

### Knowledge Graph Fixtures

- `knowledge_node_fixture`: Single test node
- `knowledge_graph_fixture`: Small test graph (10 nodes)
- `large_knowledge_graph`: Large graph for performance testing (100 nodes)
- `agent_knowledge_scenario`: Knowledge created by specific agent

### Complex Scenario Fixtures

- `multi_agent_scenario`: Complete scenario with diverse agents and coalitions
- `performance_test_scenario`: Performance test setup with 50 agents
- `stress_test_data`: Stress test data with 5x normal volume

### Parameterized Fixtures

```python
@pytest.mark.parametrize("agent_by_template", ["grid_world", "explorer"], indirect=True)
def test_agent_templates(agent_by_template):
    """Test runs for each template type."""
    assert agent_by_template.template in ["grid_world", "explorer"]
```

## Generators

For performance and scale testing, the system provides data generators:

### AgentGenerator

```python
generator = AgentGenerator()

# Generate diverse population
agents = generator.generate_diverse_population(
    total_count=1000,
    distribution={
        'resource_collector': 0.4,
        'explorer': 0.3,
        'coordinator': 0.3
    }
)

# Generate spatial clusters
agents = generator.generate_spatial_clusters(
    total_count=500,
    num_clusters=5,
    cluster_std=10.0
)
```

### KnowledgeGraphGenerator

```python
generator = KnowledgeGraphGenerator()

# Generate scale-free graph
graph = generator.generate_scale_free_graph(
    num_nodes=1000,
    initial_nodes=5,
    edges_per_new_node=3
)

# Generate connected graph
graph = generator.generate_connected_graph(
    num_nodes=500,
    connectivity=0.1,
    ensure_connected=True
)
```

### PerformanceDataGenerator

```python
from tests.fixtures.schemas import PerformanceTestConfigSchema

config = PerformanceTestConfigSchema(
    num_agents=10000,
    num_coalitions=100,
    num_knowledge_nodes=50000,
    batch_size=1000
)

# Generate to database
generator = PerformanceDataGenerator()
results = generator.generate_to_database(session, config)

# Export to file
generator.export_to_file(dataset, "test_data.json", format="json")
```

## Best Practices

### 1. Use Builders for Custom Objects

```python
# Good: Clear, type-safe construction
agent = (AgentBuilder()
         .with_name("CustomAgent")
         .with_specific_config()
         .build())

# Avoid: Direct dictionary manipulation
agent_dict = {"name": "CustomAgent", ...}  # Error-prone
```

### 2. Use Factories for Database Objects

```python
# Good: Handles persistence correctly
agent = AgentFactory.create(session, name="TestAgent")

# Avoid: Manual ORM object creation
agent = Agent(name="TestAgent")  # Missing required fields
```

### 3. Use Fixtures for Common Scenarios

```python
def test_coalition_formation(coalition_with_agents):
    """Use pre-configured fixtures."""
    assert len(coalition_with_agents.agents) == 5
```

### 4. Validate Test Data

```python
# Schema validation happens automatically
agent = AgentBuilder().with_name("").build()  # ValidationError: name too short
```

### 5. Use Generators for Scale Testing

```python
# Generate realistic large datasets
dataset = generate_performance_dataset(
    PerformanceTestConfigSchema(num_agents=10000)
)
```

## Examples

### Example 1: Unit Test with Custom Agent

```python
def test_agent_inference():
    # Build custom agent
    agent = (AgentBuilder()
             .with_name("InferenceTestAgent")
             .with_grid_world_config(grid_size=3)
             .with_uniform_beliefs(num_states=9)
             .active()
             .build())

    # Test inference logic
    assert agent.beliefs.state_beliefs is not None
    assert len(agent.beliefs.state_beliefs) == 9
```

### Example 2: Integration Test with Database

```python
def test_coalition_persistence(db_session):
    # Create coalition with agents
    coalition, agents = CoalitionFactory.create_with_agents(
        session=db_session,
        num_agents=3,
        name="PersistenceTest"
    )

    # Verify persistence
    assert db_session.query(Coalition).count() == 1
    assert db_session.query(Agent).count() == 3
```

### Example 3: Performance Test

```python
def test_large_scale_processing(db_session):
    # Generate performance dataset
    config = PerformanceTestConfigSchema(
        num_agents=1000,
        num_coalitions=50
    )

    generator = PerformanceDataGenerator()
    results = generator.generate_to_database(db_session, config)

    # Verify performance
    assert results['timing']['agent_creation'] < 10.0  # seconds
    assert results['counts']['agents'] == 1000
```

## Extending the System

### Adding New Domain Objects

1. Create schema in `schemas.py`
2. Create builder in `builders.py`
3. Create factory in `factories.py`
4. Add fixtures in `fixtures.py`

### Adding New Generators

1. Extend `DataGenerator` base class
2. Implement generation logic
3. Add convenience functions

### Custom Fixtures

```python
@pytest.fixture
def specialized_scenario(db_session, agent_batch):
    """Create custom test scenario."""
    # Custom setup logic
    return custom_data
```

## Performance Considerations

- **Batch Operations**: Use `create_batch()` for multiple objects
- **Streaming**: Use generators for very large datasets
- **Database Commits**: Control transaction boundaries
- **Memory Usage**: Stream data instead of loading all at once

## Troubleshooting

### Common Issues

1. **Validation Errors**: Check schema constraints
2. **Database Errors**: Ensure proper session management
3. **Performance Issues**: Use batch operations and streaming

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```
