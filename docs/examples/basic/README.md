# CogniticNet Code Examples

This directory contains comprehensive code examples demonstrating various aspects of the CogniticNet multi-agent system. These examples complement the documentation guides and provide practical implementation patterns for developers.

## 📁 Example Files

### 1. [agent-creation-examples.py](./agent-creation-examples.py)

**Purpose**: Demonstrates agent creation patterns and configurations

**Examples Included**:

- Basic agent creation with standard configurations
- Advanced agent configurations with custom parameters
- Coalition formation and multi-agent coordination
- Communication patterns between agents
- Error handling and validation techniques
- Performance optimization strategies

**Key Functions**:

- `create_basic_explorer()` - Simple explorer agent creation
- `create_advanced_specialist()` - Complex agent with custom settings
- `create_coalition_example()` - Multi-agent coalition formation
- `demonstrate_communication()` - Agent messaging patterns
- `error_handling_example()` - Validation and error recovery
- `performance_example()` - Batch creation and caching

**Run Example**:

```bash
cd docs/examples
python agent-creation-examples.py
```

### 2. [coalition-formation-examples.py](./coalition-formation-examples.py)

**Purpose**: Comprehensive coalition formation patterns and strategies

**Examples Included**:

- Exploration coalition formation
- Emergency response team assembly
- Research and development teams
- Dynamic coalition adaptation
- Compatibility assessment algorithms
- Performance-based team optimization

**Key Functions**:

- `example_1_exploration_coalition()` - Systematic exploration team
- `example_2_emergency_response_coalition()` - Crisis response formation
- `example_3_research_coalition()` - Expertise-based research teams
- `example_4_dynamic_coalition_adaptation()` - Adaptive team evolution

**Run Example**:

```bash
cd docs/examples
python coalition-formation-examples.py
```

## 🚀 Getting Started

### Prerequisites

Before running the examples, ensure you have the required dependencies:

```bash
# Install core dependencies
pip install httpx pydantic

# Optional: For enhanced features
pip install numpy scipy torch
```

### Running Examples

Each example file can be run independently:

```bash
# Navigate to examples directory
cd docs/examples

# Run agent creation examples
python agent-creation-examples.py

# Run coalition formation examples
python coalition-formation-examples.py
```

### Understanding the Output

The examples provide detailed console output showing:

- **Configuration Details**: Agent parameters and settings
- **Process Flow**: Step-by-step execution information
- **Performance Metrics**: Timing and efficiency statistics
- **Validation Results**: Error checking and validation outcomes
- **Collaboration Patterns**: Coalition formation and coordination

## 📚 Example Categories

### Agent Creation Patterns

#### Basic Agent Creation

```python
# Simple explorer agent
config = {
    "name": "Explorer-001",
    "agent_type": "explorer",
    "personality": {
        "exploration_tendency": 0.9,
        "cooperation_level": 0.6
    },
    "capabilities": ["move", "observe", "map"]
}
```

#### Advanced Configuration

```python
# Specialist agent with complex setup
config = {
    "name": "Advanced-Specialist",
    "agent_type": "specialist",
    "coalition_preferences": {...},
    "behavior_weights": {...},
    "learning_parameters": {...},
    "security_clearance": "high"
}
```

### Coalition Formation Strategies

#### Compatibility-Based Formation

- Personality trait matching
- Capability complementarity
- Communication style alignment
- Leadership preference compatibility

#### Task-Oriented Assembly

- Required capability mapping
- Performance history analysis
- Availability and resource assessment
- Geographic proximity optimization

#### Dynamic Adaptation

- Performance monitoring
- Capability gap identification
- Member addition/removal
- Role reassignment

### Communication Patterns

#### Direct Messaging

```python
message = {
    "sender_id": "agent_001",
    "recipient_ids": ["agent_002"],
    "message_type": "discovery_report",
    "content": {"resource_discovered": True}
}
```

#### Broadcast Communication

```python
alert = {
    "sender_id": "guardian_001",
    "recipient_ids": ["all_agents_in_area"],
    "message_type": "security_alert",
    "priority": "urgent"
}
```

#### Coordination Protocols

```python
coordination = {
    "sender_id": "coordinator_001",
    "message_type": "mission_assignment",
    "content": {
        "mission_id": "alpha_001",
        "assignments": {...}
    }
}
```

## 🛠️ Development Patterns

### Error Handling Best Practices

The examples demonstrate robust error handling:

```python
def validate_agent_config(config):
    """Comprehensive validation with detailed error reporting"""
    errors = []

    # Name validation
    if not config.get('name') or len(config['name']) < 3:
        errors.append("Agent name must be at least 3 characters")

    # Type validation
    valid_types = ["explorer", "monitor", "coordinator", ...]
    if config.get('agent_type') not in valid_types:
        errors.append(f"Invalid agent type: {config.get('agent_type')}")

    return errors
```

### Performance Optimization Techniques

#### Batch Operations

```python
async def create_agents_batch(configs, batch_size=5):
    """Create multiple agents efficiently in batches"""
    for i in range(0, len(configs), batch_size):
        batch = configs[i:i+batch_size]
        tasks = [create_agent(config) for config in batch]
        results = await asyncio.gather(*tasks)
        # Process batch results
```

#### Configuration Caching

```python
class ConfigCache:
    """Cache for agent configuration templates"""
    def get_template(self, agent_type):
        return self.cache.get(agent_type, {}).copy()

    def cache_template(self, agent_type, template):
        self.cache[agent_type] = template
```

### Monitoring and Metrics

The examples include comprehensive monitoring:

- **Performance Tracking**: Creation time, success rates, resource usage
- **Coalition Metrics**: Compatibility scores, strength calculations, adaptation rates
- **Communication Analysis**: Message flow, response times, protocol efficiency

## 🔧 Customization Guide

### Extending Examples

To add your own examples:

1. **Create New Example Function**:

```python
async def my_custom_example():
    """Your custom example description"""
    print("=== My Custom Example ===")
    # Implementation here
    return result
```

2. **Add to Main Function**:

```python
async def main():
    # Existing examples...
    await my_custom_example()
```

3. **Follow Naming Conventions**:

- Use descriptive function names
- Include comprehensive docstrings
- Add console output for clarity
- Return meaningful results

### Configuration Templates

Create reusable configuration templates:

```python
AGENT_TEMPLATES = {
    "explorer": {
        "personality": {"exploration_tendency": 0.9},
        "capabilities": ["move", "observe", "map"]
    },
    "guardian": {
        "personality": {"cooperation_level": 0.95},
        "capabilities": ["defend", "protect", "alert"]
    }
}
```

### Custom Coalition Algorithms

Implement specialized coalition formation:

```python
def calculate_custom_compatibility(agent1, agent2):
    """Your custom compatibility algorithm"""
    # Implement custom logic
    return compatibility_score
```

## 📖 Related Documentation

- **[Agent Creation Guide](../guides/agent-creation.md)** - Comprehensive agent creation documentation
- **[Agent Types and Behaviors](../guides/agent-types-behaviors-interactions.md)** - Agent type specifications
- **[Coalition Formation Guide](../guides/coalition-formation.md)** - Coalition mechanics documentation
- **[API Reference](../api/README.md)** - Complete API documentation

## 🤝 Contributing

To contribute additional examples:

1. Follow the existing code structure and documentation style
2. Include comprehensive error handling and validation
3. Provide clear console output and progress indicators
4. Add meaningful comments and docstrings
5. Test examples thoroughly before submission

## ⚠️ Important Notes

- **Demo Mode**: Examples include simulation modes for testing without live API connections
- **Error Handling**: All examples include robust error handling and validation
- **Performance**: Examples demonstrate both individual and batch operations
- **Scalability**: Patterns shown are designed for production use
- **Security**: Examples include security considerations and validation

## 🎯 Next Steps

After exploring these examples:

1. **Experiment**: Modify examples to understand different configurations
2. **Integrate**: Use patterns in your own CogniticNet applications
3. **Extend**: Build upon examples for specific use cases
4. **Optimize**: Apply performance patterns for production deployments
5. **Collaborate**: Share insights and improvements with the community

---

**Happy Coding!** 🤖✨

For questions or support, refer to the main documentation or community resources.
