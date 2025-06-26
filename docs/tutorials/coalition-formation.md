# Tutorial: Forming Agent Coalitions in FreeAgentics

This tutorial will guide you through the process of forming coalitions between agents in the FreeAgentics system. Coalitions allow agents to collaborate, share resources, and achieve goals that would be difficult or impossible for individual agents.

## Prerequisites

Before starting this tutorial, make sure you have:

- Completed the [Creating an Agent tutorial](creating-an-agent.md)
- Basic understanding of Python programming
- Familiarity with Active Inference concepts
- Multiple agents already created in your simulation

## Step 1: Import Required Modules

First, let's import the necessary modules for coalition formation.

```python
from coalitions.formation.coalition_builder import CoalitionBuilder
from coalitions.formation.coalition_formation_algorithms import PreferenceMatchingAlgorithm
from coalitions.coalition.coalition_criteria import CoalitionCriteria
from coalitions.contracts.coalition_contract import CoalitionContract
```

## Step 2: Define Coalition Criteria

Coalition criteria define the rules and conditions for forming a coalition. These criteria help determine which agents are compatible and how they should collaborate.

```python
# Create coalition criteria
criteria = CoalitionCriteria(
    min_agents=2,                # Minimum number of agents required
    max_agents=5,                # Maximum number of agents allowed
    required_capabilities=["movement", "communication"],  # Capabilities all agents must have
    min_resource_contribution={"energy": 20},  # Minimum resources each agent must contribute
    min_trust_level=0.6,         # Minimum trust level between agents
    max_distance=10,             # Maximum distance between agents
    coalition_purpose="resource_gathering"  # Purpose of the coalition
)
```

## Step 3: Create a Coalition Builder

The `CoalitionBuilder` is responsible for finding compatible agents and forming coalitions based on the criteria.

```python
# Create a coalition builder with a preference matching algorithm
coalition_builder = CoalitionBuilder(
    algorithm=PreferenceMatchingAlgorithm(),
    criteria=criteria
)
```

## Step 4: Find Compatible Agents

Now, let's find agents that are compatible for forming a coalition.

```python
from world.h3_world import H3World

# Get the world instance
world = H3World.get_instance()

# Get all agents in the world
all_agents = world.get_all_agents()

# Find compatible agents based on the criteria
compatible_agents = coalition_builder.find_compatible_agents(all_agents)

print(f"Found {len(compatible_agents)} compatible agents for coalition formation")
```

## Step 5: Calculate Coalition Value

Before forming a coalition, it's important to calculate its potential value to ensure it's beneficial for all participating agents.

```python
# Select agents for the coalition (for example, the first 3 compatible agents)
selected_agents = compatible_agents[:3]

# Calculate the coalition value
coalition_value = coalition_builder.calculate_coalition_value(selected_agents)

print(f"Potential coalition value: {coalition_value}")
print(f"Individual agent benefits:")
for agent in selected_agents:
    benefit = coalition_builder.calculate_agent_benefit(agent, selected_agents)
    print(f"  - {agent.name}: {benefit}")
```

## Step 6: Create a Coalition Contract

A coalition contract formalizes the agreement between agents, specifying their roles, responsibilities, and benefits.

```python
# Create a coalition contract
contract = CoalitionContract(
    agents=selected_agents,
    purpose="resource_gathering",
    duration=1000,  # Duration in simulation steps
    resource_sharing={
        "energy": "proportional",      # Share energy proportionally to contribution
        "materials": "equal",          # Share materials equally
        "knowledge": "contribution"    # Share knowledge based on contribution
    },
    decision_making="consensus",       # Decisions require consensus among agents
    termination_conditions={
        "goal_achieved": True,         # Terminate when the goal is achieved
        "resource_depletion": True     # Terminate if resources are depleted
    }
)
```

## Step 7: Form the Coalition

Now, let's form the coalition using the builder and the contract.

```python
# Form the coalition
coalition = coalition_builder.form_coalition(
    agents=selected_agents,
    contract=contract,
    name="ResourceGatherers"
)

print(f"Coalition '{coalition.name}' formed with {len(coalition.agents)} agents")
```

## Step 8: Register the Coalition

Register the coalition with the world to make it active in the simulation.

```python
from coalitions.formation.coalition_registry import CoalitionRegistry

# Get the coalition registry
registry = CoalitionRegistry.get_instance()

# Register the coalition
registry.register_coalition(coalition)

print(f"Coalition '{coalition.name}' registered with ID: {coalition.id}")
```

## Step 9: Coalition Operations

Once a coalition is formed, agents can perform collaborative operations.

```python
# Example: Collaborative resource gathering
target_location = Position(x=20, y=30)
coalition.coordinate_movement(target_location)

# Example: Share knowledge among coalition members
coalition.share_knowledge()

# Example: Distribute resources based on contract
coalition.distribute_resources()
```

## Step 10: Monitor Coalition Performance

You can monitor the performance and status of the coalition during simulation.

```python
# Run the simulation for 100 steps
for i in range(100):
    sim_engine.step()

    # Every 10 steps, check coalition status
    if i % 10 == 0:
        print(f"Step {i}: Coalition '{coalition.name}' status:")
        print(f"  - Total resources: {coalition.get_total_resources()}")
        print(f"  - Cohesion level: {coalition.get_cohesion_level()}")
        print(f"  - Goal progress: {coalition.get_goal_progress()}")
```

## Step 11: Coalition Dissolution

Coalitions can be dissolved when they are no longer beneficial or when their goals are achieved.

```python
# Check if the coalition should be dissolved
if coalition.should_dissolve():
    # Dissolve the coalition
    registry.dissolve_coalition(coalition.id)
    print(f"Coalition '{coalition.name}' dissolved")
```

## Complete Example

Here's a complete example that puts everything together:

```python
from agents.base.agent_factory import AgentFactory
from agents.base.data_model import AgentType, Personality, Position
from coalitions.formation.coalition_builder import CoalitionBuilder
from coalitions.formation.coalition_formation_algorithms import PreferenceMatchingAlgorithm
from coalitions.coalition.coalition_criteria import CoalitionCriteria
from coalitions.contracts.coalition_contract import CoalitionContract
from coalitions.formation.coalition_registry import CoalitionRegistry
from world.h3_world import H3World
from world.simulation.engine import SimulationEngine

# Create some agents
agent_factory = AgentFactory()

explorer = agent_factory.create_agent(
    name="Explorer1",
    agent_type=AgentType.EXPLORER,
    personality=Personality(openness=0.8, conscientiousness=0.6, extraversion=0.7, agreeableness=0.6, neuroticism=0.3),
    position=Position(x=10, y=15),
    initial_resources={"energy": 80, "materials": 30}
)

merchant = agent_factory.create_agent(
    name="Merchant1",
    agent_type=AgentType.MERCHANT,
    personality=Personality(openness=0.6, conscientiousness=0.8, extraversion=0.8, agreeableness=0.7, neuroticism=0.2),
    position=Position(x=12, y=14),
    initial_resources={"energy": 60, "materials": 100}
)

scholar = agent_factory.create_agent(
    name="Scholar1",
    agent_type=AgentType.SCHOLAR,
    personality=Personality(openness=0.9, conscientiousness=0.7, extraversion=0.5, agreeableness=0.8, neuroticism=0.3),
    position=Position(x=11, y=16),
    initial_resources={"energy": 50, "knowledge": 80}
)

# Get the world instance and register agents
world = H3World.get_instance()
world.register_agent(explorer)
world.register_agent(merchant)
world.register_agent(scholar)

# Define coalition criteria
criteria = CoalitionCriteria(
    min_agents=2,
    max_agents=5,
    required_capabilities=["movement", "communication"],
    min_resource_contribution={"energy": 20},
    min_trust_level=0.6,
    max_distance=10,
    coalition_purpose="resource_gathering"
)

# Create a coalition builder
coalition_builder = CoalitionBuilder(
    algorithm=PreferenceMatchingAlgorithm(),
    criteria=criteria
)

# Find compatible agents
all_agents = [explorer, merchant, scholar]
compatible_agents = coalition_builder.find_compatible_agents(all_agents)

print(f"Found {len(compatible_agents)} compatible agents for coalition formation")

# Create a coalition contract
contract = CoalitionContract(
    agents=compatible_agents,
    purpose="resource_gathering",
    duration=1000,
    resource_sharing={
        "energy": "proportional",
        "materials": "equal",
        "knowledge": "contribution"
    },
    decision_making="consensus",
    termination_conditions={
        "goal_achieved": True,
        "resource_depletion": True
    }
)

# Form the coalition
coalition = coalition_builder.form_coalition(
    agents=compatible_agents,
    contract=contract,
    name="ResourceGatherers"
)

print(f"Coalition '{coalition.name}' formed with {len(coalition.agents)} agents")

# Register the coalition
registry = CoalitionRegistry.get_instance()
registry.register_coalition(coalition)

print(f"Coalition '{coalition.name}' registered with ID: {coalition.id}")

# Get the simulation engine instance
sim_engine = SimulationEngine.get_instance()

# Run the simulation
print("Starting simulation...")
for i in range(100):
    sim_engine.step()

    # Every 10 steps, check coalition status
    if i % 10 == 0:
        print(f"Step {i}: Coalition '{coalition.name}' status:")
        print(f"  - Total resources: {coalition.get_total_resources()}")
        print(f"  - Cohesion level: {coalition.get_cohesion_level()}")
        print(f"  - Goal progress: {coalition.get_goal_progress()}")

# Check if the coalition should be dissolved
if coalition.should_dissolve():
    registry.dissolve_coalition(coalition.id)
    print(f"Coalition '{coalition.name}' dissolved")
else:
    print(f"Coalition '{coalition.name}' remains active")
```

## Next Steps

Now that you've learned how to form coalitions, you might want to:

1. [Deploy coalitions to edge devices](edge-deployment.md) for real-world applications
2. [Create custom coalition algorithms](custom-coalition-algorithms.md) for specialized coalition formation
3. [Implement coalition-based business models](coalition-business-models.md) for resource optimization

## Troubleshooting

### Common Issues

1. **Incompatible agents**: Ensure agents meet all criteria requirements (capabilities, resources, etc.)
2. **Coalition not forming**: Check that the trust levels between agents are sufficient
3. **Coalition dissolving too quickly**: Adjust the contract terms or termination conditions
4. **Resource distribution issues**: Verify the resource sharing rules in the contract

### Getting Help

If you encounter issues not covered here, check the [API Reference](../api/index.md) or ask for help in the [FreeAgentics community forum](https://community.freeagentics.ai).
