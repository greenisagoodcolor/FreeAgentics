# Tutorial: Creating an Agent in FreeAgentics

This tutorial will guide you through the process of creating a custom agent in the FreeAgentics system. By the end, you'll understand how to define agent personalities, behaviors, and integrate them with the Active Inference engine.

## Prerequisites

Before starting this tutorial, make sure you have:

- FreeAgentics installed and running
- Basic understanding of Python programming
- Familiarity with Active Inference concepts (see [Active Inference Guide](../active-inference-guide.md))
- Access to the FreeAgentics API

## Step 1: Define Agent Personality

Every agent in FreeAgentics has a personality that influences its behavior and decision-making. Let's start by defining a personality for our agent.

```python
from agents.base.data_model import Personality

# Create a personality with specific traits
explorer_personality = Personality(
    openness=0.8,        # High curiosity and openness to new experiences
    conscientiousness=0.6,  # Moderate organization and reliability
    extraversion=0.7,    # Fairly social and energetic
    agreeableness=0.5,   # Moderate cooperation and consideration
    neuroticism=0.3      # Low anxiety and emotional instability
)
```

The five personality traits follow the standard Five-Factor Model (Big Five):

- **Openness**: Curiosity, creativity, and openness to new experiences
- **Conscientiousness**: Organization, reliability, and goal-directed behavior
- **Extraversion**: Sociability, energy, and assertiveness
- **Agreeableness**: Cooperation, consideration, and empathy
- **Neuroticism**: Anxiety, emotional instability, and negative emotions

## Step 2: Create an Agent Using AgentFactory

The `AgentFactory` provides a convenient way to create agents with specific configurations.

```python
from agents.base.agent_factory import AgentFactory
from agents.base.data_model import AgentType, Position

# Initialize the agent factory
agent_factory = AgentFactory()

# Create an explorer agent
my_agent = agent_factory.create_agent(
    name="CuriousExplorer",
    agent_type=AgentType.EXPLORER,
    personality=explorer_personality,
    position=Position(x=10, y=15),  # Starting position in the world
    initial_resources={"energy": 100, "materials": 50}
)
```

## Step 3: Define Custom Behaviors

You can customize your agent's behavior by extending the base behavior classes.

```python
from agents.explorer.explorer_behavior import ExplorerBehavior
from agents.base.data_model import Action, Observation
from typing import List

class MyCustomExplorerBehavior(ExplorerBehavior):
    def select_exploration_target(self, current_position: Position, observations: List[Observation]) -> Position:
        """
        Custom logic to select the next exploration target.
        """
        # Example: Prioritize unexplored areas with potential resources
        potential_targets = self._identify_potential_targets(observations)
        if potential_targets:
            return self._select_optimal_target(potential_targets, current_position)
        else:
            # Default behavior: move in a random direction
            return self._random_nearby_position(current_position)

    def _identify_potential_targets(self, observations: List[Observation]) -> List[Position]:
        """
        Identify positions that might be worth exploring.
        """
        potential_targets = []
        for obs in observations:
            if obs.resource_probability > 0.3 and not obs.is_explored:
                potential_targets.append(obs.position)
        return potential_targets

    def _select_optimal_target(self, targets: List[Position], current_position: Position) -> Position:
        """
        Select the optimal target based on distance and potential value.
        """
        # This is a simplified example - you would typically implement
        # more sophisticated logic here
        return min(targets, key=lambda pos: pos.distance_to(current_position))

    def _random_nearby_position(self, current_position: Position) -> Position:
        """
        Select a random position near the current one.
        """
        import random
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        return Position(x=current_position.x + dx, y=current_position.y + dy)
```

## Step 4: Integrate with Active Inference

To make your agent truly intelligent, you need to integrate it with the Active Inference engine using PyMDP's validated algorithms.

```python
from inference.engine.generative_model import DiscreteGenerativeModel, ModelDimensions, ModelParameters
from inference.engine.pymdp_generative_model import PyMDPGenerativeModel
from inference.engine.pymdp_policy_selector import PyMDPPolicySelector
from inference.engine import PolicyConfig

# Create model dimensions
dims = ModelDimensions(
    num_states=100,      # Location and resource state combinations
    num_observations=25,  # Visual and resource detection observations
    num_actions=7        # Movement and interaction actions
)

# Create a discrete generative model for the agent
params = ModelParameters(use_gpu=torch.cuda.is_available())
generative_model = DiscreteGenerativeModel(dims, params)

# Convert to PyMDP format for validated calculations
pymdp_model = PyMDPGenerativeModel.from_discrete_model(generative_model)

# Initialize PyMDP-based policy selector
policy_config = PolicyConfig(precision=1.0, planning_horizon=3)
policy_selector = PyMDPPolicySelector(policy_config, pymdp_model)

# Connect the agent to the PyMDP-based inference system
my_agent.connect_inference_system(pymdp_model, policy_selector)
```

## Step 5: Define Initial Beliefs

Initial beliefs shape how your agent perceives and interacts with the world.

```python
from inference.engine.belief_state import BeliefState

# Create initial beliefs
initial_beliefs = BeliefState()

# Add prior beliefs about the environment
initial_beliefs.add_prior("resource_distribution", {
    "energy": 0.4,    # 40% probability of finding energy resources
    "materials": 0.3, # 30% probability of finding material resources
    "knowledge": 0.2  # 20% probability of finding knowledge resources
})

# Add prior beliefs about other agents
initial_beliefs.add_prior("agent_cooperation", {
    "explorer": 0.7,  # Explorers are likely to cooperate
    "merchant": 0.8,  # Merchants are very likely to cooperate
    "scholar": 0.9,   # Scholars are extremely likely to cooperate
    "guardian": 0.5   # Guardians are moderately likely to cooperate
})

# Set the agent's initial beliefs
my_agent.set_beliefs(initial_beliefs)
```

## Step 6: Register the Agent with the World

To allow your agent to interact with the environment, you need to register it with the world simulation.

```python
from world.h3_world import H3World

# Get the world instance
world = H3World.get_instance()

# Register the agent with the world
world.register_agent(my_agent)
```

## Step 7: Define Communication Capabilities

Agents can communicate with each other to share information and coordinate actions.

```python
from communication.message_system import MessageSystem
from agents.base.communication import CommunicationCapability

# Get the message system instance
message_system = MessageSystem.get_instance()

# Create communication capability for the agent
comm_capability = CommunicationCapability(
    message_system=message_system,
    agent_id=my_agent.id,
    communication_range=5,  # How far the agent can communicate
    bandwidth=10            # How many messages the agent can send/receive per cycle
)

# Add communication capability to the agent
my_agent.add_capability(comm_capability)
```

## Step 8: Run the Agent in Simulation

Finally, let's run the agent in the simulation to see how it behaves.

```python
from world.simulation.engine import SimulationEngine

# Get the simulation engine instance
sim_engine = SimulationEngine.get_instance()

# Add the agent to the simulation
sim_engine.add_agent(my_agent)

# Run the simulation for 100 steps
for _ in range(100):
    sim_engine.step()

    # You can observe the agent's state after each step
    print(f"Agent position: {my_agent.position}")
    print(f"Agent resources: {my_agent.resources}")
    print(f"Agent beliefs: {my_agent.get_beliefs().top_beliefs(5)}")
    print("---")
```

## Step 9: Analyze Agent Performance

After running the simulation, you can analyze how your agent performed.

```python
# Get agent statistics
stats = my_agent.get_statistics()

print(f"Distance traveled: {stats.distance_traveled}")
print(f"Resources gathered: {stats.resources_gathered}")
print(f"Interactions with other agents: {stats.agent_interactions}")
print(f"Knowledge gained: {stats.knowledge_gained}")

# Visualize the agent's path
from world.visualization import WorldVisualizer
visualizer = WorldVisualizer()
visualizer.plot_agent_path(my_agent.id)
visualizer.show()
```

## Step 10: Save and Load Agents

You can save your agent's state and load it later to continue the simulation.

```python
import json

# Save agent state
agent_state = my_agent.serialize()
with open("my_agent_state.json", "w") as f:
    json.dump(agent_state, f)

# Load agent state later
with open("my_agent_state.json", "r") as f:
    saved_state = json.load(f)

# Create a new agent from the saved state
restored_agent = agent_factory.load_agent(saved_state)
```

## Complete Example

Here's a complete example that puts everything together:

```python
from agents.base.agent_factory import AgentFactory
from agents.base.data_model import AgentType, Personality, Position
from agents.explorer.explorer_behavior import ExplorerBehavior
from inference.engine.generative_model import DiscreteGenerativeModel, ModelDimensions, ModelParameters
from inference.engine.pymdp_generative_model import PyMDPGenerativeModel
from inference.engine.pymdp_policy_selector import PyMDPPolicySelector
from inference.engine import PolicyConfig
from inference.engine.belief_state import BeliefState
from communication.message_system import MessageSystem
from agents.base.communication import CommunicationCapability
from world.h3_world import H3World
from world.simulation.engine import SimulationEngine

# Step 1: Define agent personality
personality = Personality(
    openness=0.8,
    conscientiousness=0.6,
    extraversion=0.7,
    agreeableness=0.5,
    neuroticism=0.3
)

# Step 2: Create the agent
agent_factory = AgentFactory()
my_agent = agent_factory.create_agent(
    name="CuriousExplorer",
    agent_type=AgentType.EXPLORER,
    personality=personality,
    position=Position(x=10, y=15),
    initial_resources={"energy": 100, "materials": 50}
)

# Step 3: Define custom behaviors (optional)
# my_agent.set_behavior(MyCustomExplorerBehavior())

# Step 4: Integrate with Active Inference
dims = ModelDimensions(
    num_states=100,
    num_observations=25,
    num_actions=7
)
params = ModelParameters(use_gpu=torch.cuda.is_available())
generative_model = DiscreteGenerativeModel(dims, params)
pymdp_model = PyMDPGenerativeModel.from_discrete_model(generative_model)
policy_config = PolicyConfig(precision=1.0, planning_horizon=3)
policy_selector = PyMDPPolicySelector(policy_config, pymdp_model)
my_agent.connect_inference_system(pymdp_model, policy_selector)

# Step 5: Define initial beliefs
initial_beliefs = BeliefState()
initial_beliefs.add_prior("resource_distribution", {
    "energy": 0.4,
    "materials": 0.3,
    "knowledge": 0.2
})
initial_beliefs.add_prior("agent_cooperation", {
    "explorer": 0.7,
    "merchant": 0.8,
    "scholar": 0.9,
    "guardian": 0.5
})
my_agent.set_beliefs(initial_beliefs)

# Step 6: Register with the world
world = H3World.get_instance()
world.register_agent(my_agent)

# Step 7: Define communication capabilities
message_system = MessageSystem.get_instance()
comm_capability = CommunicationCapability(
    message_system=message_system,
    agent_id=my_agent.id,
    communication_range=5,
    bandwidth=10
)
my_agent.add_capability(comm_capability)

# Step 8: Run the simulation
sim_engine = SimulationEngine.get_instance()
sim_engine.add_agent(my_agent)

print("Starting simulation...")
for i in range(100):
    sim_engine.step()
    if i % 10 == 0:  # Print status every 10 steps
        print(f"Step {i}: Agent at {my_agent.position}, Energy: {my_agent.resources['energy']}")

# Step 9: Analyze performance
stats = my_agent.get_statistics()
print("\nSimulation complete!")
print(f"Distance traveled: {stats.distance_traveled}")
print(f"Resources gathered: {stats.resources_gathered}")
print(f"Interactions: {stats.agent_interactions}")
```

## Next Steps

Now that you've created your first agent, you might want to:

1. [Explore coalition formation](coalition-formation.md) to see how agents can work together
2. [Create custom GNN models](creating-gnn-models.md) to define more sophisticated agent behaviors
3. [Deploy agents to edge devices](edge-deployment.md) for real-world applications

## Troubleshooting

### Common Issues

1. **Agent doesn't move or interact**: Check that the agent is properly registered with the world and simulation engine.
2. **Beliefs not updating**: Verify that the Active Inference engine is correctly connected.
3. **No communication between agents**: Ensure the message system is initialized and the communication range is appropriate.

### Getting Help

If you encounter issues not covered here, check the [API Reference](../api/index.md) or ask for help in the [FreeAgentics community forum](https://community.freeagentics.ai).
