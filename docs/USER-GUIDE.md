# FreeAgentics User Guide

> **Complete guide to using FreeAgentics for multi-agent AI systems with Active Inference**

## Table of Contents

1. [Quick Start](#quick-start) - Get running in 5 minutes
2. [Core Concepts](#core-concepts) - Active Inference fundamentals
3. [Creating Your First Agent](#creating-your-first-agent) - Step-by-step walkthrough
4. [Agent Types & Templates](#agent-types--templates) - Explorer, Guardian, Merchant, Scholar
5. [Coalition Formation](#coalition-formation) - Multi-agent coordination
6. [Active Inference Deep Dive](#active-inference-deep-dive) - Mathematical foundations
7. [GNN Integration](#gnn-integration) - Natural language model specification
8. [World Simulation](#world-simulation) - Spatial environments and navigation
9. [Best Practices](#best-practices) - Production-ready patterns
10. [Common Patterns](#common-patterns) - Reusable solutions

---

## Quick Start

### Prerequisites

- Python 3.9+ with numpy, torch
- Node.js 18+ (for web interface)
- 8GB+ RAM (for multi-agent simulations)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/freeagentics.git
cd freeagentics

# Install Python dependencies
pip install -e .

# Install web dependencies
cd web && npm install

# Start the development environment
npm run dev
```

### Your First Agent in 2 Minutes

```python
from freeagentics import Agent

# Create an explorer agent with Active Inference
explorer = Agent.create("Explorer",
                       name="Alice",
                       personality={'curiosity': 0.8, 'caution': 0.3})

# The agent automatically implements Active Inference principles:
# - Maintains beliefs about the world state
# - Minimizes free energy through action selection
# - Updates beliefs based on observations

print(f"Agent beliefs: {explorer.beliefs}")
print(f"Next action: {explorer.select_action()}")
```

---

## Core Concepts

### Active Inference Framework

FreeAgentics implements **Active Inference** - a mathematical theory of brain function where agents:

1. **Maintain Beliefs** about hidden world states
2. **Minimize Free Energy** (surprise about observations)
3. **Plan Actions** to achieve preferred outcomes
4. **Update Beliefs** through Bayesian inference

### Mathematical Foundation

Every agent operates on four key matrices (from pymdp framework):

- **A Matrix**: Observation likelihood `P(observation|state)`
- **B Matrix**: Transition dynamics `P(next_state|state,action)`
- **C Matrix**: Preferences over observations
- **D Matrix**: Prior beliefs over initial states

### Agent Autonomy

Unlike scripted bots, FreeAgentics agents are truly autonomous:

- **No hardcoded behaviors** - all actions emerge from Active Inference
- **Adaptive** - learn and adjust to new environments
- **Mathematically principled** - based on peer-reviewed cognitive science

---

## Creating Your First Agent

### Step 1: Choose an Agent Template

FreeAgentics provides four base templates:

```python
from freeagentics import Agent

# Explorer: High curiosity, seeks novel information
explorer = Agent.create("Explorer", name="Scout")

# Guardian: Defensive, maintains system stability
guardian = Agent.create("Guardian", name="Protector")

# Merchant: Trade-focused, resource optimization
merchant = Agent.create("Merchant", name="Trader")

# Scholar: Knowledge-focused, information processing
scholar = Agent.create("Scholar", name="Researcher")
```

### Step 2: Customize Agent Parameters

```python
# Fine-tune Active Inference parameters
custom_explorer = Agent.create("Explorer",
    name="CustomScout",
    personality={
        'curiosity': 0.9,      # High exploration drive
        'caution': 0.2,        # Low risk aversion
        'social': 0.6          # Moderate social tendency
    },
    inference_params={
        'precision': 1.0,      # Confidence in beliefs
        'learning_rate': 0.1,  # Speed of belief updates
        'planning_horizon': 5   # Steps ahead to plan
    }
)
```

### Step 3: Add to Environment

```python
from freeagentics import World

# Create a simulated world
world = World(grid_size=20, resource_density=0.3)

# Add your agent
world.add_agent(custom_explorer)

# Run simulation
for step in range(100):
    world.step()  # Agents act according to Active Inference

    # Monitor agent beliefs and actions
    if step % 10 == 0:
        print(f"Step {step}: {custom_explorer.beliefs}")
```

---

## Agent Types & Templates

### Explorer Template

**Purpose**: Information gathering and environment mapping

**Active Inference Characteristics**:

- High precision on exploration actions
- Preferences for novel observations
- Low entropy beliefs (confident in what they've observed)

```python
explorer = Agent.create("Explorer",
    name="PathFinder",
    inference_params={
        'curiosity_weight': 0.8,  # Strong preference for information gain
        'entropy_threshold': 0.1   # Seek areas of high uncertainty
    }
)

# Explorers excel at:
# - Mapping unknown territories
# - Discovering resources
# - Providing information to other agents
```

### Guardian Template

**Purpose**: System protection and stability maintenance

**Active Inference Characteristics**:

- High precision on defensive actions
- Strong preferences for maintaining current state
- Conservative belief updating

```python
guardian = Agent.create("Guardian",
    name="Sentinel",
    inference_params={
        'stability_weight': 0.9,   # Strong preference for current state
        'threat_sensitivity': 0.8   # High alertness to changes
    }
)

# Guardians excel at:
# - Detecting anomalies
# - Preventing system degradation
# - Coordinating defensive responses
```

### Merchant Template

**Purpose**: Resource optimization and trade facilitation

**Active Inference Characteristics**:

- High precision on trade actions
- Preferences for resource accumulation
- Economic belief models

```python
merchant = Agent.create("Merchant",
    name="Trader",
    inference_params={
        'profit_weight': 0.7,      # Strong preference for resource gain
        'market_sensitivity': 0.6   # Responsiveness to price signals
    }
)

# Merchants excel at:
# - Resource allocation optimization
# - Market making and price discovery
# - Supply chain coordination
```

### Scholar Template

**Purpose**: Knowledge processing and research

**Active Inference Characteristics**:

- High precision on learning actions
- Preferences for information accuracy
- Deep belief hierarchies

```python
scholar = Agent.create("Scholar",
    name="Researcher",
    inference_params={
        'accuracy_weight': 0.9,    # Strong preference for correct beliefs
        'depth_preference': 0.8     # Preference for detailed understanding
    }
)

# Scholars excel at:
# - Pattern recognition and analysis
# - Knowledge synthesis
# - Research coordination
```

---

## Coalition Formation

### Understanding Coalitions

Coalitions emerge when agents discover mutual benefit through cooperation. In FreeAgentics, coalition formation follows Active Inference principles:

1. **Agents detect** potential collaboration opportunities
2. **Evaluate expected utility** of joining vs. staying independent
3. **Form coalitions** when expected free energy decreases
4. **Maintain coalitions** as long as benefits persist

### Basic Coalition Example

```python
from freeagentics import World, Coalition

# Create complementary agents
explorer = Agent.create("Explorer", name="Scout")
guardian = Agent.create("Guardian", name="Guard")
merchant = Agent.create("Merchant", name="Trader")

# Add to world
world = World(grid_size=20)
world.add_agents([explorer, guardian, merchant])

# Coalitions form automatically based on:
# - Complementary capabilities
# - Shared objectives
# - Proximity and communication opportunities

# Run simulation to observe coalition formation
for step in range(200):
    world.step()

    # Check for emergent coalitions
    if step % 50 == 0:
        coalitions = world.get_coalitions()
        print(f"Step {step}: {len(coalitions)} coalitions active")

        for coalition in coalitions:
            print(f"  Coalition: {[a.name for a in coalition.members]}")
            print(f"  Purpose: {coalition.objective}")
```

### Manual Coalition Creation

```python
# Create explicit coalition with defined objectives
coalition = Coalition(
    name="ExplorationTeam",
    objective="Map unknown sector Alpha-7",
    members=[explorer, guardian],
    coordination_strategy="hierarchical"  # or "democratic", "market"
)

# Define role assignments
coalition.assign_role(explorer, "primary_scout")
coalition.assign_role(guardian, "security_escort")

# Set coalition-level preferences
coalition.set_preferences({
    'safety_priority': 0.8,     # High safety preference
    'speed_priority': 0.6,      # Moderate speed preference
    'resource_efficiency': 0.7  # Good resource management
})

# Add to world
world.add_coalition(coalition)
```

### Coalition Strategies

#### Hierarchical Coordination

```python
# One agent leads, others follow
hierarchical_coalition = Coalition(
    coordination_strategy="hierarchical",
    leader=guardian,  # Guardian makes decisions
    decision_threshold=0.6  # 60% confidence required for action
)
```

#### Democratic Coordination

```python
# Decisions made by consensus
democratic_coalition = Coalition(
    coordination_strategy="democratic",
    voting_method="majority",  # or "consensus", "weighted"
    discussion_rounds=3  # Max debate cycles
)
```

#### Market Coordination

```python
# Internal resource markets
market_coalition = Coalition(
    coordination_strategy="market",
    currency="energy_tokens",
    auction_frequency=10  # Bidding every 10 steps
)
```

---

## Active Inference Deep Dive

### Belief State Management

Every FreeAgentics agent maintains probabilistic beliefs about the world:

```python
# Access agent's current beliefs
beliefs = agent.beliefs  # numpy array, sums to 1.0

# Beliefs represent probability distribution over world states
print(f"Agent believes state 0 with probability: {beliefs[0]:.3f}")
print(f"Agent believes state 1 with probability: {beliefs[1]:.3f}")
print(f"Entropy (uncertainty): {agent.entropy:.3f}")
```

### Free Energy Calculation

Active Inference agents minimize **free energy** - a measure of surprise:

```python
# Calculate current free energy
current_fe = agent.calculate_free_energy()

# Evaluate free energy for potential actions
action_free_energies = {}
for action in agent.get_available_actions():
    action_free_energies[action] = agent.calculate_expected_free_energy(action)

# Agent selects action with lowest expected free energy
best_action = min(action_free_energies, key=action_free_energies.get)
print(f"Selected action: {best_action}")
```

### Custom Observation Models

Define how agents perceive their environment:

```python
from freeagentics.inference import ObservationModel

class CustomVisionModel(ObservationModel):
    def __init__(self, visual_range=5, accuracy=0.9):
        self.visual_range = visual_range
        self.accuracy = accuracy

    def observe(self, agent_state, world_state):
        """Convert world state to agent observation."""
        nearby_entities = world_state.get_entities_in_range(
            agent_state.position, self.visual_range
        )

        # Add noise based on accuracy
        noisy_observations = self.add_noise(nearby_entities, self.accuracy)
        return noisy_observations

# Use custom observation model
agent = Agent.create("Explorer",
    observation_model=CustomVisionModel(visual_range=7, accuracy=0.95)
)
```

### Advanced Inference Parameters

```python
# Fine-tune Active Inference behavior
advanced_agent = Agent.create("Scholar",
    inference_params={
        # Belief updating
        'learning_rate': 0.15,          # Speed of belief updates (0.0-1.0)
        'prior_strength': 2.0,          # Strength of initial beliefs

        # Action selection
        'precision': 1.5,               # Confidence in action selection
        'temperature': 0.1,             # Randomness in action selection

        # Planning
        'planning_horizon': 8,          # Steps ahead to plan
        'discount_factor': 0.95,        # Future reward discounting

        # Exploration
        'curiosity_weight': 0.3,        # Information-seeking behavior
        'epistemic_precision': 0.8      # Confidence in exploration
    }
)
```

---

## GNN Integration

### Natural Language Agent Specification

Use the Generalized Model Notation (GMN) to create agents from natural language:

```python
from freeagentics.gnn import create_agent_from_description

# Natural language description
description = """
Create an agent that:
- Explores new areas when energy is high
- Returns to base when energy is low
- Avoids dangerous areas
- Shares discoveries with nearby agents
- Prefers areas with valuable resources
"""

# Automatically converts to Active Inference parameters
agent = create_agent_from_description(description)

# The GNN parser creates appropriate:
# - State space definitions
# - Action repertoires
# - Preference structures
# - Belief initialization
```

### GMN File Format

Create reusable agent specifications:

```yaml
# explorer_template.gmn
agent_type: Explorer
description: "Autonomous exploration agent with safety awareness"

states:
  - location: [continuous, x_range: 0-100, y_range: 0-100]
  - energy: [discrete, levels: [low, medium, high]]
  - knowledge: [discrete, areas: [unknown, explored, mapped]]

actions:
  - move: [direction: [north, south, east, west], distance: [1, 2, 3]]
  - observe: [range: [short, medium, long]]
  - communicate: [message_type: [discovery, warning, request]]
  - rest: [duration: [short, long]]

preferences:
  - high_energy_exploration: 0.8
  - safety_maintenance: 0.9
  - knowledge_sharing: 0.6
  - resource_efficiency: 0.7

constraints:
  - no_movement_when_low_energy
  - communicate_discoveries_within_range
  - avoid_known_dangerous_areas
```

Load GMN specifications:

```python
from freeagentics.gnn import load_agent_from_gmn

# Load pre-defined agent template
agent = load_agent_from_gmn("explorer_template.gmn")

# Customize loaded template
agent.update_preferences({
    'high_energy_exploration': 0.9,  # Increase exploration drive
    'safety_maintenance': 0.95       # Increase safety priority
})
```

---

## World Simulation

### Creating Environments

```python
from freeagentics import World, Environment

# Basic grid world
world = World(
    grid_size=50,
    resource_density=0.2,     # 20% of cells contain resources
    obstacle_density=0.1,     # 10% of cells are obstacles
    communication_range=5     # Agents can communicate within 5 cells
)

# Add environmental dynamics
world.add_weather_system(
    patterns=['sunny', 'rainy', 'stormy'],
    transition_probability=0.1,  # 10% chance of weather change per step
    effects={
        'rainy': {'movement_cost': 1.5, 'visibility': 0.7},
        'stormy': {'movement_cost': 2.0, 'visibility': 0.3}
    }
)

# Add resource regeneration
world.add_resource_regeneration(
    rate=0.05,               # 5% chance per step per empty cell
    resource_types=['food', 'materials', 'energy'],
    distribution='clustered'  # Resources appear in clusters
)
```

### Advanced Spatial Features

```python
# Hex-based world (better for movement)
from freeagentics.world import HexWorld

hex_world = HexWorld(
    radius=25,                # 25-hex radius from center
    elevation_variance=0.3,   # Terrain elevation differences
    river_probability=0.1,    # Chance of rivers affecting movement
    biome_diversity=True      # Multiple terrain types
)

# Real-world geographic integration
from freeagentics.world import GeographicWorld

geo_world = GeographicWorld(
    bounds=(37.7749, -122.4194, 37.8049, -122.3894),  # San Francisco area
    resolution='high',        # High-detail geographic data
    real_time_weather=True,   # Use actual weather APIs
    traffic_simulation=True   # Simulate real traffic patterns
)
```

### Multi-Scale Environments

```python
# Hierarchical world with multiple scales
from freeagentics.world import MultiScaleWorld

multi_world = MultiScaleWorld(
    scales={
        'macro': World(grid_size=100),      # Large-scale strategic view
        'micro': World(grid_size=1000),     # Detailed local view
        'nano': World(grid_size=10000)      # Individual interaction view
    },
    scale_transitions={
        'macro_to_micro': 10,               # 1 macro cell = 10 micro cells
        'micro_to_nano': 10                 # 1 micro cell = 10 nano cells
    }
)

# Agents can operate at appropriate scales
strategic_agent = Agent.create("Commander", scale='macro')
tactical_agent = Agent.create("Squad", scale='micro')
individual_agent = Agent.create("Explorer", scale='nano')
```

---

## Best Practices

### Performance Optimization

```python
# Vectorized operations for large agent populations
from freeagentics.optimization import VectorizedWorld

# Efficiently handle 1000+ agents
large_world = VectorizedWorld(
    grid_size=200,
    max_agents=5000,
    batch_size=100,          # Process agents in batches
    gpu_acceleration=True,   # Use GPU for belief updates
    jit_compilation=True     # Numba JIT for critical paths
)

# Enable agent pooling to reduce memory allocation
large_world.enable_agent_pooling(
    pool_size=1000,
    recycle_inactive=True
)
```

### Monitoring and Debugging

```python
from freeagentics.monitoring import AgentMonitor, WorldMonitor

# Track agent behavior over time
monitor = AgentMonitor(agent)
monitor.track_metrics([
    'free_energy',           # Active Inference objective
    'belief_entropy',        # Uncertainty level
    'action_diversity',      # Behavioral variety
    'coalition_membership'   # Social connections
])

# World-level monitoring
world_monitor = WorldMonitor(world)
world_monitor.track_emergent_properties([
    'coalition_formation_rate',
    'resource_distribution_entropy',
    'agent_spatial_clustering',
    'information_flow_patterns'
])

# Real-time visualization
monitor.start_live_dashboard(port=8080)
```

### Production Deployment

```python
# Configuration for production environments
from freeagentics.deployment import ProductionConfig

config = ProductionConfig(
    # Reliability
    checkpointing=True,           # Save state periodically
    checkpoint_interval=1000,     # Every 1000 steps
    auto_recovery=True,           # Restart from checkpoints on failure

    # Scalability
    distributed_computing=True,   # Multi-node execution
    load_balancing='dynamic',     # Adaptive agent distribution
    memory_optimization=True,     # Efficient memory usage

    # Monitoring
    logging_level='INFO',         # Production logging
    metrics_collection=True,      # Performance metrics
    alerting_enabled=True,        # System health alerts

    # Security
    secure_communication=True,    # Encrypted agent communication
    access_control='rbac',        # Role-based access control
    audit_logging=True            # Security event logging
)

# Deploy with production configuration
world = World.create_production(config=config)
```

---

## Common Patterns

### Pattern 1: Information Cascade

```python
# Agents share discoveries, creating information waves
def setup_information_cascade():
    # Create diverse agent network
    agents = [
        Agent.create("Explorer", name=f"Scout_{i}",
                    communication_range=3)
        for i in range(20)
    ]

    # Sparse initial placement
    world = World(grid_size=50)
    world.scatter_agents(agents, density=0.1)

    # One agent discovers important information
    agents[0].add_knowledge("valuable_resource_location", (25, 25))

    return world, agents

# Run simulation to watch information spread
world, agents = setup_information_cascade()
for step in range(100):
    world.step()

    # Track information propagation
    informed_agents = [a for a in agents
                      if a.has_knowledge("valuable_resource_location")]
    print(f"Step {step}: {len(informed_agents)} agents informed")
```

### Pattern 2: Emergent Specialization

```python
# Agents develop specialized roles based on experience
def setup_emergent_specialization():
    # Start with identical general-purpose agents
    agents = [Agent.create("General", name=f"Agent_{i}") for i in range(10)]

    # Diverse environment with different challenges
    world = World(grid_size=30)
    world.add_zones({
        'resource_rich': [(0, 0, 10, 10)],      # Good for gathering
        'dangerous': [(20, 20, 30, 30)],        # Requires caution
        'communication_hub': [(15, 15, 20, 20)] # Social interaction
    })

    return world, agents

# Agents adapt to their environment and develop specializations
world, agents = setup_emergent_specialization()
for step in range(500):
    world.step()

    # Check for specialization emergence
    if step % 100 == 0:
        for agent in agents:
            specialization = agent.get_dominant_behavior()
            print(f"{agent.name}: specialized in {specialization}")
```

### Pattern 3: Hierarchical Coordination

```python
# Multi-level command structure emerges from local interactions
def setup_hierarchical_coordination():
    # Create agents with different leadership capabilities
    commanders = [Agent.create("Guardian", name=f"Cmd_{i}",
                              leadership_weight=0.8) for i in range(3)]

    lieutenants = [Agent.create("Explorer", name=f"Lt_{i}",
                               leadership_weight=0.4) for i in range(6)]

    soldiers = [Agent.create("General", name=f"Soldier_{i}",
                            leadership_weight=0.1) for i in range(15)]

    all_agents = commanders + lieutenants + soldiers

    # Mission-oriented environment
    world = World(grid_size=40)
    world.add_objectives([
        'secure_perimeter',
        'gather_intelligence',
        'establish_supply_lines'
    ])

    return world, all_agents

# Watch hierarchical structure emerge
world, agents = setup_hierarchical_coordination()
for step in range(300):
    world.step()

    # Analyze command structure
    if step % 50 == 0:
        hierarchy = world.analyze_command_structure()
        print(f"Step {step}: Command depth = {hierarchy.depth}")
        print(f"  Span of control = {hierarchy.average_span}")
```

---

## Next Steps

### Advanced Features

- **Multi-Modal Agents**: Vision, audio, and sensor integration
- **Continuous State Spaces**: Real-valued belief states
- **Hierarchical Active Inference**: Multi-level planning
- **Cultural Evolution**: Agent societies and traditions

### Integration Opportunities

- **LLM Integration**: Natural language understanding and generation
- **Robotics**: Physical embodiment of Active Inference agents
- **Game Development**: NPCs with genuine intelligence
- **Scientific Modeling**: Cognitive science research platform

### Community Resources

- **Discord Server**: Real-time community support
- **GitHub Discussions**: Technical questions and feature requests
- **Academic Papers**: Research publications using FreeAgentics
- **Example Gallery**: Community-contributed implementations

---

_This guide represents the collaborative expertise of the FreeAgentics community and expert committee. For technical questions, consult the [Developer Guide](DEVELOPER-GUIDE.md) or [API Reference](API-REFERENCE.md)._

---

## Appendix: Glossary

*(Former standalone Glossary merged for convenience.)*

# FreeAgentics Glossary

This glossary provides definitions for technical terms used throughout the FreeAgentics project documentation.

## A

### Active Inference

A theoretical framework ...
