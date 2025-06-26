# Active Inference Principles in FreeAgentics

## Introduction

Active Inference is the theoretical foundation of FreeAgentics. It provides a unified framework for perception, learning, and action based on the Free Energy Principle. This guide explains how Active Inference is implemented throughout the platform.

## Core Concepts

### Free Energy Minimization

The fundamental principle: agents act to minimize their free energy, which can be understood as minimizing surprise or uncertainty about the world.

```
Free Energy = Complexity - Accuracy
```

Where:

- **Complexity**: The difference between prior and posterior beliefs
- **Accuracy**: How well beliefs predict observations

### Generative Models

Each agent maintains a generative model of its environment, specified in GNN format:

```gnn
Model: AgentCognition
  States: s ∈ S
  Observations: o ∈ O
  Actions: a ∈ A

  Generative Process:
    P(o|s): observation model
    P(s'|s,a): transition model

  Preferences:
    C(o): preferred observations
```

### Belief Updates

Agents continuously update their beliefs using variational inference:

1. **Perception**: Update beliefs about hidden states given observations
2. **Prediction**: Anticipate future states based on current beliefs
3. **Action Selection**: Choose actions that minimize expected free energy

## Implementation in FreeAgentics

### 1. Agent Base Class

All agents inherit from `ActiveInferenceAgent`:

```python
class ActiveInferenceAgent:
    def __init__(self, gnn_model):
        self.model = gnn_model
        self.beliefs = self.initialize_beliefs()
        self.free_energy_history = []

    def perceive(self, observation):
        # Update beliefs to minimize free energy
        self.beliefs = self.update_beliefs(observation)

    def act(self):
        # Select action that minimizes expected free energy
        return self.select_action(self.beliefs)
```

### 2. Free Energy Calculation

The system calculates free energy for belief evaluation:

```python
def calculate_free_energy(beliefs, observation, model):
    # Complexity term
    complexity = kl_divergence(beliefs, model.prior)

    # Accuracy term
    accuracy = expected_log_likelihood(observation, beliefs, model)

    return complexity - accuracy
```

### 3. Learning Through Experience

Agents learn by updating their generative models to better predict observations:

```python
def learn_from_experience(experience, model):
    # Extract patterns that reduce prediction error
    patterns = extract_patterns(experience)

    # Update model parameters
    if pattern.reduces_free_energy():
        model.integrate_pattern(pattern)
```

## Practical Examples

### Explorer Agent

Explorers minimize uncertainty about the environment:

```gnn
Preferences:
  C_location: uniform  # No location preference
  C_information: high  # Prefer informative observations

Action Policy:
  IF uncertainty(location) > threshold:
    THEN explore_unknown_areas()
```

### Merchant Agent

Merchants minimize surprise about resource availability:

```gnn
Preferences:
  C_resources: high    # Prefer resource-rich observations
  C_trade: positive    # Prefer successful trades

Action Policy:
  IF expected_resources(location) > current:
    THEN move_to(location)
```

## Advanced Topics

### Hierarchical Models

Agents can use hierarchical generative models for complex reasoning:

```gnn
Model: HierarchicalCognition
  Level 1: immediate_environment
  Level 2: local_patterns
  Level 3: global_understanding

  Each level minimizes free energy at its scale
```

### Social Active Inference

When agents interact, they model each other's beliefs:

```gnn
Model: SocialCognition
  Own_beliefs: μ_self
  Other_beliefs: μ_other

  Joint_free_energy = F(μ_self) + F(μ_other|μ_self)
```

## Best Practices

1. **Start Simple**: Begin with basic generative models and add complexity as needed
2. **Monitor Free Energy**: Track free energy to ensure agents are learning effectively
3. **Balance Exploration/Exploitation**: Use information gain to drive exploration
4. **Validate Models**: Ensure GNN models correctly implement Active Inference principles

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active inference: the free energy principle in mind, brain, and behavior
- Heins, C., et al. (2022). pymdp: A Python library for active inference in discrete state spaces

## See Also

- [GNN Model Format](./gnn_models/model_format.md)
- [Agent Implementation Guide](./architecture.md#agents)
- [Example Models](./gnn_models/examples/)
