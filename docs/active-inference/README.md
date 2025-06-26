# Active Inference Framework for FreeAgentics

## Overview

The Active Inference framework in FreeAgentics provides a comprehensive implementation of the Free Energy Principle and Active Inference for creating adaptive, intelligent agents. This framework enables agents to:

- **Perceive** their environment through probabilistic inference
- **Learn** from experience by updating their generative models
- **Act** to minimize expected free energy and achieve their goals
- **Adapt** to changing environments through precision control and model updates

## Key Components

### 1. Generative Models

- **Discrete Models**: For categorical state spaces
- **Continuous Models**: For continuous state representations
- **Hybrid Models**: Combining discrete and continuous aspects

### 2. Inference Algorithms

- **Variational Inference**: Efficient approximate inference
- **Particle Filters**: For non-linear continuous models
- **Message Passing**: For hierarchical models

### 3. Policy Selection

- **PyMDP Integration**: Peer-reviewed expected free energy calculations
- **Sophisticated Inference**: Multi-step planning
- **Habit Learning**: Efficient cached policies

### 4. Learning Mechanisms

- **Parameter Learning**: Online Bayesian learning of model parameters
- **Structure Learning**: Discovering model structure from data
- **Active Learning**: Directed exploration for model improvement

### 5. Advanced Features

- **Hierarchical Inference**: Multi-level temporal abstractions
- **Precision Control**: Adaptive uncertainty weighting
- **Computational Optimization**: GPU acceleration and sparse operations

## Quick Start

### Installation

```bash
# Install required dependencies
pip install torch numpy scipy matplotlib networkx pymdp

# Install FreeAgentics (if not already installed)
pip install -e .
```

### Basic Usage

```python
import torch
from inference.engine import (
    DiscreteGenerativeModel, ModelDimensions, ModelParameters,
    VariationalMessagePassing, InferenceConfig,
    PolicyConfig
)
from inference.engine.pymdp_generative_model import PyMDPGenerativeModel
from inference.engine.pymdp_policy_selector import PyMDPPolicySelector

# Define model dimensions
dims = ModelDimensions(
    num_states=4,
    num_observations=3,
    num_actions=2
)

# Create generative model
params = ModelParameters(use_gpu=torch.cuda.is_available())
gen_model = DiscreteGenerativeModel(dims, params)

# Convert to PyMDP format for validated calculations
pymdp_model = PyMDPGenerativeModel.from_discrete_model(gen_model)

# Create inference engine
inference_config = InferenceConfig(num_iterations=16, convergence_threshold=1e-4)
inference = VariationalMessagePassing(inference_config)

# Create PyMDP-based policy selector
policy_config = PolicyConfig(precision=1.0, planning_horizon=3)
policy_selector = PyMDPPolicySelector(policy_config, pymdp_model)

# Initialize belief state
belief = gen_model.D  # Use model's initial prior

# Perception: Update belief given observation
observation = torch.tensor(1)  # Observed state 1
belief = inference.infer_states(observation, gen_model, belief)

# Action: Select action using PyMDP's validated algorithms
selected_policy = policy_selector.select_policy(belief, pymdp_model)
action = selected_policy.actions[0]  # First action in policy

print(f"Updated belief: {belief}")
print(f"Selected action: {action} with policy probability {selected_policy.probability:.3f}")
```

## Documentation

- [Mathematical Framework](mathematical-framework.md) - Detailed mathematical foundations
- [Implementation Guide](implementation-guide.md) - Step-by-step implementation details
- [API Reference](api-reference.md) - Complete API documentation

## License

This module is part of FreeAgentics and is released under the same license.
