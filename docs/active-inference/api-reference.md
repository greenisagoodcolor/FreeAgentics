# Active Inference API Reference

This document provides detailed API documentation for all Active Inference engine modules in FreeAgentics.

## Core Modules

### inference.engine.generative_model

Contains generative model implementations for different state space types.

#### Classes

##### `ModelDimensions`
Dataclass defining the dimensions of the generative model.

**Attributes:**
- `num_states: int` - Number of hidden states
- `num_observations: int` - Number of observation types
- `num_actions: int` - Number of possible actions

##### `ModelParameters`
Configuration parameters for generative models.

**Attributes:**
- `use_gpu: bool = True` - Whether to use GPU acceleration
- `dtype: torch.dtype = torch.float32` - Data type for tensors
- `precision: float = 1e-6` - Numerical precision threshold

##### `GenerativeModel(ABC)`
Abstract base class for all generative models.

**Abstract Methods:**
- `forward(states: torch.Tensor) -> torch.Tensor` - Forward model prediction
- `observation_model(states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]` - Observation likelihood
- `transition_model(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor` - State transition

##### `DiscreteGenerativeModel(GenerativeModel)`
Generative model for discrete state spaces.

**Parameters:**
- `dims: ModelDimensions` - Model dimensions
- `params: ModelParameters` - Model parameters

**Attributes:**
- `A: torch.Tensor` - Observation model matrix [num_obs x num_states]
- `B: torch.Tensor` - Transition model tensor [num_states x num_states x num_actions]
- `C: torch.Tensor` - Preference matrix [num_obs x time_horizon]
- `D: torch.Tensor` - Initial state prior [num_states]

**Methods:**
- `update_model(observations: List[torch.Tensor], beliefs: List[torch.Tensor], actions: List[torch.Tensor])` - Update model parameters from experience

##### `ContinuousGenerativeModel(GenerativeModel)`
Generative model for continuous state spaces using neural networks.

**Parameters:**
- `state_dim: int` - Dimensionality of state space
- `obs_dim: int` - Dimensionality of observation space
- `action_dim: int` - Dimensionality of action space

**Attributes:**
- `obs_mean_fn: nn.Module` - Neural network for observation mean
- `obs_log_var: nn.Parameter` - Log variance for observation noise
- `trans_mean_fn: nn.Module` - Neural network for transition mean
- `trans_log_var: nn.Parameter` - Log variance for transition noise

### inference.engine.active_inference

Core inference algorithms for belief updating.

#### Classes

##### `InferenceConfig`
Configuration for inference algorithms.

**Attributes:**
- `num_iterations: int = 16` - Maximum iterations for convergence
- `convergence_threshold: float = 0.0001` - Convergence criterion
- `learning_rate: float = 0.1` - Learning rate for gradient-based methods
- `use_natural_gradient: bool = True` - Whether to use natural gradients
- `use_gpu: bool = True` - GPU acceleration flag

##### `InferenceAlgorithm(ABC)`
Abstract base class for inference algorithms.

**Abstract Methods:**
- `infer_states(observations: torch.Tensor, generative_model: GenerativeModel, prior: Optional[torch.Tensor]) -> torch.Tensor` - Infer hidden states
- `compute_free_energy(beliefs: torch.Tensor, observations: torch.Tensor, generative_model: GenerativeModel) -> torch.Tensor` - Compute variational free energy

##### `VariationalMessagePassing(InferenceAlgorithm)`
Variational message passing for discrete state spaces.

**Methods:**
- `infer_states(observations, generative_model, prior=None)` - VMP belief updating
- `compute_free_energy(beliefs, observations, generative_model)` - Free energy calculation

**Example:**
```python
from inference.engine import VariationalMessagePassing, InferenceConfig

config = InferenceConfig(num_iterations=10, convergence_threshold=1e-4)
vmp = VariationalMessagePassing(config)
beliefs = vmp.infer_states(observations, generative_model)
```

##### `GradientDescentInference(InferenceAlgorithm)`
Gradient-based inference for continuous state spaces.

**Methods:**
- `infer_states(observations, generative_model, prior=None)` - Returns mean and variance of Gaussian beliefs
- `compute_free_energy(beliefs, observations, generative_model)` - Free energy for continuous states

##### `ParticleFilterInference(InferenceAlgorithm)`
Sequential Monte Carlo inference for non-linear models.

**Parameters:**
- `num_particles: int = 100` - Number of particles to maintain

**Methods:**
- `infer_states(observations, generative_model, prior=None, particles=None, weights=None)` - Returns (mean, particles, weights)

#### Factory Functions

##### `create_inference_algorithm(algorithm_type: str, config: Optional[InferenceConfig] = None, **kwargs) -> InferenceAlgorithm`

Create inference algorithms by type.

**Parameters:**
- `algorithm_type: str` - One of: 'vmp', 'bp', 'gradient', 'natural', 'em', 'particle'
- `config: InferenceConfig` - Algorithm configuration
- `**kwargs` - Algorithm-specific parameters

**Example:**
```python
vmp = create_inference_algorithm('vmp', config)
particle_filter = create_inference_algorithm('particle', config, num_particles=200)
```

### inference.engine.policy_selection

Policy evaluation and action selection using PyMDP's validated expected free energy calculations.

#### Classes

##### `PolicyConfig`
Configuration for policy selection.

**Attributes:**
- `precision: float = 1.0` - Precision parameter for action selection
- `planning_horizon: int = 5` - Number of steps to plan ahead
- `use_sophisticated_inference: bool = False` - Enable tree search planning
- `exploration_bonus: float = 0.1` - Epistemic exploration weight

##### `Policy`
Represents a sequence of actions.

**Attributes:**
- `actions: torch.Tensor` - Action sequence [horizon]
- `probability: float` - Policy probability
- `expected_free_energy: float` - Expected free energy value

##### `PolicySelector(ABC)`
Abstract base class for policy selection.

**Abstract Methods:**
- `evaluate_policies(beliefs: torch.Tensor, generative_model: GenerativeModel) -> torch.Tensor` - Evaluate policy values
- `select_action(policy_values: torch.Tensor) -> int` - Select action from policy values

##### `PyMDPPolicySelector(PolicySelector)`
PyMDP-based expected free energy calculation for discrete models using peer-reviewed algorithms.

**Methods:**
- `select_policy(beliefs, generative_model, preferences)` - Policy selection using pymdp.Agent
- `evaluate_policies(beliefs, generative_model)` - Policy evaluation via pymdp
- `select_action(policy_values)` - Softmax action selection

**Example:**
```python
from inference.engine import PyMDPPolicySelector, PolicyConfig
from inference.engine.pymdp_generative_model import PyMDPGenerativeModel

config = PolicyConfig(precision=2.0, planning_horizon=3)
pymdp_model = PyMDPGenerativeModel.from_discrete_model(discrete_model)
policy_selector = PyMDPPolicySelector(config, pymdp_model)
selected_policy = policy_selector.select_policy(beliefs, pymdp_model, preferences)
```

##### `HierarchicalPolicySelector(PolicySelector)`
Hierarchical policy selection across multiple temporal scales.

**Parameters:**
- `levels: List[PolicyConfig]` - Configuration for each hierarchy level

### inference.engine.belief_update

Belief updating with external model integration.

#### Classes

##### `BeliefUpdateConfig`
Configuration for belief updating systems.

**Attributes:**
- `update_rate: float = 0.1` - Rate of belief updating
- `use_gnn_features: bool = True` - Whether to use GNN features
- `attention_heads: int = 4` - Number of attention heads for graph attention

##### `GNNBeliefUpdater`
Belief updating using Graph Neural Network features.

**Parameters:**
- `gnn_model: nn.Module` - Pre-trained GNN model
- `config: BeliefUpdateConfig` - Update configuration

**Methods:**
- `update_beliefs(graph_obs: Data, current_beliefs: torch.Tensor) -> torch.Tensor` - Update beliefs using GNN features
- `extract_graph_features(graph_obs: Data) -> torch.Tensor` - Extract features from graph observation

### inference.engine.temporal_planning

Temporal planning and tree search algorithms.

#### Classes

##### `PlanningConfig`
Configuration for temporal planning.

**Attributes:**
- `max_depth: int = 5` - Maximum planning depth
- `num_simulations: int = 100` - Number of MCTS simulations
- `exploration_constant: float = 1.414` - UCB exploration constant
- `use_parallel: bool = True` - Enable parallel planning

##### `TemporalPlanner(ABC)`
Abstract base class for temporal planners.

**Abstract Methods:**
- `plan(initial_beliefs: torch.Tensor, generative_model: GenerativeModel, horizon: int) -> List[int]` - Generate action plan
- `evaluate_plan(plan: List[int], beliefs: torch.Tensor, model: GenerativeModel) -> float` - Evaluate plan quality

##### `MonteCarloTreeSearch(TemporalPlanner)`
Monte Carlo Tree Search implementation.

**Methods:**
- `plan(initial_beliefs, generative_model, horizon)` - MCTS planning
- `select_node(node: TreeNode) -> TreeNode` - UCB node selection
- `expand_node(node: TreeNode, model: GenerativeModel)` - Node expansion
- `simulate(node: TreeNode, model: GenerativeModel) -> float` - Rollout simulation
- `backpropagate(node: TreeNode, value: float)` - Value backpropagation

### inference.engine.precision

Precision optimization for Active Inference.

#### Classes

##### `PrecisionConfig`
Configuration for precision optimization.

**Attributes:**
- `initial_precision: float = 1.0` - Initial precision value
- `learning_rate: float = 0.01` - Precision learning rate
- `adaptation_method: str = 'gradient'` - Adaptation method ('gradient', 'hierarchical', 'meta')

##### `PrecisionOptimizer(ABC)`
Abstract base class for precision optimization.

**Abstract Methods:**
- `optimize_precision(observations: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor` - Update precision parameters
- `get_precision() -> torch.Tensor` - Get current precision values

##### `AdaptivePrecisionController(PrecisionOptimizer)`
Adaptive precision control based on prediction errors.

**Methods:**
- `optimize_precision(observations, predictions)` - Adaptive precision update
- `compute_prediction_error(observations, predictions) -> torch.Tensor` - Prediction error calculation

### inference.engine.gnn_integration

Integration with Graph Neural Networks.

#### Classes

##### `GNNIntegrationConfig`
Configuration for GNN integration.

**Attributes:**
- `gnn_hidden_dim: int = 128` - GNN hidden dimension
- `num_gnn_layers: int = 3` - Number of GNN layers
- `aggregation_method: str = 'attention'` - Graph aggregation method

##### `GNNActiveInferenceAdapter`
Adapter for integrating GNN models with Active Inference.

**Parameters:**
- `gnn_model: nn.Module` - Graph neural network model
- `config: GNNIntegrationConfig` - Integration configuration

**Methods:**
- `process_graph_observation(graph_data: Data) -> torch.Tensor` - Process graph observation
- `update_gnn_beliefs(graph_features: torch.Tensor, beliefs: torch.Tensor) -> torch.Tensor` - Update beliefs with GNN features

### inference.engine.pymdp_generative_model

PyMDP integration for mathematically validated Active Inference.

#### Classes

##### `PyMDPGenerativeModel`
Wrapper for pymdp-compatible generative models with validated A/B/C/D matrices.

**Methods:**
- `from_discrete_model(discrete_model: DiscreteGenerativeModel) -> PyMDPGenerativeModel` - Convert from discrete model
- `to_pymdp_format() -> Tuple[np.ndarray, ...]` - Export matrices in pymdp format

**Example:**
```python
from inference.engine.pymdp_generative_model import PyMDPGenerativeModel

# Convert existing discrete model to pymdp format
pymdp_model = PyMDPGenerativeModel.from_discrete_model(discrete_model)
A, B, C, D = pymdp_model.to_pymdp_format()
```

## Utility Functions

### Factory Functions

All modules provide factory functions for easy instantiation:

```python
from inference.engine import (
    create_generative_model,
    create_inference_algorithm,
    create_policy_selector,
    create_temporal_planner,
    create_precision_optimizer,
    create_gnn_adapter,
    create_belief_updater
)

# Create discrete model
model = create_generative_model('discrete', dims, params)

# Create VMP inference
inference = create_inference_algorithm('vmp', config)

# Create PyMDP policy selector
policy_selector = create_policy_selector('pymdp', policy_config)
```

## Complete Example Usage

```python
import torch
from inference.engine import (
    ModelDimensions, ModelParameters,
    DiscreteGenerativeModel,
    VariationalMessagePassing, InferenceConfig,
    PolicyConfig
)
from inference.engine.pymdp_generative_model import PyMDPGenerativeModel
from inference.engine.pymdp_policy_selector import PyMDPPolicySelector

# Set up model
dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
params = ModelParameters(use_gpu=torch.cuda.is_available())
model = DiscreteGenerativeModel(dims, params)

# Convert to PyMDP format
pymdp_model = PyMDPGenerativeModel.from_discrete_model(model)

# Set up inference
inference_config = InferenceConfig(num_iterations=10)
inference = VariationalMessagePassing(inference_config)

# Set up policy selection using PyMDP
policy_config = PolicyConfig(precision=2.0, planning_horizon=3)
policy_selector = PyMDPPolicySelector(policy_config, pymdp_model)

# Agent loop
belief = model.D  # Initial belief
for t in range(100):
    # Observe environment
    observation = get_observation()  # Your observation function

    # Update beliefs
    belief = inference.infer_states(observation, model, belief)

    # Select action using PyMDP
    selected_policy = policy_selector.select_policy(belief, pymdp_model)
    action = selected_policy.actions[0]  # First action in policy

    # Take action in environment
    execute_action(action)  # Your action execution function
```

## Error Handling

All modules include comprehensive error handling:

```python
try:
    beliefs = inference.infer_states(observations, model)
except ValueError as e:
    # Handle dimension mismatches
    print(f"Dimension error: {e}")
except RuntimeError as e:
    # Handle CUDA/computation errors
    print(f"Runtime error: {e}")
```

## Performance Notes

- Use GPU acceleration when available by setting `use_gpu=True`
- Batch observations for better performance
- Use appropriate precision settings to balance accuracy and speed
- Cache frequently used computations with `@lru_cache` decorator
- Consider using mixed precision for large models

## Mathematical Foundations

The implementation follows the mathematical framework:

**Variational Free Energy:**
```
F = E_q[log q(s)] - E_q[log p(o,s)]
  = KL[q(s)||p(s)] - E_q[log p(o|s)]
```

**Expected Free Energy:**
```
G = E_q[log q(s'|π)] - E_q[log p(o',s'|π)]
  = Epistemic Value + Pragmatic Value
```

Where:
- `q(s)` = posterior beliefs over states
- `p(o|s)` = observation likelihood (A matrix)
- `p(s'|s,a)` = transition model (B tensor)
- `π` = policy (sequence of actions)
