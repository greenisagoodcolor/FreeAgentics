# Active Inference Implementation Guide

## Overview

This guide provides practical instructions for implementing Active Inference systems, bridging the mathematical framework with concrete code implementations. It covers both continuous and discrete state-spaces, integration with neural networks, and optimization techniques.

## Architecture Overview

### Core Components

1. **Generative Model**: Encodes beliefs about how observations arise from hidden states
2. **Inference Engine**: Updates beliefs based on observations
3. **Planning Module**: Evaluates policies based on expected free energy
4. **Action Selector**: Chooses actions to minimize expected free energy
5. **Learning Module**: Updates model parameters from experience

### System Flow

```
Observation → Inference → Planning → Action → Environment
     ↑                                             ↓
     └─────────────── Learning ←──────────────────┘
```

## Implementation Strategy

### 1. Choose State-Space Type

**Discrete State-Space** (Recommended for starting):

- Finite number of states and actions
- Exact inference possible
- Matrix/tensor operations
- Examples: Grid worlds, decision trees, finite state machines

**Continuous State-Space**:

- Infinite possible states
- Approximate inference required
- Gradient-based optimization
- Examples: Robot control, continuous navigation

### 2. Define the Generative Model

#### For Discrete Systems

```python
class DiscreteGenerativeModel:
    def __init__(self, num_states, num_obs, num_actions):
        # A matrix: P(o|s)
        self.A = np.random.dirichlet(np.ones(num_obs), size=num_states).T

        # B tensor: P(s'|s,a)
        self.B = np.zeros((num_states, num_states, num_actions))
        for a in range(num_actions):
            self.B[:,:,a] = np.random.dirichlet(np.ones(num_states), size=num_states).T

        # C matrix: preferences over observations
        self.C = np.zeros((num_obs, 1))  # To be set based on task

        # D vector: initial state prior
        self.D = np.ones(num_states) / num_states
```

#### For Continuous Systems

```python
class ContinuousGenerativeModel:
    def __init__(self, state_dim, obs_dim, action_dim):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Observation model parameters
        self.obs_mean_fn = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim)
        )
        self.obs_log_var = nn.Parameter(torch.zeros(obs_dim))

        # Transition model parameters
        self.trans_mean_fn = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        self.trans_log_var = nn.Parameter(torch.zeros(state_dim))

        # Prior preferences (C matrix equivalent)
        self.preferred_obs = nn.Parameter(torch.zeros(obs_dim))
```

### 3. Implement Inference

#### Discrete Inference

```python
class DiscreteInference:
    def __init__(self, model):
        self.model = model
        self.belief_state = model.D.copy()

    def update_beliefs(self, observation):
        """Update beliefs given new observation"""
        # Convert observation to one-hot if needed
        if isinstance(observation, int):
            o = np.zeros(self.model.A.shape[0])
            o[observation] = 1
        else:
            o = observation

        # Bayesian update
        likelihood = self.model.A.T @ o
        self.belief_state = self._normalize(likelihood * self.belief_state)

        return self.belief_state

    def predict_state(self, action):
        """Predict next state given action"""
        self.belief_state = self.model.B[:, :, action] @ self.belief_state
        return self.belief_state

    @staticmethod
    def _normalize(x):
        return x / (x.sum() + 1e-8)
```

#### Continuous Inference (Variational)

```python
class ContinuousInference:
    def __init__(self, model, state_dim):
        self.model = model
        self.state_dim = state_dim

        # Variational parameters
        self.q_mean = nn.Parameter(torch.zeros(state_dim))
        self.q_log_var = nn.Parameter(torch.zeros(state_dim))

        # Optimizer for variational parameters
        self.optimizer = torch.optim.Adam([self.q_mean, self.q_log_var], lr=0.01)

    def update_beliefs(self, observation, num_steps=10):
        """Update beliefs using gradient descent on free energy"""
        observation = torch.tensor(observation, dtype=torch.float32)

        for _ in range(num_steps):
            self.optimizer.zero_grad()

            # Sample from current belief
            eps = torch.randn_like(self.q_mean)
            state_sample = self.q_mean + torch.exp(0.5 * self.q_log_var) * eps

            # Compute free energy
            F = self._free_energy(state_sample, observation)

            # Minimize free energy
            F.backward()
            self.optimizer.step()

        return self.q_mean.detach(), torch.exp(self.q_log_var).detach()

    def _free_energy(self, state, observation):
        """Compute variational free energy"""
        # Observation likelihood
        obs_mean = self.model.obs_mean_fn(state)
        obs_var = torch.exp(self.model.obs_log_var)
        log_likelihood = -0.5 * torch.sum((observation - obs_mean)**2 / obs_var)

        # KL divergence from prior
        kl_div = -0.5 * torch.sum(1 + self.q_log_var - self.q_mean**2 - torch.exp(self.q_log_var))

        return -log_likelihood + kl_div
```

### 4. Implement Planning

#### Expected Free Energy Calculation

```python
class ActiveInferencePlanner:
    def __init__(self, model, inference_engine):
        self.model = model
        self.inference = inference_engine

    def evaluate_policy(self, policy, current_belief, horizon=5):
        """Evaluate a policy using expected free energy"""
        G = 0  # Expected free energy
        belief = current_belief.copy()

        for t, action in enumerate(policy[:horizon]):
            # Predict future belief
            belief = self._predict_belief(belief, action)

            # Expected observations
            expected_obs = self.model.A @ belief

            # Pragmatic value (goal-seeking)
            if t < self.model.C.shape[1]:
                G -= expected_obs @ self.model.C[:, t]

            # Epistemic value (information-seeking)
            # Entropy of predicted observations
            H_obs = -np.sum(expected_obs * np.log(expected_obs + 1e-8))

            # Expected entropy (uncertainty after observing)
            H_posterior = 0
            for o in range(len(expected_obs)):
                if expected_obs[o] > 1e-8:
                    # Entropy of posterior given this observation
                    post = self._posterior_given_obs(belief, o)
                    H_post_o = -np.sum(post * np.log(post + 1e-8))
                    H_posterior += expected_obs[o] * H_post_o

            # Information gain
            G -= (H_obs - H_posterior)

        return G

    def _predict_belief(self, belief, action):
        """Predict future belief after taking action"""
        return self.model.B[:, :, action] @ belief

    def _posterior_given_obs(self, prior, obs_idx):
        """Compute posterior belief given a specific observation"""
        likelihood = self.model.A[obs_idx, :]
        posterior = likelihood * prior
        return posterior / (posterior.sum() + 1e-8)
```

### 5. Implement Action Selection

```python
class ActionSelector:
    def __init__(self, planner, num_actions, temperature=1.0):
        self.planner = planner
        self.num_actions = num_actions
        self.temperature = temperature

    def select_action(self, current_belief, num_samples=100, horizon=5):
        """Select action by sampling and evaluating policies"""
        # Generate random policies
        policies = self._generate_policies(num_samples, horizon)

        # Evaluate each policy
        G_values = []
        for policy in policies:
            G = self.planner.evaluate_policy(policy, current_belief, horizon)
            G_values.append(G)

        # Convert to probabilities using softmax
        G_values = np.array(G_values)
        log_probs = -G_values / self.temperature
        log_probs -= log_probs.max()  # Numerical stability
        probs = np.exp(log_probs)
        probs /= probs.sum()

        # Compute action probabilities
        action_probs = np.zeros(self.num_actions)
        for i, policy in enumerate(policies):
            if len(policy) > 0:
                action_probs[policy[0]] += probs[i]

        # Sample action
        action_probs /= action_probs.sum()
        return np.random.choice(self.num_actions, p=action_probs)

    def _generate_policies(self, num_samples, horizon):
        """Generate random policy samples"""
        policies = []
        for _ in range(num_samples):
            policy = np.random.randint(0, self.num_actions, size=horizon)
            policies.append(policy)
        return policies
```

### 6. Implement Learning

```python
class ActiveInferenceLearner:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

        # Statistics accumulators
        self.A_counts = np.zeros_like(model.A)
        self.B_counts = np.zeros_like(model.B)
        self.count = 0

    def update_model(self, trajectory):
        """Update model parameters from experience"""
        observations, states, actions = trajectory

        # Update A matrix (observation model)
        for o, s in zip(observations, states):
            if isinstance(s, np.ndarray):  # Belief state
                self.A_counts += np.outer(o, s)
            else:  # Single state
                self.A_counts[o, s] += 1

        # Update B tensor (transition model)
        for t in range(len(states) - 1):
            s_curr = states[t]
            s_next = states[t + 1]
            a = actions[t]

            if isinstance(s_curr, np.ndarray):
                self.B_counts[:, :, a] += np.outer(s_next, s_curr)
            else:
                self.B_counts[s_next, s_curr, a] += 1

        self.count += 1

        # Periodic update with smoothing
        if self.count % 10 == 0:
            self._apply_updates()

    def _apply_updates(self):
        """Apply accumulated updates to model"""
        # Update A with Dirichlet prior
        A_new = (self.A_counts + 1) / (self.A_counts + 1).sum(axis=0)
        self.model.A = (1 - self.learning_rate) * self.model.A + self.learning_rate * A_new

        # Update B with Dirichlet prior
        for a in range(self.model.B.shape[2]):
            B_new = (self.B_counts[:, :, a] + 1) / (self.B_counts[:, :, a] + 1).sum(axis=0)
            self.model.B[:, :, a] = (1 - self.learning_rate) * self.model.B[:, :, a] + self.learning_rate * B_new
```

## Complete Active Inference Agent

```python
class ActiveInferenceAgent:
    def __init__(self, num_states, num_obs, num_actions, temperature=1.0):
        # Initialize components
        self.model = DiscreteGenerativeModel(num_states, num_obs, num_actions)
        self.inference = DiscreteInference(self.model)
        self.planner = ActiveInferencePlanner(self.model, self.inference)
        self.action_selector = ActionSelector(self.planner, num_actions, temperature)
        self.learner = ActiveInferenceLearner(self.model)

        # History for learning
        self.observation_history = []
        self.state_history = []
        self.action_history = []

    def set_preferences(self, preferred_obs):
        """Set goal preferences"""
        self.model.C = preferred_obs

    def perceive(self, observation):
        """Process new observation"""
        # Update beliefs
        belief = self.inference.update_beliefs(observation)

        # Store history
        self.observation_history.append(observation)
        self.state_history.append(belief.copy())

        return belief

    def act(self):
        """Select and execute action"""
        # Get current belief
        current_belief = self.inference.belief_state

        # Select action
        action = self.action_selector.select_action(current_belief)

        # Predict next state
        self.inference.predict_state(action)

        # Store action
        self.action_history.append(action)

        return action

    def learn(self):
        """Update model from experience"""
        if len(self.observation_history) > 1:
            trajectory = (
                self.observation_history,
                self.state_history,
                self.action_history
            )
            self.learner.update_model(trajectory)

    def reset(self):
        """Reset agent for new episode"""
        self.inference.belief_state = self.model.D.copy()
        self.observation_history = []
        self.state_history = []
        self.action_history = []
```

## Integration with GNN

To integrate Active Inference with Graph Neural Networks:

```python
class GNNActiveInferenceAgent(ActiveInferenceAgent):
    def __init__(self, gnn_model, num_states, num_actions):
        # GNN processes observations to states
        self.gnn = gnn_model
        num_obs = gnn_model.output_dim

        super().__init__(num_states, num_obs, num_actions)

    def perceive(self, graph_observation):
        """Process graph observation through GNN"""
        # Extract features using GNN
        with torch.no_grad():
            node_features = graph_observation.x
            edge_index = graph_observation.edge_index

            # Get graph-level representation
            gnn_output = self.gnn(node_features, edge_index)

            # Convert to observation vector
            observation = gnn_output.numpy()

        # Continue with standard Active Inference
        return super().perceive(observation)
```

## Performance Optimization

### 1. Caching

```python
from functools import lru_cache

class CachedPlanner(ActiveInferencePlanner):
    @lru_cache(maxsize=1000)
    def evaluate_policy(self, policy_tuple, belief_tuple, horizon):
        # Convert back from tuples
        policy = np.array(policy_tuple)
        belief = np.array(belief_tuple)

        return super().evaluate_policy(policy, belief, horizon)
```

### 2. Parallel Policy Evaluation

```python
from multiprocessing import Pool

class ParallelActionSelector(ActionSelector):
    def select_action(self, current_belief, num_samples=100, horizon=5):
        policies = self._generate_policies(num_samples, horizon)

        # Parallel evaluation
        with Pool() as pool:
            G_values = pool.starmap(
                self.planner.evaluate_policy,
                [(p, current_belief, horizon) for p in policies]
            )

        # Rest of selection logic...
```

### 3. GPU Acceleration

```python
class GPUActiveInference:
    def __init__(self, model):
        # Move model to GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert numpy arrays to torch tensors
        self.A = torch.from_numpy(model.A).float().to(self.device)
        self.B = torch.from_numpy(model.B).float().to(self.device)
        self.C = torch.from_numpy(model.C).float().to(self.device)
        self.D = torch.from_numpy(model.D).float().to(self.device)
```

## Debugging and Visualization

```python
class ActiveInferenceDebugger:
    def __init__(self, agent):
        self.agent = agent

    def visualize_beliefs(self):
        """Plot belief distribution"""
        import matplotlib.pyplot as plt

        beliefs = np.array(self.agent.state_history)
        plt.imshow(beliefs.T, aspect='auto', cmap='hot')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.colorbar(label='Belief')
        plt.title('Belief Evolution')
        plt.show()

    def plot_free_energy(self):
        """Track free energy over time"""
        F_values = []
        for t, (o, s) in enumerate(zip(self.agent.observation_history,
                                      self.agent.state_history)):
            F = self._compute_free_energy(o, s)
            F_values.append(F)

        plt.plot(F_values)
        plt.xlabel('Time')
        plt.ylabel('Free Energy')
        plt.title('Free Energy Minimization')
        plt.show()
```

## Best Practices

1. **Start Simple**: Begin with discrete state-spaces and known environments
2. **Validate Components**: Test each component (inference, planning, etc.) independently
3. **Monitor Convergence**: Track free energy to ensure it decreases over time
4. **Tune Hyperparameters**: Temperature, learning rate, and horizon length significantly impact performance
5. **Handle Numerical Stability**: Use log-space computations and add small constants to prevent division by zero
6. **Profile Performance**: Identify bottlenecks, especially in policy evaluation
7. **Incremental Complexity**: Add features like hierarchical models or continuous states only after basic implementation works

## Common Pitfalls and Solutions

| Problem                | Solution                                          |
| ---------------------- | ------------------------------------------------- |
| Beliefs don't converge | Check observation model (A matrix) conditioning   |
| Agent doesn't explore  | Increase epistemic value weight or temperature    |
| Poor goal-seeking      | Verify C matrix encodes preferences correctly     |
| Numerical instability  | Switch to log-space computations                  |
| Slow planning          | Reduce policy samples or horizon length           |
| Memory issues          | Use sparse representations for large state spaces |

## Conclusion

This implementation guide provides a practical foundation for building Active Inference systems. The modular architecture allows for experimentation with different components while maintaining the core principles of free energy minimization. Start with the discrete implementation for learning, then extend to continuous states and deep learning integration as needed.
