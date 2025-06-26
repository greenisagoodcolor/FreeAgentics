# Discrete State-Space Active Inference

## Overview

This document describes the implementation of Active Inference for discrete state-spaces, which is particularly relevant for agent-based systems operating in discrete environments. This formulation uses categorical distributions and matrix operations for efficient computation.

## Discrete Generative Model

### 1. Model Components

In discrete state-spaces, the generative model consists of:

- **A matrix** (observation model): `P(o_t | s_t)`
- **B tensor** (transition model): `P(s_{t+1} | s_t, a_t)`
- **C matrix** (preference model): `P(o_t | C)`
- **D vector** (initial state prior): `P(s_0)`

### 2. Matrix Representations

#### A Matrix - Observation Model

```
A[o,s] = P(o_t = o | s_t = s)
```

- Shape: `(num_observations, num_states)`
- Each column sums to 1 (probability distribution over observations)

#### B Tensor - Transition Model

```
B[s',s,a] = P(s_{t+1} = s' | s_t = s, a_t = a)
```

- Shape: `(num_states, num_states, num_actions)`
- Each column sums to 1 for each action

#### C Matrix - Preferences

```
C[o,t] = log P(o_t = o | C)
```

- Shape: `(num_observations, time_horizon)`
- Encodes desired observations at each time step

#### D Vector - Initial Prior

```
D[s] = P(s_0 = s)
```

- Shape: `(num_states,)`
- Sums to 1

## Belief Updates

### 1. State Estimation

Posterior beliefs over states are updated using:

```python
def update_beliefs(o_t, A, prior_s):
    """
    Update beliefs given new observation

    Args:
        o_t: Current observation (one-hot vector)
        A: Observation model
        prior_s: Prior belief over states

    Returns:
        posterior_s: Updated belief over states
    """
    likelihood = A.T @ o_t  # P(o|s)
    posterior_s = normalize(likelihood * prior_s)
    return posterior_s
```

### 2. Sequential Inference

For sequential observations:

```python
def sequential_inference(observations, A, B, D, actions):
    """
    Perform sequential state estimation

    Args:
        observations: Sequence of observations
        A: Observation model
        B: Transition model
        D: Initial state prior
        actions: Sequence of actions taken

    Returns:
        beliefs: Sequence of posterior beliefs
    """
    T = len(observations)
    beliefs = []

    # Initial belief
    s = D

    for t in range(T):
        # Update with observation
        s = update_beliefs(observations[t], A, s)
        beliefs.append(s)

        # Predict next state if not last timestep
        if t < T - 1:
            s = B[:, :, actions[t]] @ s

    return beliefs
```

## Action Selection

### 1. Expected Free Energy Calculation

For discrete systems, expected free energy is computed as:

```python
def expected_free_energy(qs, A, B, C, policy, tau=1):
    """
    Calculate expected free energy for a policy

    Args:
        qs: Current belief state
        A: Observation model
        B: Transition model
        C: Preference matrix
        policy: Sequence of actions
        tau: Time horizon

    Returns:
        G: Expected free energy
    """
    G = 0
    s = qs

    for t in range(tau):
        # Predict future state
        if t < len(policy):
            s = B[:, :, policy[t]] @ s

        # Expected observations
        qo = A @ s

        # Pragmatic value (preference satisfaction)
        G -= qo @ C[:, t]

        # Epistemic value (information gain)
        H_A = -np.sum(A * safe_log(A), axis=0)  # Entropy of each column
        G -= s @ H_A  # Expected entropy

    return G
```

### 2. Policy Selection

Select actions by evaluating multiple policies:

```python
def select_action(qs, A, B, C, num_policies=16, policy_len=3):
    """
    Select action using active inference

    Args:
        qs: Current belief state
        A: Observation model
        B: Transition model
        C: Preferences
        num_policies: Number of random policies to evaluate
        policy_len: Length of policy sequences

    Returns:
        action: Selected action
    """
    # Generate candidate policies
    policies = generate_policies(num_policies, policy_len, num_actions)

    # Calculate expected free energy for each
    G = np.zeros(num_policies)
    for i, policy in enumerate(policies):
        G[i] = expected_free_energy(qs, A, B, C, policy)

    # Convert to action probabilities using softmax
    log_probs = -G / temperature
    probs = softmax(log_probs)

    # Sample action from first step of policies
    action_probs = np.zeros(num_actions)
    for i, policy in enumerate(policies):
        action_probs[policy[0]] += probs[i]

    # Select action
    action = np.random.choice(num_actions, p=normalize(action_probs))

    return action
```

## Parameter Learning

### 1. Learning A Matrix

Update observation model based on experience:

```python
def update_A(A, observations, states, learning_rate=0.01):
    """
    Update observation model using experience

    Args:
        A: Current observation model
        observations: Observed data
        states: Inferred states
        learning_rate: Learning rate

    Returns:
        A_new: Updated observation model
    """
    # Accumulate statistics
    counts = np.zeros_like(A)

    for o, s in zip(observations, states):
        counts += np.outer(o, s)

    # Normalize and update
    A_new = normalize(counts + 1e-6, axis=0)  # Add small constant for stability

    # Smooth update
    A = (1 - learning_rate) * A + learning_rate * A_new

    return normalize(A, axis=0)
```

### 2. Learning B Tensor

Update transition model:

```python
def update_B(B, states, actions, learning_rate=0.01):
    """
    Update transition model using experience

    Args:
        B: Current transition model
        states: Sequence of states
        actions: Sequence of actions
        learning_rate: Learning rate

    Returns:
        B_new: Updated transition model
    """
    counts = np.zeros_like(B)

    for t in range(len(states) - 1):
        s_t = states[t]
        s_next = states[t + 1]
        a_t = actions[t]

        counts[:, :, a_t] += np.outer(s_next, s_t)

    # Normalize per action
    for a in range(B.shape[2]):
        B[:, :, a] = normalize(counts[:, :, a] + 1e-6, axis=0)

    return B
```

## Efficient Implementation Tips

### 1. Log-Space Computations

To avoid numerical underflow:

```python
def log_stable_mult(log_A, log_x):
    """Multiply in log space: exp(log_A) @ exp(log_x)"""
    return logsumexp(log_A + log_x[None, :], axis=1)

def normalize_log(log_x):
    """Normalize in log space"""
    return log_x - logsumexp(log_x)
```

### 2. Sparse Representations

For large state spaces:

```python
from scipy.sparse import csr_matrix

# Use sparse matrices for A and B when many entries are zero
A_sparse = csr_matrix(A)
B_sparse = [csr_matrix(B[:, :, a]) for a in range(num_actions)]
```

### 3. Vectorized Operations

Leverage NumPy broadcasting:

```python
# Vectorized expected free energy for all policies at once
def batch_expected_free_energy(qs, A, B, C, policies):
    """Calculate G for multiple policies simultaneously"""
    num_policies = len(policies)
    tau = len(policies[0])

    # Broadcast computations
    G = np.zeros(num_policies)
    states = np.tile(qs, (num_policies, 1)).T

    for t in range(tau):
        # Apply transitions for each policy
        for i, policy in enumerate(policies):
            if t < len(policy):
                states[:, i] = B[:, :, policy[t]] @ states[:, i]

        # Compute expected observations
        qo = A @ states  # Shape: (num_obs, num_policies)

        # Pragmatic value
        G -= np.sum(qo * C[:, t:t+1], axis=0)

        # Epistemic value
        H_A = -np.sum(A * safe_log(A), axis=0)
        G -= states.T @ H_A

    return G
```

## Example: Grid World Agent

```python
class GridWorldActiveInference:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.num_states = grid_size * grid_size
        self.num_actions = 4  # up, down, left, right
        self.num_observations = self.num_states

        # Initialize model parameters
        self.A = np.eye(self.num_observations)  # Perfect observation
        self.B = self._construct_transition_model()
        self.C = self._construct_preferences()
        self.D = np.ones(self.num_states) / self.num_states

    def _construct_transition_model(self):
        """Build transition model for grid world"""
        B = np.zeros((self.num_states, self.num_states, self.num_actions))

        for s in range(self.num_states):
            x, y = s // self.grid_size, s % self.grid_size

            # Action 0: Up
            new_x = max(0, x - 1)
            B[new_x * self.grid_size + y, s, 0] = 1

            # Action 1: Down
            new_x = min(self.grid_size - 1, x + 1)
            B[new_x * self.grid_size + y, s, 1] = 1

            # Action 2: Left
            new_y = max(0, y - 1)
            B[x * self.grid_size + new_y, s, 2] = 1

            # Action 3: Right
            new_y = min(self.grid_size - 1, y + 1)
            B[x * self.grid_size + new_y, s, 3] = 1

        return B

    def _construct_preferences(self, goal_state=12):
        """Set preferences (goal in center of grid)"""
        C = np.zeros((self.num_observations, 10))  # 10 timesteps
        C[goal_state, :] = 10  # High preference for goal state
        return C
```

## Conclusion

Discrete state-space Active Inference provides an efficient framework for implementing intelligent agents. The matrix formulation enables fast computation and clear interpretation of model components. Key advantages include:

- Tractable exact inference for moderate state spaces
- Clear separation of model components (A, B, C, D)
- Efficient matrix operations using modern linear algebra libraries
- Natural integration with discrete decision-making scenarios
