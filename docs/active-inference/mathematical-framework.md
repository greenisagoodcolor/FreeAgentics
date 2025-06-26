# Mathematical Framework for Active Inference

## Overview

Active Inference is a unified theoretical framework that describes perception, action, and learning as processes of minimizing variational free energy. This document formalizes the mathematical foundations, including the Free Energy Principle, variational inference, and Bayesian belief updating mechanisms.

## Core Mathematical Concepts

### 1. Free Energy Principle

The Free Energy Principle states that biological systems minimize a quantity called variational free energy, which bounds the surprise (negative log evidence) of sensory observations.

#### Variational Free Energy

The variational free energy functional is defined as:

```
F = E_q[log q(s) - log p(s,o)]
```

Where:

- `F` is the variational free energy
- `q(s)` is the approximate posterior distribution over hidden states
- `p(s,o)` is the joint distribution of states and observations
- `E_q[·]` denotes expectation under q

This can be decomposed into:

```
F = D_KL[q(s)||p(s|o)] - log p(o)
```

Where:

- `D_KL[·||·]` is the Kullback-Leibler divergence
- `p(s|o)` is the true posterior
- `p(o)` is the model evidence (marginal likelihood)

Since KL divergence is non-negative, F provides an upper bound on negative log evidence (surprise).

### 2. Generative Model

The generative model specifies the probabilistic relationships between hidden states, observations, and actions:

```
p(õ, s̃, π) = p(s_0) ∏_τ p(o_τ|s_τ) p(s_τ+1|s_τ, a_τ) p(a_τ|π)
```

Where:

- `õ` = {o_0, o_1, ..., o_T} are observations over time
- `s̃` = {s_0, s_1, ..., s_T} are hidden states
- `π` is a policy (sequence of actions)
- `p(o_τ|s_τ)` is the observation model (likelihood)
- `p(s_τ+1|s_τ, a_τ)` is the transition model
- `p(s_0)` is the initial state prior

### 3. Belief Updating

Beliefs are updated through gradient descent on free energy:

```
ṡ = -∂F/∂s = -∂/∂s E_q[log q(s) - log p(s,o)]
```

For mean-field approximations where q(s) = N(μ, Σ):

```
μ̇ = Σ(∂log p(o,s)/∂s)|_{s=μ}
```

This implements precision-weighted prediction error minimization.

### 4. Expected Free Energy

For action selection, agents minimize expected free energy G over policies:

```
G(π) = E_q(o,s|π)[log q(s|π) - log p(s) - log p(o|C)]
```

This can be decomposed into:

```
G(π) = -E_q[D_KL[q(o|π)||p(o|C)]] - E_q[H[p(s|o,π)]]
```

Where:

- First term: pragmatic value (expected preference satisfaction)
- Second term: epistemic value (expected information gain)
- `C` encodes prior preferences over outcomes

### 5. Precision-Weighted Prediction Errors

The belief update equations can be expressed as precision-weighted prediction errors:

```
μ̇ = κ_μ · ε_μ
```

Where:

- `ε_μ = ∂log p(o,s)/∂μ` is the prediction error
- `κ_μ = Σ` is the precision (inverse variance)

### 6. Hierarchical Models

For hierarchical generative models with levels i = 1, 2, ..., L:

```
p(o, s^(1:L)) = p(o|s^(1)) ∏_{i=1}^{L-1} p(s^(i)|s^(i+1)) p(s^(L))
```

Prediction errors flow up the hierarchy:

```
ε^(i) = μ^(i) - f^(i)(μ^(i+1))
```

While predictions flow down:

```
μ̇^(i) = -κ^(i) · ε^(i) + κ^(i-1) · ∂f^(i-1)/∂μ^(i) · ε^(i-1)
```

## Mathematical Properties

### 1. Convergence Guarantees

Under mild conditions (Lipschitz continuous gradients), gradient descent on F converges to local minima corresponding to approximate posterior modes.

### 2. Relationship to Other Frameworks

- **Maximum Likelihood**: When q(s) = δ(s - s\*), minimizing F reduces to maximum likelihood estimation
- **Variational Bayes**: F minimization implements variational Bayesian inference
- **Predictive Coding**: Hierarchical models with Gaussian assumptions yield predictive coding
- **Optimal Control**: Expected free energy minimization generalizes KL control

## Implementation Considerations

### 1. Numerical Stability

- Use log-space computations to avoid underflow
- Implement gradient clipping for stability
- Use adaptive learning rates (e.g., Adam optimizer)

### 2. Approximations

- **Laplace Approximation**: q(s) ≈ N(μ, Σ) with Σ^(-1) = -∂²F/∂s²|\_{s=μ}
- **Mean Field**: q(s) = ∏_i q_i(s_i) for factorized posteriors
- **Sampling**: Use particle filters or MCMC for complex posteriors

### 3. Computational Efficiency

- Cache repeated computations
- Use sparse representations for large state spaces
- Parallelize belief updates across independent factors

## Key References

1. Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.

2. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. Neural Computation, 29(1), 1-49.

3. Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active inference: the free energy principle in mind, brain, and behavior. MIT Press.

4. Buckley, C. L., Kim, C. S., McGregor, S., & Seth, A. K. (2017). The free energy principle for action and perception: A mathematical review. Journal of Mathematical Psychology, 81, 55-79.

5. Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: A synthesis. Journal of Mathematical Psychology, 99, 102447.

## Mathematical Notation Summary

| Symbol    | Description                                 |
| --------- | ------------------------------------------- |
| F         | Variational free energy                     |
| G         | Expected free energy                        |
| q(s)      | Approximate posterior (recognition density) |
| p(s,o)    | Generative model (joint distribution)       |
| p(o\|s)   | Observation model (likelihood)              |
| p(s\|s,a) | Transition model                            |
| π         | Policy (action sequence)                    |
| C         | Prior preferences                           |
| μ         | Posterior mean                              |
| Σ         | Posterior covariance                        |
| κ         | Precision (inverse variance)                |
| ε         | Prediction error                            |
| D_KL      | Kullback-Leibler divergence                 |
| H         | Entropy                                     |

## Conclusion

This mathematical framework provides the foundation for implementing Active Inference systems. The key insight is that perception, action, and learning can all be understood as processes that minimize variational free energy, providing a unified account of adaptive behavior.
