"""
Test fixtures for Active Inference Engine testing.
"""

from typing import Dict, Optional

import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    # Handle PyTorch import errors or runtime errors
    TORCH_AVAILABLE = False
    torch = None
    print(f"Warning: PyTorch not available for fixtures: {e}")

try:
    from inference.engine.active_inference import InferenceConfig, VariationalMessagePassing
    from inference.engine.generative_model import GenerativeModel, ModelDimensions, ModelParameters

    INFERENCE_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    # Handle inference engine import errors
    INFERENCE_AVAILABLE = False
    InferenceConfig = None
    VariationalMessagePassing = None
    GenerativeModel = None
    ModelDimensions = None
    ModelParameters = None
    print(f"Warning: Inference engine not available for fixtures: {e}")


@pytest.fixture
def inference_config():
    """Basic inference configuration for testing."""
    if not TORCH_AVAILABLE or not INFERENCE_AVAILABLE:
        pytest.skip("PyTorch or Inference engine not available")

    return InferenceConfig(
        algorithm="variational_message_passing",
        num_iterations=10,
        convergence_threshold=1e-4,
        learning_rate=0.1,
        gradient_clip=1.0,
        use_natural_gradient=True,
        damping_factor=0.1,
        momentum=0.9,
        use_gpu=False,  # CPU for consistent testing
        dtype=torch.float32,
    )


@pytest.fixture
def model_dimensions():
    """Standard model dimensions for testing."""
    if not INFERENCE_AVAILABLE:
        pytest.skip("Inference engine not available")
    return ModelDimensions(
        num_states=4,
        num_observations=3,
        num_actions=2,
        time_horizon=5)


@pytest.fixture
def model_parameters():
    """Standard model parameters for testing."""
    if not TORCH_AVAILABLE or not INFERENCE_AVAILABLE:
        pytest.skip("PyTorch or Inference engine not available")
    return ModelParameters(
        learning_rate=0.01,
        precision_init=1.0,
        use_sparse=False,
        use_gpu=False,  # CPU for consistent testing
        dtype=torch.float32,
        eps=1e-8,
        temperature=1.0,
    )


@pytest.fixture
def simple_generative_model(model_dimensions, model_parameters):
    """Create a simple generative model for testing."""
    if not TORCH_AVAILABLE or not INFERENCE_AVAILABLE:
        pytest.skip("PyTorch or Inference engine not available")

    class SimpleGenerativeModel(GenerativeModel):
        def __init__(self, dims: ModelDimensions, params: ModelParameters):
            super().__init__(dims, params)
            self.dims = dims

            # Observation model: p(o|s)
            self.A = torch.tensor(
                [
                    [0.8, 0.1, 0.05, 0.05],  # obs 0
                    [0.1, 0.8, 0.05, 0.05],  # obs 1
                    [0.1, 0.1, 0.4, 0.4],  # obs 2
                ],
                dtype=torch.float32,
            )

            # Transition model: p(s'|s,a)
            self.B = torch.zeros(
                dims.num_actions,
                dims.num_states,
                dims.num_states)
            # Action 0: stay in place
            self.B[0] = torch.eye(dims.num_states)
            # Action 1: shift states
            self.B[1] = torch.roll(
                torch.eye(
                    dims.num_states),
                shifts=1,
                dims=0)

            # Prior preferences (goals)
            self.C = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

            # Initial state prior
            self.D = torch.ones(dims.num_states) / dims.num_states

        def sample_observation(self, state: torch.Tensor) -> torch.Tensor:
            """Sample observation given state."""
            if state.dim() == 0:
                state_idx = state.item()
            else:
                state_idx = torch.argmax(state).item()
            obs_probs = self.A[:, state_idx]
            return torch.multinomial(obs_probs, 1).squeeze()

        def compute_expected_observations(
                self, beliefs: torch.Tensor) -> torch.Tensor:
            """Compute expected observations given beliefs."""
            return torch.matmul(self.A, beliefs)

        def compute_expected_free_energy(
            self, beliefs: torch.Tensor, policies: torch.Tensor
        ) -> torch.Tensor:
            """Compute expected free energy for policies."""
            num_policies = policies.shape[0]
            efe = torch.zeros(num_policies)

            for p in range(num_policies):
                policies[p]
                expected_obs = self.compute_expected_observations(beliefs)

                # Epistemic value (information gain)
                entropy = -torch.sum(expected_obs *
                                     torch.log(expected_obs + 1e-8))

                # Pragmatic value (preference satisfaction)
                preference = torch.dot(expected_obs, self.C)

                efe[p] = entropy - preference

            return efe

        def observation_model(self, states: torch.Tensor) -> torch.Tensor:
            """Compute p(o|s)"""
            if states.dim() == 0:
                return self.A[:, states.item()]
            elif states.dim() == 1 and len(states) == self.dims.num_states:
                # states is a distribution
                return torch.matmul(self.A, states)
            else:
                # batch of state indices
                return self.A[:, states]

        def transition_model(
                self,
                states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
            """Compute p(s'|s, a)"""
            if actions.dim() == 0:
                action = actions.item()
            else:
                action = actions[0].item() if len(actions) > 0 else 0

            if states.dim() == 1:
                # states is a distribution
                return torch.matmul(self.B[action], states)
            else:
                # states is indices
                return self.B[action, :, states]

        def get_preferences(
                self,
                timestep: Optional[int] = None) -> torch.Tensor:
            """Get prior preferences p(o|C)"""
            return self.C

        def get_initial_prior(self) -> torch.Tensor:
            """Get initial state prior p(s)"""
            return self.D

    return SimpleGenerativeModel(model_dimensions, model_parameters)


@pytest.fixture
def continuous_generative_model(model_parameters):
    """Create a continuous state-space generative model."""

    class ContinuousGenerativeModel(GenerativeModel):
        def __init__(self, params: ModelParameters):
            dims = ModelDimensions(
                num_states=2,  # 2D continuous state
                num_observations=2,  # 2D continuous observation
                num_actions=2,
                time_horizon=10,
            )
            super().__init__(dims, params)

            # Linear observation model
            self.A = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            self.obs_noise = 0.1

            # Linear dynamics
            self.B = torch.tensor(
                [[0.95, 0.0], [0.0, 0.95]], dtype=torch.float32)
            self.process_noise = 0.05

            # Goal state
            self.goal = torch.tensor([1.0, 1.0], dtype=torch.float32)

        def sample_observation(self, state: torch.Tensor) -> torch.Tensor:
            """Sample observation with Gaussian noise."""
            obs = torch.matmul(self.A, state)
            noise = torch.randn_like(obs) * self.obs_noise
            return obs + noise

        def compute_expected_observations(
                self, state: torch.Tensor) -> torch.Tensor:
            """Compute expected observations."""
            return torch.matmul(self.A, state)

        def compute_expected_free_energy(
            self, state: torch.Tensor, action: torch.Tensor
        ) -> torch.Tensor:
            """Compute expected free energy for continuous case."""
            # Predict next state
            next_state = torch.matmul(self.B, state) + action

            # Expected observation
            expected_obs = self.compute_expected_observations(next_state)

            # Uncertainty (trace of covariance)
            uncertainty = self.obs_noise**2 * state.shape[0]

            # Goal alignment
            goal_distance = torch.norm(expected_obs - self.goal)

            return uncertainty + goal_distance

        def observation_model(self, states: torch.Tensor) -> torch.Tensor:
            """Compute p(o|s)"""
            return self.compute_expected_observations(states)

        def transition_model(
                self,
                states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
            """Compute p(s'|s, a)"""
            return torch.matmul(self.B, states) + actions

        def get_preferences(
                self,
                timestep: Optional[int] = None) -> torch.Tensor:
            """Get prior preferences p(o|C)"""
            return self.goal

        def get_initial_prior(self) -> torch.Tensor:
            """Get initial state prior p(s)"""
            return torch.zeros(2)  # Start at origin

    return ContinuousGenerativeModel(model_parameters)


@pytest.fixture
def hierarchical_generative_model(model_parameters):
    """Create a hierarchical generative model with multiple levels."""

    class HierarchicalGenerativeModel(GenerativeModel):
        def __init__(self, params: ModelParameters):
            # Define dimensions for each level
            self.levels = [
                ModelDimensions(
                    num_states=8,
                    num_observations=4,
                    num_actions=3,
                    time_horizon=5),
                ModelDimensions(
                    num_states=4,
                    num_observations=8,
                    num_actions=2,
                    time_horizon=10),
                ModelDimensions(
                    num_states=2,
                    num_observations=4,
                    num_actions=1,
                    time_horizon=20),
            ]
            super().__init__(self.levels[0], params)

            # Initialize models for each level
            self.level_models = []
            for dims in self.levels:
                level_model = {
                    "A": torch.rand(
                        dims.num_observations, dims.num_states), "B": torch.rand(
                        dims.num_actions, dims.num_states, dims.num_states), "C": torch.rand(
                        dims.num_observations), "D": torch.ones(
                        dims.num_states) / dims.num_states, }
                # Normalize probability distributions
                level_model["A"] = level_model["A"] / \
                    level_model["A"].sum(dim=0, keepdim=True)
                for a in range(dims.num_actions):
                    level_model["B"][a] = level_model["B"][a] / \
                        level_model["B"][a].sum(dim=0, keepdim=True)
                self.level_models.append(level_model)

        def get_level_model(self, level: int) -> Dict[str, torch.Tensor]:
            """Get model parameters for a specific level."""
            return self.level_models[level]

        def compute_top_down_prediction(
            self, level: int, higher_state: torch.Tensor
        ) -> torch.Tensor:
            """Compute top-down prediction from higher level."""
            if level == 0 or level >= len(self.levels):
                return torch.zeros(self.levels[level].num_states)

            # Simple linear mapping for demonstration
            higher_dim = self.levels[level - 1].num_states
            lower_dim = self.levels[level].num_states
            mapping = torch.rand(lower_dim, higher_dim)
            mapping = mapping / mapping.sum(dim=1, keepdim=True)

            return torch.matmul(mapping, higher_state)

        def observation_model(self, states: torch.Tensor) -> torch.Tensor:
            """Compute p(o|s) for first level"""
            level_0 = self.level_models[0]
            if states.dim() == 1:
                return torch.matmul(level_0["A"], states)
            else:
                return level_0["A"][:, states]

        def transition_model(
                self,
                states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
            """Compute p(s'|s, a) for first level"""
            level_0 = self.level_models[0]
            if actions.dim() == 0:
                action = actions.item()
            else:
                action = actions[0].item() if len(actions) > 0 else 0

            if states.dim() == 1:
                return torch.matmul(level_0["B"][action], states)
            else:
                return level_0["B"][action, :, states]

        def get_preferences(
                self,
                timestep: Optional[int] = None) -> torch.Tensor:
            """Get prior preferences p(o|C) for first level"""
            return self.level_models[0]["C"]

        def get_initial_prior(self) -> torch.Tensor:
            """Get initial state prior p(s) for first level"""
            return self.level_models[0]["D"]

    return HierarchicalGenerativeModel(model_parameters)


@pytest.fixture
def sample_observations():
    """Generate sample observations for testing."""
    return {
        "discrete": torch.tensor([0, 1, 2, 1, 0], dtype=torch.long),
        "continuous": torch.randn(5, 3),
        "binary": torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32),
        "multi_modal": {
            "visual": torch.randn(5, 64),
            "auditory": torch.randn(5, 32),
            "tactile": torch.randn(5, 16),
        },
    }


@pytest.fixture
def sample_beliefs():
    """Generate sample belief states for testing."""
    beliefs = {
        "uniform": torch.ones(4) / 4,
        "peaked": torch.tensor([0.7, 0.2, 0.05, 0.05]),
        "bimodal": torch.tensor([0.45, 0.05, 0.45, 0.05]),
        "sequence": torch.stack(
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0, 0.0]),
                torch.tensor([0.0, 0.0, 1.0, 0.0]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            ]
        ),
    }
    # Ensure all beliefs are normalized
    for key in beliefs:
        if beliefs[key].dim() == 1:
            beliefs[key] = beliefs[key] / beliefs[key].sum()
        else:
            beliefs[key] = beliefs[key] / \
                beliefs[key].sum(dim=-1, keepdim=True)
    return beliefs


@pytest.fixture
def sample_policies():
    """Generate sample policies for testing."""
    return {
        "greedy": torch.tensor([[0], [0], [0], [0], [0]]),  # Always action 0
        "exploratory": torch.tensor([[0], [1], [0], [1], [0]]),  # Alternate
        "random": torch.randint(0, 2, (5, 1)),
        "multi_step": torch.tensor(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        ),  # Multi-step policies
    }


@pytest.fixture
def expected_results():
    """Expected results for validation."""
    return {
        "free_energy_bounds": (-10.0, 10.0),
        "belief_sum": 1.0,
        "convergence_iterations": 10,
        "kl_divergence_threshold": 0.01,
        "entropy_bounds": (0.0, 2.0),
    }
