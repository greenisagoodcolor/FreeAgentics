"""Mock PyMDP for CI environments without full ML dependencies."""

import numpy as np
from unittest.mock import MagicMock


class MockAgent:
    """Mock PyMDP Agent for testing."""

    def __init__(self, *args, **kwargs):
        self.A = MagicMock()  # Observation model
        self.B = MagicMock()  # Transition model
        self.C = MagicMock()  # Preference model
        self.D = MagicMock()  # Prior beliefs
        self.qs = MagicMock()  # Posterior beliefs
        self.q_pi = MagicMock()  # Policy beliefs

    def infer_states(self, *args, **kwargs):
        """Mock state inference."""
        return MagicMock()

    def infer_policies(self, *args, **kwargs):
        """Mock policy inference."""
        return MagicMock()

    def sample_action(self, *args, **kwargs):
        """Mock action sampling."""
        return np.array([0])


class MockUtils:
    """Mock PyMDP utilities."""

    @staticmethod
    def random_A_matrix(*args, **kwargs):
        """Mock random A matrix generation."""
        return np.random.rand(2, 2)

    @staticmethod
    def random_B_matrix(*args, **kwargs):
        """Mock random B matrix generation."""
        return np.random.rand(2, 2, 2)

    @staticmethod
    def onehot(*args, **kwargs):
        """Mock one-hot encoding."""
        return np.array([1, 0])

    @staticmethod
    def softmax(*args, **kwargs):
        """Mock softmax function."""
        x = args[0] if args else np.array([1, 1])
        return np.exp(x) / np.sum(np.exp(x))


# Create module-level exports to match pymdp structure
Agent = MockAgent
utils = MockUtils()

# Add module-level attributes
agent = MockAgent
maths = MagicMock()
core = MagicMock()


def __getattr__(name):
    """Mock any missing attributes."""
    return MagicMock()
