"""
Standardized numeric mock utilities for mathematical operations in tests.

This module provides mock factories that properly handle mathematical operations
while maintaining test isolation and predictability.
"""

from contextlib import contextmanager
from typing import Optional, Tuple, Union
from unittest.mock import MagicMock, Mock

import numpy as np
import torch


class NumericMock(MagicMock):
    """Mock that supports numeric operations."""

    def __init__(self,
                 value: Union[float,
                              int,
                              np.ndarray,
                              torch.Tensor] = 1.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._value = value
        self._setup_numeric_operations()

    def _setup_numeric_operations(self):
        """Setup all numeric operations to use the underlying value."""
        # Arithmetic operations
        self.__add__ = lambda other: NumericMock(
            self._value + self._get_value(other))
        self.__radd__ = lambda other: NumericMock(
            self._get_value(other) + self._value)
        self.__sub__ = lambda other: NumericMock(
            self._value - self._get_value(other))
        self.__rsub__ = lambda other: NumericMock(
            self._get_value(other) - self._value)
        self.__mul__ = lambda other: NumericMock(
            self._value * self._get_value(other))
        self.__rmul__ = lambda other: NumericMock(
            self._get_value(other) * self._value)
        self.__truediv__ = lambda other: NumericMock(
            self._value / self._get_value(other))
        self.__rtruediv__ = lambda other: NumericMock(
            self._get_value(other) / self._value)
        self.__pow__ = lambda other: NumericMock(
            self._value ** self._get_value(other))
        self.__rpow__ = lambda other: NumericMock(
            self._get_value(other) ** self._value)
        self.__neg__ = lambda: NumericMock(-self._value)

        # Comparison operations
        self.__lt__ = lambda other: self._value < self._get_value(other)
        self.__le__ = lambda other: self._value <= self._get_value(other)
        self.__gt__ = lambda other: self._value > self._get_value(other)
        self.__ge__ = lambda other: self._value >= self._get_value(other)
        self.__eq__ = lambda other: self._value == self._get_value(other)
        self.__ne__ = lambda other: self._value != self._get_value(other)

        # Array/tensor operations
        self.__getitem__ = lambda key: NumericMock(
            self._value[key] if hasattr(
                self._value, "__getitem__") else self._value)
        self.__len__ = lambda: len(
            self._value) if hasattr(
            self._value,
            "__len__") else 1

        # Common numpy/torch operations
        self.shape = self._get_shape()
        self.dtype = self._get_dtype()
        self.ndim = self._get_ndim()
        self.size = self._get_size
        self.sum = lambda axis=None, keepdims=False: NumericMock(
            np.sum(self._value, axis=axis, keepdims=keepdims)
        )
        self.mean = lambda axis=None, keepdims=False: NumericMock(
            np.mean(self._value, axis=axis, keepdims=keepdims)
        )
        self.max = lambda axis=None, keepdims=False: NumericMock(
            np.max(self._value, axis=axis, keepdims=keepdims)
        )
        self.min = lambda axis=None, keepdims=False: NumericMock(
            np.min(self._value, axis=axis, keepdims=keepdims)
        )
        self.reshape = lambda *shape: NumericMock(
            np.reshape(self._value, shape))
        self.transpose = lambda *axes: NumericMock(
            np.transpose(self._value, axes if axes else None)
        )
        self.squeeze = lambda axis=None: NumericMock(
            np.squeeze(self._value, axis=axis))
        self.unsqueeze = lambda dim: NumericMock(
            torch.unsqueeze(torch.tensor(self._value), dim)
            if isinstance(self._value, (int, float))
            else self._value
        )

        # Type conversion
        self.numpy = lambda: (
            self._value if isinstance(
                self._value,
                np.ndarray) else np.array(
                self._value))
        self.item = lambda: (
            float(
                self._value) if isinstance(
                self._value, (int, float)) else self._value.item())
        self.float = lambda: float(self._value)
        self.int = lambda: int(self._value)

    def _get_value(self, other):
        """Extract numeric value from other object."""
        if isinstance(other, NumericMock):
            return other._value
        elif isinstance(other, Mock):
            return 1.0  # Default value for unknown mocks
        return other

    def _get_shape(self):
        """Get shape of the underlying value."""
        if hasattr(self._value, "shape"):
            return self._value.shape
        elif isinstance(self._value, (list, tuple)):
            return (len(self._value),)
        return ()

    def _get_dtype(self):
        """Get dtype of the underlying value."""
        if hasattr(self._value, "dtype"):
            return self._value.dtype
        elif isinstance(self._value, float):
            return np.float32
        elif isinstance(self._value, int):
            return np.int32
        return np.float32

    def _get_ndim(self):
        """Get number of dimensions."""
        if hasattr(self._value, "ndim"):
            return self._value.ndim
        elif isinstance(self._value, (list, tuple)):
            return 1
        return 0

    def _get_size(self):
        """Get size method."""
        if hasattr(self._value, "size"):
            return self._value.size
        elif isinstance(self._value, (list, tuple)):
            return lambda: len(self._value)
        return lambda: 1


def create_tensor_mock(
    shape: Tuple[int, ...],
    value: Optional[Union[float, np.ndarray]] = None,
    requires_grad: bool = False,
    device: str = "cpu",
) -> NumericMock:
    """Create a mock that behaves like a PyTorch tensor."""
    if value is None:
        value = np.random.randn(*shape).astype(np.float32)
    elif isinstance(value, (int, float)):
        value = np.full(shape, value, dtype=np.float32)

    mock = NumericMock(value)
    mock.requires_grad = requires_grad
    mock.device = device
    mock.grad = None if not requires_grad else NumericMock(
        np.zeros_like(value))
    mock.detach = lambda: NumericMock(value)
    mock.to = lambda device: create_tensor_mock(
        shape, value, requires_grad, device)
    mock.cpu = lambda: create_tensor_mock(shape, value, requires_grad, "cpu")
    mock.cuda = lambda: create_tensor_mock(shape, value, requires_grad, "cuda")

    return mock


def create_array_mock(
    shape: Tuple[int, ...], value: Optional[Union[float, np.ndarray]] = None
) -> NumericMock:
    """Create a mock that behaves like a NumPy array."""
    if value is None:
        value = np.random.randn(*shape)
    elif isinstance(value, (int, float)):
        value = np.full(shape, value)

    return NumericMock(value)


def create_belief_state_mock(
        num_states: int = 4,
        normalized: bool = True) -> NumericMock:
    """Create a mock belief state for active inference tests."""
    beliefs = np.random.rand(num_states)
    if normalized:
        beliefs = beliefs / beliefs.sum()
    return NumericMock(beliefs)


def create_precision_matrix_mock(
    shape: Tuple[int, ...], positive_definite: bool = True
) -> NumericMock:
    """Create a mock precision (inverse covariance) matrix."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Precision matrix must be square")

    if positive_definite:
        # Create a positive definite matrix
        A = np.random.randn(*shape)
        value = A @ A.T + np.eye(shape[0]) * 0.1
    else:
        value = np.random.randn(*shape)

    return NumericMock(value)


def create_generative_model_mock(
    num_states: int = 4, num_observations: int = 4, num_actions: int = 4
) -> Mock:
    """Create a mock generative model for active inference."""
    mock = Mock()

    # A matrix (observation model)
    A = np.random.rand(num_observations, num_states)
    A = A / A.sum(axis=0, keepdims=True)  # Normalize columns
    mock.A = NumericMock(A)

    # B matrix (transition model)
    B = np.random.rand(num_states, num_states, num_actions)
    B = B / B.sum(axis=0, keepdims=True)  # Normalize columns
    mock.B = NumericMock(B)

    # C vector (preferences)
    C = np.random.rand(num_observations)
    mock.C = NumericMock(C)

    # D vector (initial state prior)
    D = np.random.rand(num_states)
    D = D / D.sum()  # Normalize
    mock.D = NumericMock(D)

    # Model dimensions
    mock.num_states = num_states
    mock.num_observations = num_observations
    mock.num_actions = num_actions

    return mock


def create_gnn_layer_mock(
    in_features: int = 64, out_features: int = 64, num_nodes: int = 10
) -> Mock:
    """Create a mock GNN layer."""
    mock = Mock()

    # Weight parameters
    mock.weight = NumericMock(np.random.randn(in_features, out_features))
    mock.bias = NumericMock(np.random.randn(out_features))

    # Forward method
    def forward(x, edge_index, edge_attr=None):
        # Simple mock forward pass
        batch_size = x.shape[0] if hasattr(x, "shape") else num_nodes
        return NumericMock(np.random.randn(batch_size, out_features))

    mock.forward = forward
    mock.in_features = in_features
    mock.out_features = out_features

    return mock


@contextmanager
def numeric_mock_context():
    """Context manager that temporarily replaces common numeric operations with mocked versions."""
    import numpy as np
    import torch

    # Store original functions
    original_torch_tensor = torch.tensor
    original_torch_zeros = torch.zeros
    original_torch_ones = torch.ones
    original_torch_randn = torch.randn
    original_np_array = np.array
    original_np_zeros = np.zeros
    original_np_ones = np.ones

    # Replace with mock-aware versions
    torch.tensor = lambda data, **kwargs: create_tensor_mock(
        data.shape if hasattr(data, "shape") else (1,), data
    )
    torch.zeros = lambda *shape, **kwargs: create_tensor_mock(shape, 0.0)
    torch.ones = lambda *shape, **kwargs: create_tensor_mock(shape, 1.0)
    torch.randn = lambda *shape, **kwargs: create_tensor_mock(shape)
    np.array = lambda data, **kwargs: create_array_mock(
        np.array(data).shape, data)
    np.zeros = lambda shape, **kwargs: create_array_mock(shape, 0.0)
    np.ones = lambda shape, **kwargs: create_array_mock(shape, 1.0)

    try:
        yield
    finally:
        # Restore original functions
        torch.tensor = original_torch_tensor
        torch.zeros = original_torch_zeros
        torch.ones = original_torch_ones
        torch.randn = original_torch_randn
        np.array = original_np_array
        np.zeros = original_np_zeros
        np.ones = original_np_ones


# Commonly used mock fixtures
def mock_active_inference_engine():
    """Create a mock active inference engine with numeric operations."""
    mock = Mock()
    mock.beliefs = create_belief_state_mock()
    mock.precision = NumericMock(1.0)
    mock.free_energy = NumericMock(0.0)
    mock.generative_model = create_generative_model_mock()

    def update_beliefs(observation):
        # Mock belief update
        mock.beliefs = create_belief_state_mock()
        return mock.beliefs

    mock.update_beliefs = update_beliefs
    mock.select_action = lambda: 0  # Default action
    mock.calculate_free_energy = lambda: NumericMock(np.random.rand())

    return mock


def mock_precision_optimizer():
    """Create a mock precision optimizer."""
    mock = Mock()
    mock.precision = NumericMock(1.0)
    mock.learning_rate = 0.01

    def optimize(errors, context=None):
        # Mock optimization
        return NumericMock(1.0 + np.random.randn() * 0.1)

    mock.optimize = optimize
    mock.estimate_volatility = lambda history: NumericMock(0.1)

    return mock
