"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .generative_model import DiscreteGenerativeModel, ModelDimensions, ModelParameters

"""\npymdp-Compatible Generative Model Implementation

This module provides generative models that are compatible with the official pymdp library,
converting existing custom implementations to use pymdp's expected matrix formats.
"""
logger = logging.getLogger(__name__)


class PyMDPGenerativeModel:
    """
    Generative model that provides pymdp-compatible A, B, C, D matrices.
    This class converts existing PyTorch-based generative models to numpy arrays
    in the format expected by pymdp.Agent.
    """

    def __init__(self, dimensions: ModelDimensions, parameters: ModelParameters) -> None:
        self.dims = dimensions
        self.params = parameters
        # Initialize pymdp-compatible matrices as numpy arrays
        self.A = self._initialize_A_matrix()
        self.B = self._initialize_B_matrix()
        self.C = self._initialize_C_matrix()
        self.D = self._initialize_D_matrix()
        logger.info(
            f"Initialized pymdp-compatible generative model with "
            f"{dimensions.num_states} states, {dimensions.num_observations} observations, "
            f"{dimensions.num_actions} actions"
        )

    def _initialize_A_matrix(self) -> np.ndarray:
        """
        Initialize observation model matrix A.
        Returns:
            A: numpy array of shape [num_observations, num_states]
                A[o, s] = p(o|s) - probability of observation o given state s
        """

        # Create random observation model
        A = np.random.rand(self.dims.num_observations, self.dims.num_states)
        # Normalize columns so each column sums to 1 (proper probability distribution)
        A = A / A.sum(axis=0, keepdims=True)
        return A

    def _initialize_B_matrix(self) -> np.ndarray:
        """
        Initialize transition model matrix B.
        Returns:
            B: numpy array of shape [num_states, num_states, num_actions]
                B[s', s, a] = p(s'|s, a) - probability of next state s' given current state s and action a
        """

        B = np.zeros((self.dims.num_states, self.dims.num_states, self.dims.num_actions))
        for a in range(self.dims.num_actions):
            # Create random transition matrix for this action
            B_a = np.random.rand(self.dims.num_states, self.dims.num_states)
            # Normalize columns so each column sums to 1
            B[:, :, a] = B_a / B_a.sum(axis=0, keepdims=True)
        return B

    def _initialize_C_matrix(self) -> np.ndarray:
        """

        Initialize preference matrix C.
        Returns:
            C: numpy array of shape [num_observations, time_horizon]
                C[o, t] = log preference for observation o at time t
        """
        # Initialize with neutral preferences (zeros in log space)
        C = np.zeros((self.dims.num_observations, self.dims.time_horizon))
        return C

    def _initialize_D_matrix(self) -> np.ndarray:
        """
        Initialize initial state prior D.
        Returns:
            D: numpy array of shape [num_states]
                D[s] = p(s) - prior probability of initial state s
        """
        # Uniform prior over states
        D = np.ones(self.dims.num_states) / self.dims.num_states
        return D

    def set_A_matrix(self, A: Union[np.ndarray, torch.Tensor]) -> None:
        """Set observation model matrix from external source"""
        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().numpy()
        assert A.shape == (
            self.dims.num_observations,
            self.dims.num_states,
        ), f"A matrix shape {A.shape} doesn't match expected {(
            self.dims.num_observations,
            self.dims.num_states
        )}"
        # Ensure proper normalization
        self.A = A / A.sum(axis=0, keepdims=True)

    def set_B_matrix(self, B: Union[np.ndarray, torch.Tensor]) -> None:
        """Set transition model matrix from external source"""
        if isinstance(B, torch.Tensor):
            B = B.detach().cpu().numpy()
        assert B.shape == (
            self.dims.num_states,
            self.dims.num_states,
            self.dims.num_actions,
        ), f"B matrix shape {B.shape} doesn't match expected {(
            self.dims.num_states,
            self.dims.num_states,
            self.dims.num_actions
        )}"
        # Ensure proper normalization for each action
        for a in range(self.dims.num_actions):
            self.B[:, :, a] = B[:, :, a] / B[:, :, a].sum(axis=0, keepdims=True)

    def set_C_matrix(self, C: Union[np.ndarray, torch.Tensor]) -> None:
        """Set preference matrix from external source"""
        if isinstance(C, torch.Tensor):
            C = C.detach().cpu().numpy()
        if C.ndim == 1:
            # Broadcast to all timesteps
            C = np.tile(C.reshape(-1, 1), (1, self.dims.time_horizon))
        assert C.shape == (
            self.dims.num_observations,
            self.dims.time_horizon,
        ), f"C matrix shape {C.shape} doesn't match expected {(
            self.dims.num_observations,
            self.dims.time_horizon
        )}"
        self.C = C

    def set_D_matrix(self, D: Union[np.ndarray, torch.Tensor]) -> None:
        """Set initial state prior from external source"""

        if isinstance(D, torch.Tensor):
            D = D.detach().cpu().numpy()
        assert D.shape == (
            self.dims.num_states,
        ), f"D matrix shape {D.shape} doesn't match expected {(self.dims.num_states, )}"
        # Ensure normalization
        self.D = D / D.sum()

    def get_pymdp_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all matrices in pymdp format.
        Returns:
            Tuple of (A, B, C, D) matrices ready for pymdp.Agent
        """
        return self.A, self.B, self.C, self.D

    @classmethod
    def from_discrete_model(cls, discrete_model: DiscreteGenerativeModel) -> "PyMDPGenerativeModel":
        """
        Create PyMDPGenerativeModel from existing DiscreteGenerativeModel.
        Args:
            discrete_model: Existing DiscreteGenerativeModel instance
        Returns:
            PyMDPGenerativeModel with converted matrices
        """
        pymdp_model = cls(discrete_model.dims, discrete_model.params)
        # Convert PyTorch tensors to numpy arrays in correct format
        # A matrix is already [obs, states] - correct for pymdp
        pymdp_model.set_A_matrix(discrete_model.A)
        # B matrix is [states, states, actions] but pymdp expects [s', s, a]
        # Current format: B[s_next, s_curr, action] - this is already correct for pymdp!
        pymdp_model.set_B_matrix(discrete_model.B)
        # C matrix might be 1D [obs] or 2D [obs, time]
        if discrete_model.C.dim() == 1:
            # Expand to [obs, time_horizon]
            C_expanded = discrete_model.C.unsqueeze(1).repeat(1, discrete_model.dims.time_horizon)
            pymdp_model.set_C_matrix(C_expanded)
        else:
            pymdp_model.set_C_matrix(discrete_model.C)
        # D matrix is [states] - correct for pymdp
        pymdp_model.set_D_matrix(discrete_model.D)
        return pymdp_model


class PyMDPGenerativeModelAdapter:
    """
    Adapter that wraps existing generative models to provide pymdp compatibility.
    This allows gradual migration from custom implementations to pymdp without
    breaking existing code.
    """

    def __init__(self, base_model: Union[DiscreteGenerativeModel, PyMDPGenerativeModel]) -> None:
        self.base_model = base_model
        if isinstance(base_model, DiscreteGenerativeModel):
            self.pymdp_model = PyMDPGenerativeModel.from_discrete_model(base_model)
        elif isinstance(base_model, PyMDPGenerativeModel):
            self.pymdp_model = base_model
        else:
            raise ValueError(f"Unsupported model type: {type(base_model)}")

    def observation_model(self, states: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Compute observations using pymdp format"""
        if isinstance(states, torch.Tensor):
            states = states.detach().cpu().numpy()
        # A @ states gives observation probabilities
        result: np.ndarray = self.pymdp_model.A @ states
        return result

    def transition_model(self, states: Union[np.ndarray, torch.Tensor], action: int) -> np.ndarray:
        """Compute state transitions using pymdp format"""
        if isinstance(states, torch.Tensor):
            states = states.detach().cpu().numpy()
        # B[:, :, action] @ states gives next state probabilities
        result: np.ndarray = self.pymdp_model.B[:, :, action] @ states
        return result

    def get_preferences(self, timestep: int = 0) -> np.ndarray:
        """Get preferences for given timestep"""
        if timestep >= self.pymdp_model.C.shape[1]:
            timestep = -1  # Use last timestep
        return self.pymdp_model.C[:, timestep]

    def get_initial_prior(self) -> np.ndarray:
        """Get initial state prior"""
        return self.pymdp_model.D

    def get_pymdp_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get matrices for pymdp.Agent creation"""
        return self.pymdp_model.get_pymdp_matrices()


def create_pymdp_generative_model(
    num_states: int,
    num_observations: int,
    num_actions: int,
    time_horizon: int = 1,
    **kwargs: Any,
) -> PyMDPGenerativeModel:
    """
    Factory function to create pymdp-compatible generative model.
    Args:
        num_states: Number of hidden states
        num_observations: Number of possible observations
        num_actions: Number of possible actions
        time_horizon: Planning horizon
        **kwargs: Additional parameters
    Returns:
        PyMDPGenerativeModel instance
    """
    dimensions = ModelDimensions(
        num_states=num_states,
        num_observations=num_observations,
        num_actions=num_actions,
        time_horizon=time_horizon,
    )
    parameters = ModelParameters(**kwargs)
    return PyMDPGenerativeModel(dimensions, parameters)


def convert_torch_to_pymdp_matrices(
    A_torch: torch.Tensor,
    B_torch: torch.Tensor,
    C_torch: torch.Tensor,
    D_torch: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert PyTorch tensors to pymdp-compatible numpy arrays.
    Args:
        A_torch: Observation model [obs, states] or [states, obs]
        B_torch: Transition model [states, states, actions] or other permutation
        C_torch: Preferences [obs] or [obs, time]
        D_torch: Initial prior [states]
    Returns:
        Tuple of (A, B, C, D) in pymdp format
    """
    # Convert to numpy
    A = A_torch.detach().cpu().numpy()
    B = B_torch.detach().cpu().numpy()
    C = C_torch.detach().cpu().numpy()
    D = D_torch.detach().cpu().numpy()
    # Ensure correct shapes for pymdp
    if A.shape[0] != A.shape[1]:  # Not square, assume [obs, states]
        pass  # Already correct
    else:
        # If square, we need to determine orientation - assume [states, obs] and transpose
        A = A.T
    # B should be [s', s, a] - if it's [s, s', a], transpose first two dimensions
    if B.ndim == 3:
        # Assume current format is [s, s', a] and convert to [s', s, a]
        B = B.transpose(1, 0, 2)
    # C should be [obs, time] - if 1D, expand to 2D
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    # Normalize matrices
    A = A / A.sum(axis=0, keepdims=True)
    for a in range(B.shape[2]):
        B[:, :, a] = B[:, :, a] / B[:, :, a].sum(axis=0, keepdims=True)
    D = D / D.sum()
    return A, B, C, D
