"""
Comprehensive tests for PyMDP Generative Model

This test suite provides complete coverage of the PyMDP-compatible generative model
implementation, including matrix conversions, adapters, and factory functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock torch before importing the module under test to avoid PyTorch compatibility issues
mock_torch = MagicMock()
mock_torch.rand = MagicMock(return_value=MagicMock())
mock_torch.randn = MagicMock(return_value=MagicMock())
mock_torch.Tensor = MagicMock()
mock_torch.nn = MagicMock()

# Mock torch tensor methods
mock_tensor = MagicMock()
mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(3, 4)
mock_tensor.dim.return_value = 2
mock_tensor.unsqueeze.return_value.repeat.return_value = mock_tensor
mock_tensor.transpose.return_value.numpy.return_value = np.random.rand(4, 4)
mock_tensor.T.numpy.return_value = np.random.rand(4, 4)

mock_torch.rand.return_value = mock_tensor
mock_torch.randn.return_value = mock_tensor

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn

# Import the module under test after mocking
from inference.engine.pymdp_generative_model import (
    PyMDPGenerativeModel,
    PyMDPGenerativeModelAdapter,
    create_pymdp_generative_model,
    convert_torch_to_pymdp_matrices,
)

# Import dependencies
from inference.engine.generative_model import (
    DiscreteGenerativeModel,
    ModelDimensions,
    ModelParameters,
)


class TestPyMDPGenerativeModel:
    """Test the PyMDPGenerativeModel class"""

    @pytest.fixture
    def model_dimensions(self):
        """Create model dimensions for testing"""
        return ModelDimensions(
            num_states=4,
            num_observations=3,
            num_actions=2,
            time_horizon=5
        )

    @pytest.fixture
    def model_parameters(self):
        """Create model parameters for testing"""
        return ModelParameters(
            learning_rate=0.01,
            temperature=1.0,
            prior_strength=1.0
        )

    @pytest.fixture
    def pymdp_model(self, model_dimensions, model_parameters):
        """Create a PyMDPGenerativeModel instance"""
        return PyMDPGenerativeModel(model_dimensions, model_parameters)

    def test_initialization(self, pymdp_model, model_dimensions):
        """Test proper initialization of the model"""
        assert pymdp_model.dims == model_dimensions
        assert isinstance(pymdp_model.A, np.ndarray)
        assert isinstance(pymdp_model.B, np.ndarray)
        assert isinstance(pymdp_model.C, np.ndarray)
        assert isinstance(pymdp_model.D, np.ndarray)

    def test_A_matrix_shape_and_properties(self, pymdp_model, model_dimensions):
        """Test A matrix (observation model) shape and properties"""
        A = pymdp_model.A
        
        # Check shape
        expected_shape = (model_dimensions.num_observations, model_dimensions.num_states)
        assert A.shape == expected_shape
        
        # Check that each column sums to 1 (proper probability distribution)
        column_sums = A.sum(axis=0)
        np.testing.assert_allclose(column_sums, 1.0, rtol=1e-10)
        
        # Check that all values are non-negative
        assert np.all(A >= 0)

    def test_B_matrix_shape_and_properties(self, pymdp_model, model_dimensions):
        """Test B matrix (transition model) shape and properties"""
        B = pymdp_model.B
        
        # Check shape
        expected_shape = (
            model_dimensions.num_states,
            model_dimensions.num_states,
            model_dimensions.num_actions
        )
        assert B.shape == expected_shape
        
        # Check that each column for each action sums to 1
        for a in range(model_dimensions.num_actions):
            column_sums = B[:, :, a].sum(axis=0)
            np.testing.assert_allclose(column_sums, 1.0, rtol=1e-10)
        
        # Check that all values are non-negative
        assert np.all(B >= 0)

    def test_C_matrix_shape_and_properties(self, pymdp_model, model_dimensions):
        """Test C matrix (preferences) shape and properties"""
        C = pymdp_model.C
        
        # Check shape
        expected_shape = (model_dimensions.num_observations, model_dimensions.time_horizon)
        assert C.shape == expected_shape
        
        # Default initialization should be zeros
        np.testing.assert_array_equal(C, np.zeros(expected_shape))

    def test_D_matrix_shape_and_properties(self, pymdp_model, model_dimensions):
        """Test D matrix (initial prior) shape and properties"""
        D = pymdp_model.D
        
        # Check shape
        expected_shape = (model_dimensions.num_states,)
        assert D.shape == expected_shape
        
        # Check that it sums to 1 (proper probability distribution)
        assert np.isclose(D.sum(), 1.0)
        
        # Check that all values are non-negative
        assert np.all(D >= 0)
        
        # Default should be uniform
        expected_uniform = np.ones(model_dimensions.num_states) / model_dimensions.num_states
        np.testing.assert_allclose(D, expected_uniform)

    def test_set_A_matrix_numpy(self, pymdp_model, model_dimensions):
        """Test setting A matrix with numpy array"""
        # Create test A matrix
        new_A = np.random.rand(model_dimensions.num_observations, model_dimensions.num_states)
        
        pymdp_model.set_A_matrix(new_A)
        
        # Check that it was normalized
        column_sums = pymdp_model.A.sum(axis=0)
        np.testing.assert_allclose(column_sums, 1.0, rtol=1e-10)

    def test_set_A_matrix_torch(self, pymdp_model, model_dimensions):
        """Test setting A matrix with torch tensor"""
        # Create mock tensor
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(
            model_dimensions.num_observations, model_dimensions.num_states
        )
        
        pymdp_model.set_A_matrix(mock_tensor)
        
        # Check that it was converted and normalized
        assert isinstance(pymdp_model.A, np.ndarray)
        column_sums = pymdp_model.A.sum(axis=0)
        np.testing.assert_allclose(column_sums, 1.0, rtol=1e-10)

    def test_set_A_matrix_wrong_shape(self, pymdp_model):
        """Test setting A matrix with wrong shape raises assertion"""
        wrong_A = np.random.rand(5, 5)  # Wrong shape
        
        with pytest.raises(AssertionError):
            pymdp_model.set_A_matrix(wrong_A)

    def test_set_B_matrix_numpy(self, pymdp_model, model_dimensions):
        """Test setting B matrix with numpy array"""
        # Create test B matrix
        new_B = np.random.rand(
            model_dimensions.num_states,
            model_dimensions.num_states,
            model_dimensions.num_actions
        )
        
        pymdp_model.set_B_matrix(new_B)
        
        # Check that it was normalized for each action
        for a in range(model_dimensions.num_actions):
            column_sums = pymdp_model.B[:, :, a].sum(axis=0)
            np.testing.assert_allclose(column_sums, 1.0, rtol=1e-10)

    def test_set_B_matrix_torch(self, pymdp_model, model_dimensions):
        """Test setting B matrix with torch tensor"""
        # Create mock tensor
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(
            model_dimensions.num_states,
            model_dimensions.num_states,
            model_dimensions.num_actions
        )
        
        pymdp_model.set_B_matrix(mock_tensor)
        
        # Check that it was converted and normalized
        assert isinstance(pymdp_model.B, np.ndarray)
        for a in range(model_dimensions.num_actions):
            column_sums = pymdp_model.B[:, :, a].sum(axis=0)
            np.testing.assert_allclose(column_sums, 1.0, rtol=1e-10)

    def test_set_B_matrix_wrong_shape(self, pymdp_model):
        """Test setting B matrix with wrong shape raises assertion"""
        wrong_B = np.random.rand(3, 3, 3)  # Wrong shape
        
        with pytest.raises(AssertionError):
            pymdp_model.set_B_matrix(wrong_B)

    def test_set_C_matrix_numpy_2d(self, pymdp_model, model_dimensions):
        """Test setting C matrix with 2D numpy array"""
        # Create test C matrix
        new_C = np.random.randn(model_dimensions.num_observations, model_dimensions.time_horizon)
        
        pymdp_model.set_C_matrix(new_C)
        
        np.testing.assert_array_equal(pymdp_model.C, new_C)

    def test_set_C_matrix_numpy_1d(self, pymdp_model, model_dimensions):
        """Test setting C matrix with 1D numpy array (broadcasts to all timesteps)"""
        # Create test C matrix (1D)
        new_C_1d = np.random.randn(model_dimensions.num_observations)
        
        pymdp_model.set_C_matrix(new_C_1d)
        
        # Should be broadcast to all timesteps
        expected_C = np.tile(new_C_1d.reshape(-1, 1), (1, model_dimensions.time_horizon))
        np.testing.assert_array_equal(pymdp_model.C, expected_C)

    def test_set_C_matrix_torch(self, pymdp_model, model_dimensions):
        """Test setting C matrix with torch tensor"""
        # Create mock tensor
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(
            model_dimensions.num_observations, model_dimensions.time_horizon
        )
        
        pymdp_model.set_C_matrix(mock_tensor)
        
        # Check that it was converted
        assert isinstance(pymdp_model.C, np.ndarray)
        assert pymdp_model.C.shape == (model_dimensions.num_observations, model_dimensions.time_horizon)

    def test_set_C_matrix_wrong_shape(self, pymdp_model):
        """Test setting C matrix with wrong shape raises assertion"""
        wrong_C = np.random.rand(10, 10)  # Wrong shape
        
        with pytest.raises(AssertionError):
            pymdp_model.set_C_matrix(wrong_C)

    def test_set_D_matrix_numpy(self, pymdp_model, model_dimensions):
        """Test setting D matrix with numpy array"""
        # Create test D matrix
        new_D = np.random.rand(model_dimensions.num_states)
        
        pymdp_model.set_D_matrix(new_D)
        
        # Check that it was normalized
        assert np.isclose(pymdp_model.D.sum(), 1.0)

    def test_set_D_matrix_torch(self, pymdp_model, model_dimensions):
        """Test setting D matrix with torch tensor"""
        # Create mock tensor
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(
            model_dimensions.num_states
        )
        
        pymdp_model.set_D_matrix(mock_tensor)
        
        # Check that it was converted and normalized
        assert isinstance(pymdp_model.D, np.ndarray)
        assert np.isclose(pymdp_model.D.sum(), 1.0)

    def test_set_D_matrix_wrong_shape(self, pymdp_model):
        """Test setting D matrix with wrong shape raises assertion"""
        wrong_D = np.random.rand(10)  # Wrong shape
        
        with pytest.raises(AssertionError):
            pymdp_model.set_D_matrix(wrong_D)

    def test_get_pymdp_matrices(self, pymdp_model, model_dimensions):
        """Test getting all matrices in tuple format"""
        A, B, C, D = pymdp_model.get_pymdp_matrices()
        
        # Check that all matrices are returned
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert isinstance(C, np.ndarray)
        assert isinstance(D, np.ndarray)
        
        # Check shapes
        assert A.shape == (model_dimensions.num_observations, model_dimensions.num_states)
        assert B.shape == (model_dimensions.num_states, model_dimensions.num_states, model_dimensions.num_actions)
        assert C.shape == (model_dimensions.num_observations, model_dimensions.time_horizon)
        assert D.shape == (model_dimensions.num_states,)

    @patch('inference.engine.pymdp_generative_model.logger')
    def test_initialization_logging(self, mock_logger, model_dimensions, model_parameters):
        """Test that initialization logs the model parameters"""
        PyMDPGenerativeModel(model_dimensions, model_parameters)
        
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert str(model_dimensions.num_states) in log_message
        assert str(model_dimensions.num_observations) in log_message
        assert str(model_dimensions.num_actions) in log_message

    def test_from_discrete_model(self, model_dimensions, model_parameters):
        """Test creating PyMDPGenerativeModel from DiscreteGenerativeModel"""
        # Create a discrete model
        discrete_model = DiscreteGenerativeModel(model_dimensions, model_parameters)
        
        # Convert to PyMDP model
        pymdp_model = PyMDPGenerativeModel.from_discrete_model(discrete_model)
        
        # Check that it's properly initialized
        assert isinstance(pymdp_model, PyMDPGenerativeModel)
        assert pymdp_model.dims == model_dimensions
        
        # Check that matrices have correct shapes
        A, B, C, D = pymdp_model.get_pymdp_matrices()
        assert A.shape == (model_dimensions.num_observations, model_dimensions.num_states)
        assert B.shape == (model_dimensions.num_states, model_dimensions.num_states, model_dimensions.num_actions)
        assert C.shape == (model_dimensions.num_observations, model_dimensions.time_horizon)
        assert D.shape == (model_dimensions.num_states,)

    def test_from_discrete_model_1d_C_matrix(self, model_dimensions, model_parameters):
        """Test conversion with 1D C matrix from discrete model"""
        # Create a discrete model with 1D C matrix
        discrete_model = DiscreteGenerativeModel(model_dimensions, model_parameters)
        
        # Make C matrix 1D by mocking
        mock_tensor = Mock()
        mock_tensor.dim.return_value = 1
        mock_tensor.unsqueeze.return_value.repeat.return_value = mock_tensor
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(
            model_dimensions.num_observations, model_dimensions.time_horizon
        )
        discrete_model.C = mock_tensor
        
        # Convert to PyMDP model
        pymdp_model = PyMDPGenerativeModel.from_discrete_model(discrete_model)
        
        # Check that C was expanded to 2D
        assert pymdp_model.C.shape == (model_dimensions.num_observations, model_dimensions.time_horizon)


class TestPyMDPGenerativeModelAdapter:
    """Test the PyMDPGenerativeModelAdapter class"""

    @pytest.fixture
    def model_dimensions(self):
        """Create model dimensions for testing"""
        return ModelDimensions(
            num_states=4,
            num_observations=3,
            num_actions=2,
            time_horizon=5
        )

    @pytest.fixture
    def model_parameters(self):
        """Create model parameters for testing"""
        return ModelParameters(
            learning_rate=0.01,
            temperature=1.0,
            prior_strength=1.0
        )

    @pytest.fixture
    def discrete_model(self, model_dimensions, model_parameters):
        """Create a DiscreteGenerativeModel for testing"""
        return DiscreteGenerativeModel(model_dimensions, model_parameters)

    @pytest.fixture
    def pymdp_model(self, model_dimensions, model_parameters):
        """Create a PyMDPGenerativeModel for testing"""
        return PyMDPGenerativeModel(model_dimensions, model_parameters)

    def test_adapter_with_discrete_model(self, discrete_model):
        """Test adapter initialization with DiscreteGenerativeModel"""
        adapter = PyMDPGenerativeModelAdapter(discrete_model)
        
        assert adapter.base_model == discrete_model
        assert isinstance(adapter.pymdp_model, PyMDPGenerativeModel)

    def test_adapter_with_pymdp_model(self, pymdp_model):
        """Test adapter initialization with PyMDPGenerativeModel"""
        adapter = PyMDPGenerativeModelAdapter(pymdp_model)
        
        assert adapter.base_model == pymdp_model
        assert adapter.pymdp_model == pymdp_model

    def test_adapter_with_unsupported_model(self):
        """Test adapter initialization with unsupported model type"""
        unsupported_model = "not_a_model"
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            PyMDPGenerativeModelAdapter(unsupported_model)

    def test_observation_model_numpy(self, discrete_model, model_dimensions):
        """Test observation model computation with numpy input"""
        adapter = PyMDPGenerativeModelAdapter(discrete_model)
        
        # Create test states
        states = np.random.rand(model_dimensions.num_states)
        
        # Compute observations
        observations = adapter.observation_model(states)
        
        # Check result
        assert isinstance(observations, np.ndarray)
        assert observations.shape == (model_dimensions.num_observations,)

    def test_observation_model_torch(self, discrete_model, model_dimensions):
        """Test observation model computation with torch input"""
        adapter = PyMDPGenerativeModelAdapter(discrete_model)
        
        # Create mock tensor
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(
            model_dimensions.num_states
        )
        
        # Compute observations
        observations = adapter.observation_model(mock_tensor)
        
        # Check result
        assert isinstance(observations, np.ndarray)
        assert observations.shape == (model_dimensions.num_observations,)

    def test_transition_model_numpy(self, discrete_model, model_dimensions):
        """Test transition model computation with numpy input"""
        adapter = PyMDPGenerativeModelAdapter(discrete_model)
        
        # Create test states
        states = np.random.rand(model_dimensions.num_states)
        action = 0
        
        # Compute next states
        next_states = adapter.transition_model(states, action)
        
        # Check result
        assert isinstance(next_states, np.ndarray)
        assert next_states.shape == (model_dimensions.num_states,)

    def test_transition_model_torch(self, discrete_model, model_dimensions):
        """Test transition model computation with torch input"""
        adapter = PyMDPGenerativeModelAdapter(discrete_model)
        
        # Create mock tensor
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(
            model_dimensions.num_states
        )
        action = 1
        
        # Compute next states
        next_states = adapter.transition_model(mock_tensor, action)
        
        # Check result
        assert isinstance(next_states, np.ndarray)
        assert next_states.shape == (model_dimensions.num_states,)

    def test_get_preferences(self, discrete_model, model_dimensions):
        """Test getting preferences for specific timestep"""
        adapter = PyMDPGenerativeModelAdapter(discrete_model)
        
        # Get preferences for timestep 0
        prefs_0 = adapter.get_preferences(0)
        assert isinstance(prefs_0, np.ndarray)
        assert prefs_0.shape == (model_dimensions.num_observations,)
        
        # Get preferences for timestep 2
        prefs_2 = adapter.get_preferences(2)
        assert isinstance(prefs_2, np.ndarray)
        assert prefs_2.shape == (model_dimensions.num_observations,)

    def test_get_preferences_out_of_bounds(self, discrete_model, model_dimensions):
        """Test getting preferences for timestep beyond time horizon"""
        adapter = PyMDPGenerativeModelAdapter(discrete_model)
        
        # Get preferences for timestep beyond horizon (should use last timestep)
        prefs = adapter.get_preferences(100)
        assert isinstance(prefs, np.ndarray)
        assert prefs.shape == (model_dimensions.num_observations,)

    def test_get_initial_prior(self, discrete_model, model_dimensions):
        """Test getting initial state prior"""
        adapter = PyMDPGenerativeModelAdapter(discrete_model)
        
        prior = adapter.get_initial_prior()
        assert isinstance(prior, np.ndarray)
        assert prior.shape == (model_dimensions.num_states,)
        assert np.isclose(prior.sum(), 1.0)

    def test_get_pymdp_matrices(self, discrete_model, model_dimensions):
        """Test getting all PyMDP matrices from adapter"""
        adapter = PyMDPGenerativeModelAdapter(discrete_model)
        
        A, B, C, D = adapter.get_pymdp_matrices()
        
        # Check that all matrices have correct shapes
        assert A.shape == (model_dimensions.num_observations, model_dimensions.num_states)
        assert B.shape == (model_dimensions.num_states, model_dimensions.num_states, model_dimensions.num_actions)
        assert C.shape == (model_dimensions.num_observations, model_dimensions.time_horizon)
        assert D.shape == (model_dimensions.num_states,)


class TestFactoryFunction:
    """Test the create_pymdp_generative_model factory function"""

    def test_create_pymdp_generative_model_basic(self):
        """Test basic factory function usage"""
        model = create_pymdp_generative_model(
            num_states=3,
            num_observations=2,
            num_actions=2,
            time_horizon=4
        )
        
        assert isinstance(model, PyMDPGenerativeModel)
        assert model.dims.num_states == 3
        assert model.dims.num_observations == 2
        assert model.dims.num_actions == 2
        assert model.dims.time_horizon == 4

    def test_create_pymdp_generative_model_default_time_horizon(self):
        """Test factory function with default time horizon"""
        model = create_pymdp_generative_model(
            num_states=4,
            num_observations=3,
            num_actions=2
        )
        
        assert isinstance(model, PyMDPGenerativeModel)
        assert model.dims.time_horizon == 1

    def test_create_pymdp_generative_model_with_kwargs(self):
        """Test factory function with additional kwargs"""
        model = create_pymdp_generative_model(
            num_states=3,
            num_observations=2,
            num_actions=2,
            learning_rate=0.05,
            temperature=2.0
        )
        
        assert isinstance(model, PyMDPGenerativeModel)
        assert model.params.learning_rate == 0.05
        assert model.params.temperature == 2.0


class TestMatrixConversionFunction:
    """Test the convert_torch_to_pymdp_matrices function"""

    def test_convert_torch_to_pymdp_matrices_basic(self):
        """Test basic matrix conversion"""
        # Create mock tensors
        A_torch = Mock()
        A_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(3, 4)
        
        B_torch = Mock()
        B_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(4, 4, 2)
        B_torch.transpose.return_value.numpy.return_value = np.random.rand(4, 4, 2)
        
        C_torch = Mock()
        C_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(3)
        C_torch.ndim = 1
        
        D_torch = Mock()
        D_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(4)
        
        A, B, C, D = convert_torch_to_pymdp_matrices(A_torch, B_torch, C_torch, D_torch)
        
        # Check types
        assert isinstance(A, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert isinstance(C, np.ndarray)
        assert isinstance(D, np.ndarray)

    def test_convert_torch_to_pymdp_matrices_square_A(self):
        """Test matrix conversion with square A matrix"""
        # Create mock square A matrix
        A_torch = Mock()
        square_matrix = np.random.rand(4, 4)
        A_torch.detach.return_value.cpu.return_value.numpy.return_value = square_matrix
        A_torch.T.numpy.return_value = square_matrix.T
        A_torch.shape = [4, 4]
        
        B_torch = Mock()
        B_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(4, 4, 2)
        B_torch.transpose.return_value.numpy.return_value = np.random.rand(4, 4, 2)
        
        C_torch = Mock()
        C_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(4)
        C_torch.ndim = 1
        
        D_torch = Mock()
        D_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(4)
        
        A, B, C, D = convert_torch_to_pymdp_matrices(A_torch, B_torch, C_torch, D_torch)
        
        # Should be transposed for square matrices
        assert A.shape == (4, 4)

    def test_convert_torch_to_pymdp_matrices_2d_C(self):
        """Test matrix conversion with 2D C matrix"""
        A_torch = Mock()
        A_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(3, 4)
        A_torch.shape = [3, 4]
        
        B_torch = Mock()
        B_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(4, 4, 2)
        B_torch.transpose.return_value.numpy.return_value = np.random.rand(4, 4, 2)
        
        C_torch = Mock()
        C_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(3, 5)
        C_torch.ndim = 2
        
        D_torch = Mock()
        D_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(4)
        
        A, B, C, D = convert_torch_to_pymdp_matrices(A_torch, B_torch, C_torch, D_torch)
        
        # C should remain 2D
        assert C.shape == (3, 5)

    def test_convert_torch_to_pymdp_matrices_normalization(self):
        """Test that matrices are properly normalized"""
        A_torch = Mock()
        A_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(3, 4)
        A_torch.shape = [3, 4]
        
        B_torch = Mock()
        B_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(4, 4, 2)
        B_torch.transpose.return_value.numpy.return_value = np.random.rand(4, 4, 2)
        
        C_torch = Mock()
        C_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(3)
        C_torch.ndim = 1
        
        D_torch = Mock()
        D_torch.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(4)
        
        A, B, C, D = convert_torch_to_pymdp_matrices(A_torch, B_torch, C_torch, D_torch)
        
        # Check normalization
        # A columns should sum to 1
        np.testing.assert_allclose(A.sum(axis=0), 1.0, rtol=1e-10)
        
        # B columns for each action should sum to 1
        for a in range(2):
            np.testing.assert_allclose(B[:, :, a].sum(axis=0), 1.0, rtol=1e-10)
        
        # D should sum to 1
        np.testing.assert_allclose(D.sum(), 1.0, rtol=1e-10) 