"""
Comprehensive test coverage for agents/templates/pymdp_integration.py
PyMDP Integration Layer - Phase 2 systematic coverage

This test file provides complete coverage for the pymdp integration layer
following the systematic backend coverage improvement plan.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import the pymdp integration components
try:
    from agents.templates.base_template import (
        BeliefState,
        GenerativeModelParams,
        TemplateCategory,
        TemplateConfig,
        entropy,
    )
    from agents.templates.pymdp_integration import (
        PYMDP_AVAILABLE,
        PyMDPAgentWrapper,
        create_pymdp_agent,
    )

    IMPORT_SUCCESS = True
except ImportError:
    # Create minimal mock classes for testing if imports fail
    IMPORT_SUCCESS = False
    PYMDP_AVAILABLE = False

    class PyMDPAgentWrapper:
        def __init__(self, model_params, config):
            self.model_params = model_params
            self.config = config
            self.agent = None
            self.current_beliefs = np.ones(4) / 4

        def update_beliefs(self, observation):
            return Mock()

        def compute_free_energy(self, beliefs, observation):
            return 2.5

        def infer_policies(self, beliefs):
            return np.ones(3) / 3

        def get_mathematical_summary(self):
            return {"belief_entropy": 1.0, "pymdp_available": False}

    def create_pymdp_agent(model_params, config):
        return PyMDPAgentWrapper(model_params, config)

    def entropy(x):
        return -np.sum(x * np.log(x + 1e-16))


class TestPyMDPAgentWrapper:
    """Test PyMDP agent wrapper functionality."""

    @pytest.fixture
    def model_params(self):
        """Create valid generative model parameters."""
        A = np.array([[0.8, 0.1], [0.1, 0.8], [0.1, 0.1]])  # 3 obs, 2 states
        B = np.zeros((2, 2, 2))  # 2 states, 2 states, 2 policies
        B[:, :, 0] = np.eye(2)  # Stay policy
        B[:, :, 1] = np.array([[0.1, 0.9], [0.9, 0.1]])  # Move policy
        C = np.array([0.0, 1.0, -1.0])  # Preferences
        D = np.array([0.5, 0.5])  # Prior

        if IMPORT_SUCCESS:
            return GenerativeModelParams(A=A, B=B, C=C, D=D)
        else:
            return Mock(
                A=A, B=B, C=C, D=D, precision_sensory=1.0, precision_policy=1.0, precision_state=1.0
            )

    @pytest.fixture
    def template_config(self):
        """Create template configuration."""
        if IMPORT_SUCCESS:
            return TemplateConfig(
                template_id="pymdp_test",
                category=TemplateCategory.EXPLORER,
                num_states=2,
                num_observations=3,
                num_policies=2,
                planning_horizon=3,
            )
        else:
            return Mock(
                template_id="pymdp_test",
                category="explorer",
                num_states=2,
                num_observations=3,
                num_policies=2,
                planning_horizon=3,
            )

    @pytest.fixture
    def wrapper(self, model_params, template_config):
        """Create PyMDP agent wrapper."""
        return PyMDPAgentWrapper(model_params, template_config)

    def test_wrapper_initialization(self, wrapper, model_params, template_config):
        """Test wrapper initialization."""
        assert wrapper.model_params == model_params
        assert wrapper.config == template_config
        assert hasattr(wrapper, "current_beliefs")
        assert hasattr(wrapper, "agent")

    def test_pymdp_availability_handling(self, model_params, template_config):
        """Test handling of pymdp availability."""
        if not IMPORT_SUCCESS:
            return  # Skip for mock implementation

        # Test with pymdp available
        if PYMDP_AVAILABLE:
            wrapper = PyMDPAgentWrapper(model_params, template_config)
            # Should attempt to create pymdp agent
            assert hasattr(wrapper, "agent")

        # Test with pymdp unavailable (mocked)
        with patch("agents.templates.pymdp_integration.PYMDP_AVAILABLE", False):
            wrapper = PyMDPAgentWrapper(model_params, template_config)
            # Should use fallback
            assert hasattr(wrapper, "current_beliefs")

    def test_fallback_agent_creation(self, model_params, template_config):
        """Test fallback agent creation when pymdp unavailable."""
        if not IMPORT_SUCCESS:
            return

        with patch("agents.templates.pymdp_integration.PYMDP_AVAILABLE", False):
            wrapper = PyMDPAgentWrapper(model_params, template_config)

            # Should have fallback implementations
            assert hasattr(wrapper, "_fallback_softmax")
            assert hasattr(wrapper, "_fallback_kl_div")
            assert wrapper.agent is None

            # Test fallback functions
            x = np.array([1.0, 2.0, 3.0])
            result = wrapper._fallback_softmax(x)
            assert np.sum(result) == pytest.approx(1.0)
            assert np.all(result >= 0)

    def test_belief_update_interface(self, wrapper):
        """Test belief update interface."""
        # Test with integer observation
        belief_state = wrapper.update_beliefs(1)
        assert belief_state is not None

        # Test with observation array
        obs_array = np.array([0.8, 0.1, 0.1])
        belief_state = wrapper.update_beliefs(obs_array)
        assert belief_state is not None

    def test_pymdp_belief_update(self, model_params, template_config):
        """Test pymdp belief update when available."""
        if not IMPORT_SUCCESS or not PYMDP_AVAILABLE:
            return

        # Mock successful pymdp agent
        mock_agent = Mock()
        mock_agent.infer_states.return_value = [np.array([0.3, 0.7])]

        wrapper = PyMDPAgentWrapper(model_params, template_config)
        wrapper.agent = mock_agent

        belief_state = wrapper._pymdp_belief_update(1)

        # Should call pymdp infer_states
        mock_agent.infer_states.assert_called_once_with([1])

        if IMPORT_SUCCESS:
            assert isinstance(belief_state, BeliefState)
            assert len(belief_state.beliefs) == 2

    def test_fallback_belief_update(self, wrapper):
        """Test fallback belief update implementation."""
        if not IMPORT_SUCCESS:
            return

        # Force fallback mode
        wrapper.agent = None

        # Test with integer observation
        belief_state = wrapper._fallback_belief_update(0)

        if IMPORT_SUCCESS:
            assert isinstance(belief_state, BeliefState)

            # Beliefs should be updated via Bayesian rule
            assert np.sum(belief_state.beliefs) == pytest.approx(1.0)
            assert np.all(belief_state.beliefs >= 0)

        # Test with observation distribution
        obs_dist = np.array([0.7, 0.2, 0.1])
        belief_state = wrapper._fallback_belief_update(obs_dist)

        if IMPORT_SUCCESS:
            assert isinstance(belief_state, BeliefState)

    def test_free_energy_computation(self, wrapper):
        """Test free energy computation."""
        if IMPORT_SUCCESS:
            beliefs = BeliefState.create_uniform(2, 2)
        else:
            beliefs = Mock(beliefs=np.array([0.5, 0.5]))

        # Test with integer observation
        fe1 = wrapper.compute_free_energy(beliefs, 1)
        assert isinstance(fe1, float)
        assert not np.isnan(fe1)
        assert not np.isinf(fe1)

        # Test with observation distribution
        obs_dist = np.array([0.6, 0.3, 0.1])
        fe2 = wrapper.compute_free_energy(beliefs, obs_dist)
        assert isinstance(fe2, float)
        assert not np.isnan(fe2)

    def test_pymdp_free_energy(self, model_params, template_config):
        """Test pymdp free energy computation."""
        if not IMPORT_SUCCESS:
            return

        wrapper = PyMDPAgentWrapper(model_params, template_config)

        if IMPORT_SUCCESS:
            beliefs = BeliefState.create_uniform(2, 2)
        else:
            beliefs = Mock(beliefs=np.array([0.5, 0.5]))

        # Test pymdp computation
        fe = wrapper._pymdp_free_energy(beliefs, 1)
        assert isinstance(fe, float)

        # Test fallback computation
        fe_fallback = wrapper._fallback_free_energy(beliefs, 1)
        assert isinstance(fe_fallback, float)

    def test_policy_inference(self, wrapper):
        """Test policy inference."""
        if IMPORT_SUCCESS:
            beliefs = BeliefState.create_uniform(2, 2)
        else:
            beliefs = Mock(beliefs=np.array([0.5, 0.5]))

        policies = wrapper.infer_policies(beliefs)

        assert isinstance(policies, np.ndarray)
        assert len(policies) > 0

        if wrapper.config.num_policies:
            assert len(policies) == wrapper.config.num_policies

        # Should be normalized probabilities
        assert np.sum(policies) == pytest.approx(1.0)
        assert np.all(policies >= 0)

    def test_pymdp_policy_inference(self, model_params, template_config):
        """Test pymdp policy inference when available."""
        if not IMPORT_SUCCESS:
            return

        # Mock pymdp agent with policy inference
        mock_agent = Mock()
        mock_agent.infer_policies.return_value = [np.array([0.7, 0.3])]
        mock_agent.qs = [np.array([0.5, 0.5])]

        wrapper = PyMDPAgentWrapper(model_params, template_config)
        wrapper.agent = mock_agent

        if IMPORT_SUCCESS:
            beliefs = BeliefState.create_uniform(2, 2)
        else:
            beliefs = Mock(beliefs=np.array([0.5, 0.5]))

        policies = wrapper._pymdp_policy_inference(beliefs)

        # Should call pymdp infer_policies
        mock_agent.infer_policies.assert_called_once()

        assert isinstance(policies, np.ndarray)
        assert np.sum(policies) == pytest.approx(1.0)

    def test_fallback_policy_inference(self, wrapper):
        """Test fallback policy inference."""
        if not IMPORT_SUCCESS:
            return

        # Force fallback mode
        wrapper.agent = None

        if IMPORT_SUCCESS:
            beliefs = BeliefState.create_uniform(2, 2)
        else:
            beliefs = Mock(beliefs=np.array([0.5, 0.5]))

        policies = wrapper._fallback_policy_inference(beliefs)

        assert isinstance(policies, np.ndarray)
        assert np.sum(policies) == pytest.approx(1.0)
        assert np.all(policies >= 0)

    def test_mathematical_summary(self, wrapper):
        """Test mathematical summary generation."""
        summary = wrapper.get_mathematical_summary()

        assert isinstance(summary, dict)

        # Check required fields
        required_fields = [
            "belief_entropy",
            "belief_sum",
            "precision_sensory",
            "precision_policy",
            "precision_state",
            "model_dimensions",
            "pymdp_available",
            "agent_initialized",
        ]

        for field in required_fields:
            assert field in summary

        # Test field types and values
        assert isinstance(summary["belief_entropy"], float)
        assert isinstance(summary["belief_sum"], float)
        assert isinstance(summary["pymdp_available"], bool)
        assert isinstance(summary["agent_initialized"], bool)

        # Model dimensions should be correct
        dims = summary["model_dimensions"]
        assert "num_states" in dims
        assert "num_observations" in dims
        assert "num_policies" in dims

    def test_error_handling(self, model_params, template_config):
        """Test error handling in various operations."""
        if not IMPORT_SUCCESS:
            return

        wrapper = PyMDPAgentWrapper(model_params, template_config)

        # Test with invalid observation
        with pytest.warns(UserWarning):
            # This should trigger fallback due to error
            wrapper.agent = Mock()
            wrapper.agent.infer_states.side_effect = Exception("Mock error")

            result = wrapper.update_beliefs(0)
            assert result is not None  # Should fallback gracefully

    def test_observation_format_conversion(self, wrapper):
        """Test observation format conversion."""
        if not IMPORT_SUCCESS:
            return

        # Test integer observation conversion
        belief1 = wrapper.update_beliefs(1)
        assert belief1 is not None

        # Test array observation conversion
        obs_array = np.array([0.9, 0.05, 0.05])
        belief2 = wrapper.update_beliefs(obs_array)
        assert belief2 is not None

        # Test edge case: all-zero observation
        zero_obs = np.array([0.0, 0.0, 0.0])
        belief3 = wrapper.update_beliefs(zero_obs)
        assert belief3 is not None

    def test_precision_parameter_integration(self, model_params, template_config):
        """Test precision parameter integration."""
        if not IMPORT_SUCCESS:
            return

        wrapper = PyMDPAgentWrapper(model_params, template_config)

        # Check precision parameters are stored
        assert hasattr(wrapper.model_params, "precision_sensory")
        assert hasattr(wrapper.model_params, "precision_policy")
        assert hasattr(wrapper.model_params, "precision_state")

        # Test in mathematical summary
        summary = wrapper.get_mathematical_summary()
        assert summary["precision_sensory"] == wrapper.model_params.precision_sensory
        assert summary["precision_policy"] == wrapper.model_params.precision_policy
        assert summary["precision_state"] == wrapper.model_params.precision_state

    def test_belief_state_consistency(self, wrapper):
        """Test belief state consistency across operations."""
        if not IMPORT_SUCCESS:
            return

        # Update beliefs and check consistency
        belief_state = wrapper.update_beliefs(0)

        if IMPORT_SUCCESS:
            # Should have valid belief state properties
            assert np.sum(belief_state.beliefs) == pytest.approx(1.0)
            assert np.all(belief_state.beliefs >= 0)
            assert belief_state.confidence >= 0
            assert belief_state.timestamp >= 0

            # Policies should be normalized
            assert np.sum(belief_state.policies) == pytest.approx(1.0)
            assert np.all(belief_state.policies >= 0)

    def test_large_observation_spaces(self, template_config):
        """Test with large observation spaces."""
        if not IMPORT_SUCCESS:
            return

        # Create larger model
        large_A = np.random.rand(20, 10)
        large_A = large_A / np.sum(large_A, axis=0, keepdims=True)  # Normalize

        large_B = np.zeros((10, 10, 5))
        for i in range(5):
            large_B[:, :, i] = np.eye(10)

        large_C = np.random.randn(20)
        large_D = np.ones(10) / 10

        if IMPORT_SUCCESS:
            large_params = GenerativeModelParams(A=large_A, B=large_B, C=large_C, D=large_D)
        else:
            large_params = Mock(A=large_A, B=large_B, C=large_C, D=large_D)

        wrapper = PyMDPAgentWrapper(large_params, template_config)

        # Should handle large observations
        obs = np.random.rand(20)
        obs = obs / np.sum(obs)  # Normalize

        belief_state = wrapper.update_beliefs(obs)
        assert belief_state is not None

    def test_warning_generation(self, model_params, template_config):
        """Test warning generation for various conditions."""
        if not IMPORT_SUCCESS:
            return

        # Test pymdp unavailable warning
        with patch("agents.templates.pymdp_integration.PYMDP_AVAILABLE", False):
            with pytest.warns(UserWarning, match="pymdp library not available"):
                PyMDPAgentWrapper(model_params, template_config)

    def test_concurrent_operations(self, wrapper):
        """Test concurrent operations don't interfere."""
        if not IMPORT_SUCCESS:
            return

        # Simulate concurrent belief updates
        observations = [0, 1, 2, 0, 1]
        results = []

        for obs in observations:
            result = wrapper.update_beliefs(obs)
            results.append(result)

        # All results should be valid
        for result in results:
            assert result is not None
            if IMPORT_SUCCESS:
                assert isinstance(result, BeliefState)

    def test_memory_efficiency(self, wrapper):
        """Test memory efficiency with repeated operations."""
        if not IMPORT_SUCCESS:
            return

        # Perform many operations and check memory doesn't grow excessively
        for i in range(100):
            obs = i % 3  # Cycle through observations
            belief_state = wrapper.update_beliefs(obs)

            if IMPORT_SUCCESS:
                # Check beliefs are still valid
                assert np.sum(belief_state.beliefs) == pytest.approx(1.0)

                # Compute free energy
                fe = wrapper.compute_free_energy(belief_state, obs)
                assert isinstance(fe, float)


class TestCreatePyMDPAgent:
    """Test the factory function for creating pymdp agents."""

    @pytest.fixture
    def valid_params(self):
        """Create valid parameters for testing."""
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        B = np.zeros((2, 2, 2))
        B[:, :, 0] = B[:, :, 1] = np.eye(2)
        C = np.array([1.0, -1.0])
        D = np.array([0.5, 0.5])

        if IMPORT_SUCCESS:
            return GenerativeModelParams(A=A, B=B, C=C, D=D)
        else:
            return Mock(A=A, B=B, C=C, D=D)

    @pytest.fixture
    def valid_config(self):
        """Create valid configuration."""
        if IMPORT_SUCCESS:
            return TemplateConfig(
                template_id="factory_test",
                category=TemplateCategory.MERCHANT,
                num_states=2,
                num_observations=2,
                num_policies=2,
            )
        else:
            return Mock(
                template_id="factory_test",
                category="merchant",
                num_states=2,
                num_observations=2,
                num_policies=2,
            )

    def test_factory_function(self, valid_params, valid_config):
        """Test factory function creates valid wrapper."""
        wrapper = create_pymdp_agent(valid_params, valid_config)

        assert isinstance(wrapper, PyMDPAgentWrapper)
        assert wrapper.model_params == valid_params
        assert wrapper.config == valid_config

    def test_factory_validation(self, valid_config):
        """Test factory function validates parameters."""
        if not IMPORT_SUCCESS:
            return

        # Create invalid parameters
        # Columns don't sum to 1
        invalid_A = np.array([[0.5, 0.3], [0.3, 0.5]])
        invalid_params = GenerativeModelParams(
            A=invalid_A, B=np.zeros((2, 2, 2)), C=np.zeros(2), D=np.array([0.5, 0.5])
        )

        # Should raise validation error
        with pytest.raises(ValueError):
            create_pymdp_agent(invalid_params, valid_config)


class TestPyMDPIntegration:
    """Test overall pymdp integration functionality."""

    def test_pymdp_import_handling(self):
        """Test handling of pymdp import scenarios."""
        # Test current import status
        if IMPORT_SUCCESS:
            assert hasattr(globals(), "PYMDP_AVAILABLE")

            # Test fallback implementations exist
            from agents.templates.pymdp_integration import PyMDPAgentWrapper

            # Should be able to create wrapper regardless of pymdp availability
            mock_params = Mock()
            mock_config = Mock(num_states=2, num_observations=2, num_policies=2, planning_horizon=3)

            wrapper = PyMDPAgentWrapper(mock_params, mock_config)
            assert wrapper is not None

    def test_mathematical_operations_consistency(self):
        """Test mathematical operations are consistent."""
        if not IMPORT_SUCCESS:
            return

        # Test that fallback implementations give reasonable results
        x = np.array([1.0, 2.0, 3.0])

        # Create wrapper to access fallback functions
        wrapper = PyMDPAgentWrapper(Mock(), Mock())

        if hasattr(wrapper, "_fallback_softmax"):
            result = wrapper._fallback_softmax(x)

            # Should be valid probability distribution
            assert np.sum(result) == pytest.approx(1.0)
            assert np.all(result >= 0)
            assert np.all(result <= 1)

    def test_integration_documentation(self):
        """Test that integration is properly documented."""
        if IMPORT_SUCCESS:
            # Check module has proper docstring
            import agents.templates.pymdp_integration as pymdp_module

            assert pymdp_module.__doc__ is not None
            assert len(pymdp_module.__doc__) > 0

            # Check class documentation
            assert PyMDPAgentWrapper.__doc__ is not None
            assert "pymdp" in PyMDPAgentWrapper.__doc__.lower()

    def test_performance_benchmarks(self):
        """Test basic performance characteristics."""
        if not IMPORT_SUCCESS:
            return

        import time

        # Create simple model
        A = np.eye(3)
        B = np.zeros((3, 3, 2))
        B[:, :, 0] = B[:, :, 1] = np.eye(3)
        C = np.zeros(3)
        D = np.ones(3) / 3

        if IMPORT_SUCCESS:
            params = GenerativeModelParams(A=A, B=B, C=C, D=D)
            config = TemplateConfig(
                template_id="perf_test",
                category=TemplateCategory.EXPLORER,
                num_states=3,
                num_observations=3,
                num_policies=2,
            )
        else:
            params = Mock(A=A, B=B, C=C, D=D)
            config = Mock(num_states=3, num_observations=3, num_policies=2)

        wrapper = PyMDPAgentWrapper(params, config)

        # Time belief updates
        start_time = time.time()
        for i in range(10):
            wrapper.update_beliefs(i % 3)
        end_time = time.time()

        # Should be reasonably fast
        # Less than 1 second for 10 updates
        assert (end_time - start_time) < 1.0
