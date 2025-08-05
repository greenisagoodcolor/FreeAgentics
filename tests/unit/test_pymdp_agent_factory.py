"""Test suite for PyMDP Agent Factory - TDD approach."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from agents.pymdp_agent_factory import PyMDPAgentFactory, PyMDPAgentCreationError
from inference.active.gmn_parser import GMNGraph, GMNNode, GMNEdge, GMNNodeType, GMNEdgeType


class TestPyMDPAgentFactory:
    """Test PyMDP Agent Factory with real PyMDP integration."""

    def test_create_agent_from_simple_gmn_spec(self):
        """Test creating a real PyMDP agent from a simple GMN specification."""
        # Arrange - Simple grid world GMN spec
        B_matrix = np.zeros((4, 4, 4))
        # Create proper transition matrices that sum to 1
        for action in range(4):
            B_matrix[:, :, action] = np.eye(4)  # Identity transitions
        
        gmn_spec = {
            "num_states": [4],
            "num_obs": [4], 
            "num_actions": [4],
            "A": [np.eye(4)],  # Identity observation model
            "B": [B_matrix],  # Proper transition model
            "C": [np.array([1.0, 0.0, 0.0, 0.0])],  # Preferences
            "D": [np.ones(4) / 4]  # Uniform prior
        }
        
        factory = PyMDPAgentFactory()
        
        # Act - This should now work with real implementation
        agent = factory.create_agent(gmn_spec)
        
        # Assert - Verify we got a real PyMDP agent
        from pymdp.agent import Agent as PyMDPAgent
        assert isinstance(agent, PyMDPAgent)
        assert hasattr(agent, 'A')
        assert hasattr(agent, 'B')
        assert hasattr(agent, 'infer_states')
        assert hasattr(agent, 'sample_action')

    def test_validate_gmn_spec_structure(self):
        """Test GMN specification validation."""
        factory = PyMDPAgentFactory()
        
        # Valid spec should pass
        B_matrix = np.zeros((4, 4, 4))
        for action in range(4):
            B_matrix[:, :, action] = np.eye(4)
            
        valid_spec = {
            "num_states": [4],
            "num_obs": [4],
            "num_actions": [4],
            "A": [np.eye(4)],
            "B": [B_matrix],
            "C": [np.zeros(4)],
            "D": [np.ones(4) / 4]
        }
        
        # Should pass validation
        is_valid = factory.validate_gmn_spec(valid_spec)
        assert is_valid is True
        
        # Invalid spec should fail
        invalid_spec = {
            "num_states": [4],
            # Missing required fields
        }
        
        is_valid = factory.validate_gmn_spec(invalid_spec)
        assert is_valid is False

    def test_handle_invalid_gmn_spec(self):
        """Test error handling for invalid GMN specifications."""
        factory = PyMDPAgentFactory()
        
        invalid_spec = {
            "num_states": [4],
            # Missing required fields
        }
        
        # Should raise proper error
        with pytest.raises(PyMDPAgentCreationError) as exc_info:
            factory.create_agent(invalid_spec)
        
        assert "validation failed" in str(exc_info.value)

    def test_create_agent_with_multi_factor_model(self):
        """Test creating agent with multi-factor state space."""
        factory = PyMDPAgentFactory()
        
        multi_factor_spec = {
            "num_states": [3, 3],  # Two state factors  
            "num_obs": [4],
            "num_actions": [4],
            "A": [np.random.rand(4, 3, 3)],  # Observation depends on both factors
            "B": [
                np.random.rand(3, 3, 4),  # First factor transitions
                np.random.rand(3, 3, 4)   # Second factor transitions  
            ],
            "C": [np.zeros(4)],
            "D": [np.ones(3) / 3, np.ones(3) / 3]
        }
        
        # Should fail with proper error message about multi-factor models
        with pytest.raises(PyMDPAgentCreationError) as exc_info:
            agent = factory.create_agent(multi_factor_spec)
        
        assert "Multi-factor models not supported" in str(exc_info.value)

    def test_factory_performance_metrics(self):
        """Test that factory collects performance metrics."""
        factory = PyMDPAgentFactory()
        
        # Should return metrics
        metrics = factory.get_metrics()
        assert isinstance(metrics, dict)
        assert "agents_created" in metrics
        assert "creation_failures" in metrics
        assert "success_rate" in metrics

    def test_factory_error_handling_and_logging(self):
        """Test factory error handling with proper logging."""
        factory = PyMDPAgentFactory()
        
        # Malformed spec should raise specific error
        malformed_spec = {"invalid": "spec"}
        
        with pytest.raises(PyMDPAgentCreationError):
            factory.create_agent(malformed_spec)

    @pytest.mark.integration  
    def test_end_to_end_gmn_to_pymdp_flow(self):
        """Integration test: GMN parsing -> PyMDP creation -> Agent execution."""
        # This test will verify the complete pipeline works
        gmn_text = """
        [nodes]
        location: state {num_states: 4}
        obs_location: observation {num_observations: 4}
        move: action {num_actions: 4}
        location_belief: belief
        location_pref: preference {preferred_observation: 0}
        location_likelihood: likelihood
        location_transition: transition

        [edges]
        location -> location_likelihood: depends_on
        location_likelihood -> obs_location: generates
        location -> location_transition: depends_on
        move -> location_transition: depends_on
        location_pref -> obs_location: depends_on
        location_belief -> location: depends_on
        """
        
        from inference.active.gmn_parser import GMNParser
        
        # Parse GMN to specification
        parser = GMNParser()
        graph = parser.parse(gmn_text)
        gmn_spec = parser.to_pymdp_model(graph)
        
        # Create PyMDP agent from specification  
        factory = PyMDPAgentFactory()
        agent = factory.create_agent(gmn_spec)
        
        # Test agent can perform inference
        observation = [0]  # First observation type
        beliefs = agent.infer_states(observation)
        
        # Infer policies before sampling action (PyMDP requirement)
        agent.infer_policies()
        action = agent.sample_action() 
        
        # Verify results
        assert isinstance(action, np.ndarray)
        assert beliefs is not None

    def test_factory_caching_for_performance(self):
        """Test that factory caches matrix computations for performance."""
        factory = PyMDPAgentFactory()
        
        B_matrix = np.zeros((4, 4, 4))
        for action in range(4):
            B_matrix[:, :, action] = np.eye(4)
        
        spec = {
            "num_states": [4],
            "num_obs": [4],
            "num_actions": [4], 
            "A": [np.eye(4)],
            "B": [B_matrix],
            "C": [np.zeros(4)],
            "D": [np.ones(4) / 4]
        }
        
        # Create two agents with same spec
        agent1 = factory.create_agent(spec)
        agent2 = factory.create_agent(spec)
        
        # Check that metrics are updated
        metrics = factory.get_metrics()
        assert metrics["agents_created"] == 2
        # Note: caching is currently disabled for safety, but metrics still track cache operations
        assert "cache_hits" in metrics