"""
Tests for PyMDP Integration Module.

This test suite verifies that the FreeAgentics active inference implementation
correctly aligns with pymdp conventions and integrates with GeneralizedNotationNotation
(GNN/GMN) for LLM-based model generation.
"""

import pytest
import torch
import numpy as np

from inference.engine.pymdp_integration import (
    PyMDPActiveInference,
    GMNToPyMDPConverter, 
    create_pymdp_agent
)
from inference.engine.generative_model import DiscreteGenerativeModel, ModelDimensions, ModelParameters
from inference.engine.active_inference import InferenceConfig


class TestPyMDPActiveInference:
    """Test PyMDP-compatible active inference implementation."""
    
    @pytest.fixture
    def pymdp_model(self):
        """Create a pymdp-compatible discrete generative model."""
        dims = ModelDimensions(
            num_states=4,
            num_observations=3, 
            num_actions=2,
            time_horizon=1
        )
        params = ModelParameters(learning_rate=0.01, use_gpu=False)
        return DiscreteGenerativeModel(dims, params)
    
    @pytest.fixture
    def pymdp_agent(self, pymdp_model):
        """Create a PyMDP active inference agent."""
        config = InferenceConfig(use_gpu=False, num_iterations=10)
        return PyMDPActiveInference(pymdp_model, config)
    
    def test_initialization(self, pymdp_agent):
        """Test PyMDP agent initialization."""
        assert pymdp_agent.pymdp_compatible is True
        assert hasattr(pymdp_agent, 'generative_model')
        assert hasattr(pymdp_agent, 'vmp')
        
    def test_pymdp_tensor_conventions(self, pymdp_agent):
        """Test that tensor shapes follow pymdp conventions."""
        model = pymdp_agent.generative_model
        
        # A matrix: (num_obs, num_states) - pymdp convention
        assert model.A.shape == (3, 4)
        
        # B matrix: (num_states, num_states, num_actions) - pymdp convention  
        assert model.B.shape == (4, 4, 2)
        
        # D vector: (num_states,) - pymdp convention
        assert model.D.shape == (4,)
        
        # Verify normalization (pymdp requirement)
        # A matrix columns should sum to 1
        for s in range(4):
            assert torch.allclose(model.A[:, s].sum(), torch.tensor(1.0), atol=1e-6)
            
        # B matrix should be normalized along first dimension  
        for a in range(2):
            for s in range(4):
                assert torch.allclose(model.B[:, s, a].sum(), torch.tensor(1.0), atol=1e-6)
                
        # D should sum to 1
        assert torch.allclose(model.D.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_belief_updates_pymdp_style(self, pymdp_agent):
        """Test belief updates follow pymdp Bayesian inference."""
        # Test discrete observation
        obs = torch.tensor(1, dtype=torch.long)
        beliefs = pymdp_agent.update_beliefs(obs)
        
        # Should return categorical distribution
        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(beliefs >= 0)  # Non-negative probabilities
        
        # Test with prior beliefs
        prior = torch.tensor([0.4, 0.3, 0.2, 0.1])
        beliefs_with_prior = pymdp_agent.update_beliefs(obs, prior)
        assert beliefs_with_prior.shape == (4,)
        assert torch.allclose(beliefs_with_prior.sum(), torch.tensor(1.0), atol=1e-6)
        
        # Test batch observations
        obs_batch = torch.tensor([0, 1, 2], dtype=torch.long)
        beliefs_batch = pymdp_agent.update_beliefs(obs_batch)
        assert beliefs_batch.shape == (3, 4)  # (batch_size, num_states)
        
        # Each belief in batch should sum to 1
        for i in range(3):
            assert torch.allclose(beliefs_batch[i].sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_expected_free_energy_computation(self, pymdp_agent):
        """Test expected free energy follows pymdp formulation."""
        beliefs = torch.tensor([0.25, 0.25, 0.25, 0.25])  # Uniform
        
        # Test single action
        efe = pymdp_agent.compute_expected_free_energy(beliefs, 0)
        assert isinstance(efe, torch.Tensor)
        assert efe.numel() == 1
        
        # Test action sequence
        actions = torch.tensor([0, 1])
        efe_sequence = pymdp_agent.compute_expected_free_energy(beliefs, actions, time_horizon=2)
        assert isinstance(efe_sequence, torch.Tensor)
        
        # Test that EFE is finite (no NaN or inf)
        assert torch.isfinite(efe)
        assert torch.isfinite(efe_sequence)
    
    def test_action_selection_pymdp_style(self, pymdp_agent):
        """Test action selection minimizes expected free energy."""
        beliefs = torch.tensor([0.7, 0.2, 0.05, 0.05])  # Confident belief
        
        action, efe_values = pymdp_agent.select_action(beliefs)
        
        # Should return valid action
        assert isinstance(action, int)
        assert 0 <= action < 2  # Within action space
        
        # EFE values for all actions
        assert efe_values.shape == (2,)
        assert torch.all(torch.isfinite(efe_values))
        
        # Selected action should minimize EFE
        assert action == torch.argmin(efe_values).item()
    
    def test_categorical_distribution_properties(self, pymdp_agent):
        """Test that all distributions maintain categorical properties."""
        obs_sequence = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
        
        for obs in obs_sequence:
            beliefs = pymdp_agent.update_beliefs(obs)
            
            # Categorical distribution properties (pymdp requirement)
            assert torch.all(beliefs >= 0)  # Non-negative
            assert torch.allclose(beliefs.sum(), torch.tensor(1.0), atol=1e-6)  # Normalized
            assert beliefs.shape == (4,)  # Correct dimensionality


class TestGMNToPyMDPConverter:
    """Test GeneralizedNotationNotation to pymdp conversion."""
    
    @pytest.fixture
    def converter(self):
        """Create GMN to pymdp converter."""
        return GMNToPyMDPConverter()
    
    def test_converter_initialization(self, converter):
        """Test converter initialization."""
        assert hasattr(converter, 'gmn_generator')
        assert hasattr(converter, 'gmn_parser')
    
    def test_agent_class_specific_models(self, converter):
        """Test that different agent classes generate appropriate models."""
        explorer_config = {
            'agent_name': 'Explorer1',
            'agent_class': 'Explorer',
            'personality': {'exploration': 0.8, 'curiosity': 0.9}
        }
        
        merchant_config = {
            'agent_name': 'Merchant1', 
            'agent_class': 'Merchant',
            'personality': {'efficiency': 0.8, 'cooperation': 0.7}
        }
        
        guardian_config = {
            'agent_name': 'Guardian1',
            'agent_class': 'Guardian', 
            'personality': {'risk_tolerance': 0.2}
        }
        
        # Generate models
        explorer_model = converter.generate_pymdp_model_from_gmn(explorer_config)
        merchant_model = converter.generate_pymdp_model_from_gmn(merchant_config)
        guardian_model = converter.generate_pymdp_model_from_gmn(guardian_config)
        
        # Check dimensions are appropriate for each class
        assert explorer_model.dims.num_states == 4
        assert explorer_model.dims.num_observations == 3
        assert explorer_model.dims.num_actions == 2
        
        assert merchant_model.dims.num_states == 4
        assert merchant_model.dims.num_observations == 3
        assert merchant_model.dims.num_actions == 3
        
        assert guardian_model.dims.num_states == 3
        assert guardian_model.dims.num_observations == 3
        assert guardian_model.dims.num_actions == 3
        
        # Check models follow pymdp conventions
        for model in [explorer_model, merchant_model, guardian_model]:
            # A matrix normalization
            for s in range(model.dims.num_states):
                assert torch.allclose(model.A[:, s].sum(), torch.tensor(1.0), atol=1e-6)
            
            # B matrix normalization
            for a in range(model.dims.num_actions):
                for s in range(model.dims.num_states):
                    assert torch.allclose(model.B[:, s, a].sum(), torch.tensor(1.0), atol=1e-6)
            
            # D normalization
            assert torch.allclose(model.D.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_personality_customization(self, converter):
        """Test that personality traits affect model parameters."""
        high_exploration_config = {
            'agent_name': 'HighExplorer',
            'agent_class': 'Explorer',
            'personality': {'exploration': 0.9, 'curiosity': 0.8}
        }
        
        low_exploration_config = {
            'agent_name': 'LowExplorer', 
            'agent_class': 'Explorer',
            'personality': {'exploration': 0.1, 'curiosity': 0.2}
        }
        
        high_model = converter.generate_pymdp_model_from_gmn(high_exploration_config)
        low_model = converter.generate_pymdp_model_from_gmn(low_exploration_config)
        
        # Models should be different due to personality differences
        assert not torch.allclose(high_model.A, low_model.A, atol=1e-6)
        assert not torch.allclose(high_model.B, low_model.B, atol=1e-6)
        
        # But both should maintain pymdp conventions
        for model in [high_model, low_model]:
            assert torch.allclose(model.D.sum(), torch.tensor(1.0), atol=1e-6)


class TestPyMDPAgentFactory:
    """Test the create_pymdp_agent factory function."""
    
    def test_create_explorer_agent(self):
        """Test creating an explorer agent."""
        config = {
            'agent_name': 'TestExplorer',
            'agent_class': 'Explorer',
            'personality': {
                'exploration': 0.8,
                'curiosity': 0.9,
                'risk_tolerance': 0.6
            }
        }
        
        agent = create_pymdp_agent(config)
        
        assert isinstance(agent, PyMDPActiveInference)
        assert agent.pymdp_compatible is True
        
        # Test basic functionality
        obs = torch.tensor(1, dtype=torch.long)
        beliefs = agent.update_beliefs(obs)
        action, efe_values = agent.select_action(beliefs)
        
        assert beliefs.shape == (4,)
        assert isinstance(action, int)
        assert efe_values.shape == (2,)
    
    def test_create_merchant_agent(self):
        """Test creating a merchant agent.""" 
        config = {
            'agent_name': 'TestMerchant',
            'agent_class': 'Merchant',
            'personality': {
                'efficiency': 0.8,
                'cooperation': 0.7
            }
        }
        
        agent = create_pymdp_agent(config)
        
        # Merchant should have 3 actions
        beliefs = torch.ones(4) / 4
        action, efe_values = agent.select_action(beliefs)
        assert efe_values.shape == (3,)
    
    def test_create_guardian_agent(self):
        """Test creating a guardian agent."""
        config = {
            'agent_name': 'TestGuardian', 
            'agent_class': 'Guardian',
            'personality': {
                'risk_tolerance': 0.2
            }
        }
        
        agent = create_pymdp_agent(config)
        
        # Guardian should have 3 states and 3 actions
        beliefs = torch.ones(3) / 3
        action, efe_values = agent.select_action(beliefs, num_actions=3)
        assert efe_values.shape == (3,)


class TestPyMDPCompatibility:
    """Test compatibility with pymdp library conventions."""
    
    def test_belief_update_equivalence(self):
        """Test that belief updates match pymdp-style calculations."""
        # Create simple model
        dims = ModelDimensions(num_states=2, num_observations=2, num_actions=1)
        params = ModelParameters(use_gpu=False)
        model = DiscreteGenerativeModel(dims, params)
        
        # Set known A matrix: [[0.8, 0.2], [0.3, 0.7]]
        model.A.data = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        
        agent = PyMDPActiveInference(model)
        
        # Prior belief: uniform [0.5, 0.5]
        prior = torch.tensor([0.5, 0.5])
        
        # Observe outcome 0
        obs = torch.tensor(0, dtype=torch.long)
        posterior = agent.update_beliefs(obs, prior)
        
        # Manual calculation: P(s|o=0) ∝ P(o=0|s) * P(s)
        # P(o=0|s=0) = 0.8, P(o=0|s=1) = 0.2
        # Posterior ∝ [0.8*0.5, 0.2*0.5] = [0.4, 0.1]
        # Normalized: [0.8, 0.2]
        expected = torch.tensor([0.8, 0.2])
        
        assert torch.allclose(posterior, expected, atol=1e-6)
    
    def test_free_energy_formulation(self):
        """Test that free energy matches pymdp formulation."""
        # Create simple model
        config = {
            'agent_name': 'TestAgent',
            'agent_class': 'Explorer',
            'personality': {}
        }
        
        agent = create_pymdp_agent(config)
        beliefs = torch.tensor([0.6, 0.2, 0.1, 0.1])
        
        # Compute free energy manually and via agent
        model = agent.generative_model
        obs = torch.tensor(0, dtype=torch.long)
        
        # Manual calculation: F = KL[q(s)||p(s)] - E_q[ln p(o|s)]
        prior = model.D
        complexity = torch.sum(beliefs * (torch.log(beliefs + 1e-16) - torch.log(prior + 1e-16)))
        accuracy = torch.sum(beliefs * torch.log(model.A[obs, :] + 1e-16))
        expected_fe = complexity - accuracy
        
        # Agent calculation
        agent_fe = agent.vmp.compute_free_energy(beliefs, obs, model)
        
        # Should be approximately equal (allowing for numerical differences)
        assert torch.allclose(agent_fe, expected_fe, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])