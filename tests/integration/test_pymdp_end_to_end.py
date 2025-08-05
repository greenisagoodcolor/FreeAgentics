"""End-to-end integration test for PyMDP agent creation pipeline.

This test validates the complete flow:
LLM → GMN → PyMDP Agent → Database Storage → Runtime Agent
"""

import pytest
from unittest.mock import patch, AsyncMock, Mock
import numpy as np

from agents.creation.pymdp_agent_builder import PyMDPAgentBuilder
from agents.creation.models import AgentSpecification, PersonalityProfile
from agents.pymdp_agent_factory import PyMDPAgentFactory
from database.models import AgentType
from inference.active.gmn_parser import GMNParser


@pytest.mark.integration
class TestPyMDPEndToEnd:
    """Integration tests for complete PyMDP agent pipeline."""

    @pytest.fixture
    def personality_profile(self):
        """Create a test personality profile."""
        return PersonalityProfile(
            assertiveness=0.8, analytical_depth=0.9, creativity=0.6, empathy=0.5, skepticism=0.7
        )

    @pytest.fixture
    def agent_specification(self, personality_profile):
        """Create agent specification for Active Inference agent."""
        return AgentSpecification(
            name="AI Explorer",
            agent_type=AgentType.ANALYST,
            system_prompt="You are an active inference agent that explores environments",
            personality=personality_profile,
            source_prompt="Create an agent using active inference with PyMDP for exploration and belief updating",
            creation_source="integration_test",
            parameters={"use_pymdp": True},
        )

    def test_gmn_parser_to_pymdp_factory_integration(self):
        """Test that GMN parser output works with PyMDP factory."""
        # Step 1: Parse GMN specification
        parser = GMNParser()
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

        # Parse GMN to PyMDP specification
        graph = parser.parse(gmn_text)
        gmn_spec = parser.to_pymdp_model(graph)

        # Step 2: Create PyMDP agent from specification
        factory = PyMDPAgentFactory()
        agent = factory.create_agent(gmn_spec)

        # Step 3: Verify agent can perform inference
        from pymdp.agent import Agent as PyMDPAgent

        assert isinstance(agent, PyMDPAgent)

        # Test belief updating
        observation = [0]
        beliefs = agent.infer_states(observation)
        assert beliefs is not None

        # Test action sampling
        agent.infer_policies()
        action = agent.sample_action()
        assert isinstance(action, np.ndarray)

        # Verify factory metrics
        metrics = factory.get_metrics()
        assert metrics["agents_created"] == 1
        assert metrics["success_rate"] == 1.0

    @patch("agents.creation.pymdp_agent_builder.get_db")
    @patch("agents.creation.pymdp_agent_builder.get_provider_factory")
    async def test_full_agent_creation_pipeline(self, mock_factory, mock_db, agent_specification):
        """Test complete agent creation pipeline from specification to database."""
        # Mock database session
        mock_session = Mock()
        mock_db.return_value = [mock_session]
        mock_session.refresh = lambda x: setattr(x, "id", 42)

        # Mock LLM provider with valid GMN response
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = """
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
        mock_factory.return_value.create_configured_provider.return_value = mock_llm

        # Create PyMDP agent builder
        builder = PyMDPAgentBuilder()

        # Build agent through complete pipeline
        agent = await builder.build_agent(agent_specification)

        # Verify agent was created
        assert agent is not None

        # Verify database operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

        # Verify agent has PyMDP metadata
        saved_agent = mock_session.add.call_args[0][0]
        assert saved_agent.parameters["agent_type"] == "pymdp_active_inference"
        assert "gmn_spec" in saved_agent.parameters
        assert "pymdp_version" in saved_agent.parameters

        # Verify metrics
        metrics = builder.get_metrics()
        assert metrics["pymdp_agents_created"] == 1

    @patch("agents.creation.pymdp_agent_builder.get_db")
    @patch("agents.creation.pymdp_agent_builder.get_provider_factory")
    async def test_fallback_to_traditional_agent_on_failure(
        self, mock_factory, mock_db, agent_specification
    ):
        """Test fallback to traditional agent when PyMDP creation fails."""
        # Mock database session
        mock_session = Mock()
        mock_db.return_value = [mock_session]
        mock_session.refresh = lambda x: setattr(x, "id", 43)

        # Mock LLM provider with invalid GMN response
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "Invalid GMN response"
        mock_factory.return_value.create_configured_provider.return_value = mock_llm

        # Create PyMDP agent builder
        builder = PyMDPAgentBuilder()

        # Build agent - should fall back to traditional
        agent = await builder.build_agent(agent_specification)

        # Verify agent was created
        assert agent is not None

        # Verify database operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

        # Verify fallback to traditional agent
        metrics = builder.get_metrics()
        assert metrics["traditional_agents_created"] == 1
        assert metrics["pymdp_agents_created"] == 0

    def test_pymdp_factory_error_handling(self):
        """Test PyMDP factory error handling with invalid specifications."""
        factory = PyMDPAgentFactory()

        # Test invalid GMN specification
        invalid_spec = {
            "num_states": [4],
            # Missing required fields
        }

        from agents.pymdp_agent_factory import PyMDPAgentCreationError

        with pytest.raises(PyMDPAgentCreationError) as exc_info:
            factory.create_agent(invalid_spec)

        assert "validation failed" in str(exc_info.value)

        # Verify error metrics
        metrics = factory.get_metrics()
        assert metrics["creation_failures"] == 1
        assert metrics["validation_failures"] == 1

    def test_gmn_specification_validation(self):
        """Test GMN specification validation edge cases."""
        factory = PyMDPAgentFactory()

        # Valid single-factor specification
        valid_spec = {
            "num_states": [4],
            "num_obs": [4],
            "num_actions": [4],
            "A": [np.eye(4)],
            "B": [np.zeros((4, 4, 4))],
            "C": [np.zeros(4)],
            "D": [np.ones(4) / 4],
        }

        # Fill B matrix properly
        for action in range(4):
            valid_spec["B"][0][:, :, action] = np.eye(4)

        assert factory.validate_gmn_spec(valid_spec) is True

        # Invalid multi-factor specification
        multi_factor_spec = {
            "num_states": [3, 3],  # Two factors
            "num_obs": [4],
            "num_actions": [4],
            "A": [np.random.rand(4, 3, 3)],
            "B": [np.random.rand(3, 3, 4), np.random.rand(3, 3, 4)],
            "C": [np.zeros(4)],
            "D": [np.ones(3) / 3, np.ones(3) / 3],
        }

        # Should pass validation but fail creation due to multi-factor limitation
        assert factory.validate_gmn_spec(multi_factor_spec) is True

        from agents.pymdp_agent_factory import PyMDPAgentCreationError

        with pytest.raises(PyMDPAgentCreationError) as exc_info:
            factory.create_agent(multi_factor_spec)

        assert "Multi-factor models not supported" in str(exc_info.value)

    def test_pymdp_agent_real_functionality(self):
        """Test that created PyMDP agents actually work for inference."""
        # Create a simple valid GMN specification
        B_matrix = np.zeros((4, 4, 4))
        for action in range(4):
            B_matrix[:, :, action] = np.eye(4)  # Identity transitions

        gmn_spec = {
            "num_states": [4],
            "num_obs": [4],
            "num_actions": [4],
            "A": [np.eye(4)],  # Identity observation model
            "B": [B_matrix],
            "C": [np.array([1.0, 0.0, 0.0, 0.0])],  # Preferences
            "D": [np.ones(4) / 4],  # Uniform prior
        }

        # Create PyMDP agent
        factory = PyMDPAgentFactory()
        agent = factory.create_agent(gmn_spec)

        # Test multiple inference steps
        observations = [[0], [1], [2], [3]]

        for obs in observations:
            # Infer states
            beliefs = agent.infer_states(obs)
            assert beliefs is not None

            # Infer policies
            q_pi, G = agent.infer_policies()
            assert q_pi is not None
            assert G is not None

            # Sample action
            action = agent.sample_action()
            assert isinstance(action, np.ndarray)
            assert 0 <= action.item() < 4  # Valid action range

        # Verify factory metrics after multiple operations
        metrics = factory.get_metrics()
        assert metrics["agents_created"] == 1
        assert metrics["success_rate"] == 1.0
