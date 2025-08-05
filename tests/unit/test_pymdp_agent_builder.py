"""Test suite for PyMDP Agent Builder."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from agents.creation.pymdp_agent_builder import PyMDPAgentBuilder
from agents.creation.models import AgentSpecification, PersonalityProfile
from database.models import AgentType
from agents.pymdp_agent_factory import PyMDPAgentCreationError


class TestPyMDPAgentBuilder:
    """Test PyMDP Agent Builder integration."""

    @pytest.fixture
    def mock_personality(self):
        """Mock personality profile."""
        return PersonalityProfile(
            assertiveness=0.7,
            analytical_depth=0.9,
            creativity=0.6,
            empathy=0.5,
            skepticism=0.8
        )

    @pytest.fixture  
    def basic_agent_spec(self, mock_personality):
        """Basic agent specification."""
        return AgentSpecification(
            name="Test Agent",
            agent_type=AgentType.ANALYST,
            system_prompt="You are a helpful assistant",
            personality=mock_personality,
            source_prompt="Create an agent that explores environments",
            creation_source="test",
            parameters={}
        )

    @pytest.fixture
    def pymdp_agent_spec(self, mock_personality):
        """Agent specification that should trigger PyMDP creation."""
        return AgentSpecification(
            name="Active Inference Explorer",
            agent_type=AgentType.ANALYST,
            system_prompt="You are an active inference agent",
            personality=mock_personality,
            source_prompt="Create an agent using active inference and pymdp for exploration with belief updating",
            creation_source="test",
            parameters={"use_pymdp": True}
        )

    def test_should_create_pymdp_agent_explicit_request(self, basic_agent_spec):
        """Test PyMDP detection with explicit parameter."""
        builder = PyMDPAgentBuilder()
        
        # Change to CREATIVE type to avoid auto-PyMDP for ANALYST
        basic_agent_spec.agent_type = AgentType.CREATIVE
        # Should not use PyMDP by default
        assert not builder._should_create_pymdp_agent(basic_agent_spec)
        
        # Should use PyMDP when explicitly requested
        basic_agent_spec.parameters = {"use_pymdp": True}
        assert builder._should_create_pymdp_agent(basic_agent_spec)

    def test_should_create_pymdp_agent_keyword_detection(self, basic_agent_spec):
        """Test PyMDP detection based on keywords in source prompt."""
        builder = PyMDPAgentBuilder()
        
        # Test with Active Inference keywords
        basic_agent_spec.source_prompt = "Create an agent that uses active inference and belief updating"
        assert builder._should_create_pymdp_agent(basic_agent_spec)
        
        # Test with PyMDP keywords
        basic_agent_spec.source_prompt = "Agent should explore environment using curiosity and free energy"
        assert builder._should_create_pymdp_agent(basic_agent_spec)

    def test_should_create_pymdp_agent_type_based(self, basic_agent_spec):
        """Test PyMDP detection based on agent type."""
        builder = PyMDPAgentBuilder()
        
        # ANALYST agents should use PyMDP
        basic_agent_spec.agent_type = AgentType.ANALYST
        basic_agent_spec.source_prompt = "Regular agent"  # No AI keywords
        basic_agent_spec.parameters = {}  # No explicit request
        assert builder._should_create_pymdp_agent(basic_agent_spec)
        
        # Other agent types should not use PyMDP by default
        basic_agent_spec.agent_type = AgentType.CREATIVE
        assert not builder._should_create_pymdp_agent(basic_agent_spec)

    def test_create_gmn_generation_prompt(self, pymdp_agent_spec):
        """Test GMN generation prompt creation."""
        builder = PyMDPAgentBuilder()
        
        prompt = builder._create_gmn_generation_prompt(pymdp_agent_spec)
        
        assert "GMN" in prompt
        assert "Active Inference" in prompt
        assert pymdp_agent_spec.name in prompt
        assert pymdp_agent_spec.agent_type.value in prompt
        assert pymdp_agent_spec.source_prompt in prompt

    @patch('agents.creation.pymdp_agent_builder.get_provider_factory')
    async def test_generate_gmn_specification_success(self, mock_factory, pymdp_agent_spec):
        """Test successful GMN specification generation."""
        builder = PyMDPAgentBuilder()
        
        # Mock LLM response with valid GMN
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
        
        gmn_spec = await builder._generate_gmn_specification(pymdp_agent_spec)
        
        assert gmn_spec is not None
        assert "num_states" in gmn_spec
        assert "A" in gmn_spec
        assert "B" in gmn_spec
        assert builder._metrics["gmn_generations"] == 1

    @patch('agents.creation.pymdp_agent_builder.get_provider_factory')
    async def test_generate_gmn_specification_failure(self, mock_factory, pymdp_agent_spec):
        """Test GMN specification generation failure."""
        builder = PyMDPAgentBuilder()
        
        # Mock LLM failure
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "Invalid response"
        mock_factory.return_value.create_configured_provider.return_value = mock_llm
        
        gmn_spec = await builder._generate_gmn_specification(pymdp_agent_spec)
        
        assert gmn_spec is None

    @patch('agents.creation.pymdp_agent_builder.get_db')
    @patch('agents.creation.pymdp_agent_builder.get_provider_factory')
    async def test_build_pymdp_agent_success(self, mock_factory, mock_db, pymdp_agent_spec):
        """Test successful PyMDP agent building."""
        builder = PyMDPAgentBuilder()
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value = [mock_session]
        
        # Mock successful GMN generation
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
        
        # Mock database agent
        from database.models import Agent, AgentStatus
        mock_agent = Agent(
            id=1,
            name=pymdp_agent_spec.name,
            agent_type=pymdp_agent_spec.agent_type,
            status=AgentStatus.ACTIVE
        )
        mock_session.refresh = Mock(side_effect=lambda x: setattr(x, 'id', 1))
        
        agent = await builder._build_pymdp_agent(pymdp_agent_spec)
        
        # Verify PyMDP agent was created
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert builder._metrics["pymdp_agents_created"] == 1

    @patch('agents.creation.pymdp_agent_builder.get_db')
    async def test_build_traditional_agent(self, mock_db, basic_agent_spec):
        """Test traditional agent building."""
        builder = PyMDPAgentBuilder()
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value = [mock_session]
        mock_session.refresh = Mock(side_effect=lambda x: setattr(x, 'id', 1))
        
        agent = await builder._build_traditional_agent(basic_agent_spec)
        
        # Verify traditional agent was created
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert builder._metrics["traditional_agents_created"] == 1

    @patch('agents.creation.pymdp_agent_builder.get_db')
    @patch('agents.creation.pymdp_agent_builder.get_provider_factory')
    async def test_build_agent_with_fallback(self, mock_factory, mock_db, pymdp_agent_spec):
        """Test PyMDP agent building with fallback to traditional."""
        builder = PyMDPAgentBuilder()
        
        # Mock database session
        mock_session = Mock()
        mock_db.return_value = [mock_session]
        mock_session.refresh = Mock(side_effect=lambda x: setattr(x, 'id', 1))
        
        # Mock GMN generation failure
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "Invalid GMN"
        mock_factory.return_value.create_configured_provider.return_value = mock_llm
        
        agent = await builder._build_pymdp_agent(pymdp_agent_spec)
        
        # Should have fallen back to traditional agent
        assert builder._metrics["traditional_agents_created"] == 1
        assert builder._metrics["pymdp_agents_created"] == 0

    async def test_get_metrics(self):
        """Test metrics collection."""
        builder = PyMDPAgentBuilder()
        
        metrics = builder.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "pymdp_agents_created" in metrics
        assert "traditional_agents_created" in metrics
        assert "gmn_generations" in metrics
        assert "pymdp_ratio" in metrics

    def test_gmn_spec_to_text(self):
        """Test GMN specification to text conversion."""
        builder = PyMDPAgentBuilder()
        
        gmn_spec = {
            "num_states": [4],
            "num_obs": [4],
            "num_actions": [4],
            "A": [np.eye(4)],
            "B": [np.zeros((4, 4, 4))],
            "C": [np.zeros(4)],
            "D": [np.ones(4) / 4]
        }
        
        gmn_text = builder._gmn_spec_to_text(gmn_spec)
        
        assert "[nodes]" in gmn_text
        assert "[edges]" in gmn_text
        assert "location: state" in gmn_text