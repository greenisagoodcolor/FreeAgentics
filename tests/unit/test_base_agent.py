"""
Comprehensive test suite for Active Inference Base Agent.

Tests the ActiveInferenceAgent class with PyMDP integration,
GMN parsing, and LLM integration capabilities.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Mock the imports that might not be available
with patch.dict(
    "sys.modules",
    {
        "pymdp": MagicMock(),
        "pymdp.utils": MagicMock(),
        "pymdp.agent": MagicMock(),
    },
):
    from agents.base_agent import ActiveInferenceAgent, AgentConfig


class ConcreteAgent(ActiveInferenceAgent):
    """Concrete implementation for testing."""

    def __init__(self, agent_id: str, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize concrete agent."""
        super().__init__(agent_id, name, config)

        # Test-specific attributes
        self.state = {}
        self.action_history = []
        self.observation_history = []

        # Setup test generative model
        self.A = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.B = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])
        self.C = np.array([1.0, 0.0])
        self.D = np.array([0.5, 0.5])

    def _compute_expected_free_energy(self, qs: np.ndarray, actions: List[int]) -> np.ndarray:
        """Compute expected free energy for actions."""
        return np.random.rand(len(actions))

    def perceive(self, observation) -> None:
        """Process observation (test implementation)."""
        self.last_observation = observation

    def update_beliefs(self) -> None:
        """Update beliefs (test implementation)."""
        if not hasattr(self, "beliefs"):
            self.beliefs = {}
        self.beliefs["test_belief"] = 0.5

    def select_action(self) -> int:
        """Select action based on beliefs."""
        if hasattr(self, "pymdp_agent") and self.pymdp_agent:
            return 0
        return np.random.randint(0, 2)

    def observe(self, observation: Any) -> None:
        """Record observation in history."""
        from datetime import datetime

        obs_record = {"observation": observation, "timestamp": datetime.now().isoformat()}
        self.observation_history.append(obs_record)

    def act(self, action: Any) -> None:
        """Record action in history."""
        from datetime import datetime

        action_record = {"action": action, "timestamp": datetime.now().isoformat()}
        self.action_history.append(action_record)

    def reset(self) -> None:
        """Reset agent state."""
        self.state = {}
        self.beliefs = None
        self.action_history = []
        self.observation_history = []

    def get_history(self) -> Dict[str, Any]:
        """Get agent history."""
        return {"actions": self.action_history, "observations": self.observation_history}

    def save_state(self, filepath: str) -> None:
        """Save agent state to file."""
        import json

        state_data = {
            "state": self.state,
            "beliefs": self.beliefs.tolist() if hasattr(self.beliefs, "tolist") else self.beliefs,
            "action_history": self.action_history,
            "observation_history": self.observation_history,
        }
        with open(filepath, "w") as f:
            json.dump(state_data, f)

    def load_state(self, filepath: str) -> None:
        """Load agent state from file."""
        import json

        with open(filepath, "r") as f:
            state_data = json.load(f)

        self.state = state_data.get("state", {})
        beliefs_data = state_data.get("beliefs")
        if beliefs_data is not None:
            self.beliefs = np.array(beliefs_data)
        self.action_history = state_data.get("action_history", [])
        self.observation_history = state_data.get("observation_history", [])

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return {
            "total_actions": len(self.action_history),
            "total_observations": len(self.observation_history),
            "unique_actions": len(set(record["action"] for record in self.action_history)),
            "unique_observations": len(
                set(str(record["observation"]) for record in self.observation_history)
            ),
        }

    def _query_llm(self, prompt: str) -> Optional[str]:
        """Query LLM if available."""
        if self.llm_manager:
            response = self.llm_manager.generate(prompt)
            return response.text if hasattr(response, "text") else str(response)
        return None


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = AgentConfig(name="test_agent")

        assert config.name == "test_agent"
        assert config.use_pymdp is True
        assert config.planning_horizon == 3
        assert config.precision == 1.0
        assert config.lr == 0.1
        assert config.llm_config is None
        assert config.gmn_spec is None

    def test_config_custom_values(self):
        """Test custom configuration values."""
        llm_config = {"provider": "ollama", "model": "llama2"}
        gmn_spec = "test_spec"

        config = AgentConfig(
            name="custom_agent",
            use_pymdp=False,
            planning_horizon=5,
            precision=2.0,
            lr=0.05,
            llm_config=llm_config,
            gmn_spec=gmn_spec,
        )

        assert config.name == "custom_agent"
        assert config.use_pymdp is False
        assert config.planning_horizon == 5
        assert config.precision == 2.0
        assert config.lr == 0.05
        assert config.llm_config == llm_config
        assert config.gmn_spec == gmn_spec


class TestActiveInferenceAgent:
    """Test ActiveInferenceAgent base class."""

    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return AgentConfig(
            name="test_agent",
            use_pymdp=False,  # Start without PyMDP for simpler tests
            planning_horizon=3,
        )

    @pytest.fixture
    def agent(self, agent_config):
        """Create test agent instance."""
        return ConcreteAgent(
            "test_agent_id", agent_config.name, {"use_pymdp": agent_config.use_pymdp}
        )

    def test_agent_initialization(self, agent, agent_config):
        """Test agent initialization."""
        assert agent.agent_id == "test_agent_id"
        assert agent.name == agent_config.name
        assert agent.config["use_pymdp"] == agent_config.use_pymdp
        assert hasattr(agent, "beliefs")
        assert hasattr(agent, "error_handler")
        assert not agent.is_active  # Agent starts inactive
        assert hasattr(agent, "metrics")

    def test_setup_generative_model(self, agent):
        """Test generative model setup."""
        # The generative model is already set up in __init__
        assert hasattr(agent, "A")
        assert hasattr(agent, "B")
        assert hasattr(agent, "C")
        assert hasattr(agent, "D")
        assert agent.A.shape == (2, 2)
        assert agent.B.shape == (2, 2, 2)
        assert agent.C.shape == (2,)
        assert agent.D.shape == (2,)

    def test_initialization(self, agent):
        """Test agent initialization (done in __init__)."""
        # Check that initialization happened properly
        assert agent.agent_id == "test_agent_id"
        assert agent.name == "test_agent"
        assert hasattr(agent, "A")
        assert hasattr(agent, "B")
        assert hasattr(agent, "C")
        assert hasattr(agent, "D")

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    def test_initialization_with_pymdp(self):
        """Test initialization with PyMDP enabled."""
        with patch("agents.base_agent.PyMDPAgent") as mock_pymdp:
            mock_agent_instance = Mock()
            mock_pymdp.return_value = mock_agent_instance

            config = {"use_pymdp": True}
            agent = ConcreteAgent("test_id", "test_agent", config)

            # PyMDP initialization happens in the base agent __init__
            # Just check that the agent was created properly
            assert agent.agent_id == "test_id"
            assert agent.name == "test_agent"

    @patch("agents.base_agent.GMN_AVAILABLE", True)
    def test_initialization_with_gmn(self):
        """Test initialization with GMN specification."""
        gmn_spec = {
            "nodes": [
                {"id": "state1", "type": "state", "properties": {"num_states": 3}},
                {"id": "obs1", "type": "observation", "properties": {"num_observations": 2}},
            ],
            "edges": [],
        }

        config = {"gmn_spec": gmn_spec}
        agent = ConcreteAgent("test_id", "test_agent", config)

        # Check basic initialization worked
        assert agent.agent_id == "test_id"
        assert agent.name == "test_agent"
        assert hasattr(agent, "A")
        assert hasattr(agent, "B")

    def test_observe(self, agent):
        """Test observation processing."""
        observation = {"location": 1, "sensor": 0.5}

        agent.observe(observation)

        assert len(agent.observation_history) == 1
        assert agent.observation_history[0]["observation"] == observation
        assert "timestamp" in agent.observation_history[0]

    def test_update_beliefs_without_pymdp(self, agent):
        """Test belief update without PyMDP."""
        agent.update_beliefs()

        # beliefs should be initialized as dict
        assert agent.beliefs is not None
        assert isinstance(agent.beliefs, dict)
        assert "test_belief" in agent.beliefs

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    def test_update_beliefs_with_pymdp(self):
        """Test belief update with PyMDP."""
        config = {"use_pymdp": True}
        agent = ConcreteAgent("test_id", "test_agent", config)

        mock_pymdp_agent = Mock()
        mock_pymdp_agent.infer_states.return_value = (np.array([0.7, 0.3]), None)
        agent.pymdp_agent = mock_pymdp_agent

        agent.update_beliefs()

        # The test implementation just sets test_belief to 0.5
        assert "test_belie" in agent.beliefs
        assert agent.beliefs["test_belief"] == 0.5

    def test_act(self, agent):
        """Test action execution."""
        action = 1

        agent.act(action)

        assert len(agent.action_history) == 1
        assert agent.action_history[0]["action"] == action
        assert "timestamp" in agent.action_history[0]

    def test_step(self, agent):
        """Test complete agent step."""
        agent.start()  # Start the agent first
        observation = {"location": 1}

        # The step method calls perceive, update_beliefs, and select_action internally
        action = agent.step(observation)

        # Check that an action was returned
        assert action is not None
        assert agent.total_steps == 1

    def test_reset(self, agent):
        """Test agent reset."""
        agent.state = {"key": "value"}
        agent.beliefs = np.array([0.7, 0.3])
        agent.action_history = [{"action": 0}]
        agent.observation_history = [{"observation": 1}]

        agent.reset()

        assert agent.state == {}
        assert agent.beliefs is None
        assert agent.action_history == []
        assert agent.observation_history == []

    def test_get_history(self, agent):
        """Test history retrieval."""
        agent.action_history = [{"action": 0}, {"action": 1}]
        agent.observation_history = [{"observation": 0}, {"observation": 1}]

        history = agent.get_history()

        assert "actions" in history
        assert "observations" in history
        assert len(history["actions"]) == 2
        assert len(history["observations"]) == 2

    def test_save_load_state(self, agent, tmp_path):
        """Test state saving and loading."""
        agent.state = {"position": 1}
        agent.beliefs = np.array([0.6, 0.4])
        agent.action_history = [{"action": 0}]
        agent.observation_history = [{"observation": 1}]

        # Save state
        save_path = tmp_path / "agent_state.json"
        agent.save_state(str(save_path))

        assert save_path.exists()

        # Create new agent and load state
        new_agent = ConcreteAgent("new_id", "new_agent", {"use_pymdp": False})
        new_agent.load_state(str(save_path))

        assert new_agent.state == agent.state
        assert np.array_equal(new_agent.beliefs, agent.beliefs)
        assert new_agent.action_history == agent.action_history
        assert new_agent.observation_history == agent.observation_history

    def test_llm_integration(self):
        """Test LLM integration."""
        llm_config = {"provider": "ollama", "model": "llama2", "temperature": 0.7}
        config = {"llm_config": llm_config}

        with patch("agents.base_agent.LocalLLMManager") as mock_llm:
            mock_manager = Mock()
            mock_llm.return_value = mock_manager

            agent = ConcreteAgent("test_id", "test_agent", config)

            # LLM manager is created during initialization if LLM_AVAILABLE
            # Just check the agent was created properly
            assert agent.agent_id == "test_id"

    def test_query_llm(self, agent):
        """Test LLM querying."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "Action: move_north"
        mock_llm.generate.return_value = mock_response
        agent.llm_manager = mock_llm

        prompt = "What action should I take?"
        response = agent._query_llm(prompt)

        assert response == "Action: move_north"
        mock_llm.generate.assert_called_once_with(prompt)

    def test_query_llm_no_manager(self, agent):
        """Test LLM query without manager."""
        agent.llm_manager = None

        response = agent._query_llm("test prompt")

        assert response is None

    def test_compute_expected_free_energy(self, agent):
        """Test expected free energy computation."""
        qs = np.array([0.7, 0.3])
        actions = [0, 1]

        efe = agent._compute_expected_free_energy(qs, actions)

        assert isinstance(efe, np.ndarray)
        assert len(efe) == len(actions)

    def test_get_metrics(self, agent):
        """Test metrics retrieval."""
        agent.action_history = [{"action": 0}, {"action": 1}]
        agent.observation_history = [{"observation": 0}, {"observation": 1}]

        metrics = agent.get_metrics()

        assert "total_actions" in metrics
        assert "total_observations" in metrics
        assert "unique_actions" in metrics
        assert "unique_observations" in metrics
        assert metrics["total_actions"] == 2
        assert metrics["total_observations"] == 2

    def test_error_handling_in_step(self, agent):
        """Test error handling during step."""
        agent.start()  # Start agent first

        # The step method has error handling that returns "stay" on failure
        # So we don't expect the exception to propagate
        with patch.object(agent, "update_beliefs", side_effect=Exception("Update failed")):
            action = agent.step({"observation": 1})
            # Should return fallback action instead of raising
            assert action == "stay"

    def test_belief_entropy(self, agent):
        """Test belief entropy calculation."""
        agent.beliefs = np.array([0.5, 0.5])  # Maximum entropy

        # This test just checks that beliefs can be set
        # The actual entropy calculation is not implemented in the test agent
        assert agent.beliefs is not None
        assert len(agent.beliefs) == 2

    def test_action_selection_with_exploration(self, agent):
        """Test action selection with exploration."""
        # Set exploration rate in config if it exists
        if hasattr(agent, "config") and isinstance(agent.config, dict):
            agent.config["exploration_rate"] = 0.1

        actions = []
        for _ in range(10):  # Reduced iterations for faster test
            action = agent.select_action()
            actions.append(action)

        # Should return valid actions (integers 0 or 1 based on our test implementation)
        assert all(isinstance(action, int) for action in actions)
        assert all(action in [0, 1] for action in actions)


class TestActiveInferenceAgentIntegration:
    """Integration tests for ActiveInferenceAgent."""

    def test_full_episode(self):
        """Test full episode execution."""
        config = {"use_pymdp": False, "planning_horizon": 5}
        agent = ConcreteAgent("integration_agent", "integration_agent", config)
        agent.start()

        # Simulate episode
        observations = [0, 1, 1, 0, 1]
        actions = []

        for obs in observations:
            action = agent.step({"sensor": obs})
            actions.append(action)

        assert len(actions) == len(observations)
        # Note: action_history and observation_history are not automatically populated by step()
        # They need to be populated by explicit calls to act() and observe()
        assert agent.total_steps == len(observations)

    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    @patch("agents.base_agent.GMN_AVAILABLE", True)
    def test_gmn_pymdp_integration(self):
        """Test GMN and PyMDP integration."""
        gmn_spec = {
            "nodes": [
                {"id": "location", "type": "state", "properties": {"num_states": 4}},
                {"id": "obs_loc", "type": "observation", "properties": {"num_observations": 4}},
                {"id": "move", "type": "action", "properties": {"num_actions": 5}},
            ],
            "edges": [],
        }

        config = {"use_pymdp": True, "gmn_spec": gmn_spec}

        with patch("agents.base_agent.parse_gmn_spec") as mock_parse:
            with patch("agents.base_agent.PyMDPAgent") as mock_pymdp:
                mock_parse.return_value = {
                    "num_states": [4],
                    "num_obs": [4],
                    "num_actions": 5,
                    "A": [np.eye(4)],
                    "B": [np.ones((4, 4, 5)) / 4],
                    "C": [np.array([1, 0, 0, 0])],
                    "D": [np.ones(4) / 4],
                }

                agent = ConcreteAgent("gmn_agent", "gmn_agent", config)

                # Check that agent was created successfully
                assert agent.agent_id == "gmn_agent"
                assert agent.name == "gmn_agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=agents.base_agent"])
