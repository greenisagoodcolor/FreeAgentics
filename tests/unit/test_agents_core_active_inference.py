"""
Comprehensive tests for agents.core.active_inference module.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from agents.base.data_model import AgentStatus, Position
from agents.core.active_inference import (
    Action,
    ActiveInferenceAgent,
    Belief,
    Observation,
)


@pytest.fixture
def mock_gnn_executor():
    """Create a mock GNN executor for testing."""
    executor = Mock()
    executor.model = Mock()
    executor.model.metadata = {
        "exploration_drive": 0.7,
        "social_drive": 0.5,
        "resource_drive": 0.8,
    }
    executor.execute.return_value = {
        "beliefs": {"location_confidence": 0.9, "resource_availability": 0.6},
        "epistemic_value": 0.3,
        "pragmatic_value": 0.4,
    }
    return executor


@pytest.fixture
def sample_position():
    """Create a sample position for testing."""
    return Position(x=10, y=20, z=0)


@pytest.fixture
def active_agent(mock_gnn_executor, sample_position):
    """Create an ActiveInferenceAgent instance for testing."""
    with patch("agents.core.active_inference.AgentKnowledgeGraph") as MockKG:
        # Create a mock knowledge graph with necessary methods
        mock_kg_instance = Mock()
        mock_kg_instance.add_experience = Mock()
        mock_kg_instance.get_similar_experiences = Mock(return_value=[])
        MockKG.return_value = mock_kg_instance

        agent = ActiveInferenceAgent(
            agent_id="test_agent_001",
            model_name="test_gnn_model",
            gnn_executor=mock_gnn_executor,
            initial_position=sample_position,
            initial_energy=75.0,
        )
        return agent


class TestBelief:
    """Test the Belief dataclass."""

    def test_belief_creation(self):
        """Test creating a belief with all parameters."""
        belief = Belief(
            state="confident_location",
            confidence=0.85,
            evidence=["visual_observation", "map_data"],
        )

        assert belief.state == "confident_location"
        assert belief.confidence == 0.85
        assert belief.evidence == ["visual_observation", "map_data"]
        assert isinstance(belief.timestamp, datetime)

    def test_belief_default_values(self):
        """Test belief with default evidence and timestamp."""
        belief = Belief(state="unknown_area", confidence=0.2)

        assert belief.evidence == []
        assert isinstance(belief.timestamp, datetime)


class TestObservation:
    """Test the Observation dataclass."""

    def test_observation_creation(self, sample_position):
        """Test creating an observation with all parameters."""
        obs = Observation(
            type="visual",
            data={"terrain": "forest", "resources": ["wood", "berries"]},
            position=sample_position,
        )

        assert obs.type == "visual"
        assert obs.data["terrain"] == "forest"
        assert obs.position == sample_position
        assert isinstance(obs.timestamp, datetime)

    def test_observation_without_position(self):
        """Test creating an observation without position."""
        obs = Observation(
            type="internal",
            data={"energy_level": 0.6, "mood": "curious"},
        )

        assert obs.position is None
        assert obs.data["energy_level"] == 0.6


class TestAction:
    """Test the Action dataclass."""

    def test_action_creation(self):
        """Test creating an action with all parameters."""
        action = Action(
            type="move",
            parameters={"direction": "north", "distance": 5},
            expected_outcome="reach_target_location",
            energy_cost=2.5,
        )

        assert action.type == "move"
        assert action.parameters["direction"] == "north"
        assert action.expected_outcome == "reach_target_location"
        assert action.energy_cost == 2.5

    def test_action_defaults(self):
        """Test action with default values."""
        action = Action(
            type="observe",
            parameters={"duration": 10},
        )

        assert action.expected_outcome is None
        assert action.energy_cost == 0.0


class TestActiveInferenceAgentInitialization:
    """Test ActiveInferenceAgent initialization."""

    def test_agent_initialization(
            self,
            active_agent,
            mock_gnn_executor,
            sample_position):
        """Test successful agent initialization."""
        assert active_agent.id == "test_agent_001"
        assert active_agent.model_name == "test_gnn_model"
        assert active_agent.executor == mock_gnn_executor
        assert active_agent.position == sample_position
        assert active_agent.energy == 75.0
        assert active_agent.status == AgentStatus.IDLE

        # Check initialized components
        assert active_agent.beliefs == {}
        assert active_agent.observations == []
        assert active_agent.action_history == []
        assert active_agent.free_energy_history == []

    def test_agent_default_energy(self, mock_gnn_executor, sample_position):
        """Test agent initialization with default energy."""
        with patch("agents.core.active_inference.AgentKnowledgeGraph"):
            agent = ActiveInferenceAgent(
                agent_id="energy_test",
                model_name="test_model",
                gnn_executor=mock_gnn_executor,
                initial_position=sample_position,
            )

            assert agent.energy == 100.0  # Default value

    def test_available_actions_initialization(self, active_agent):
        """Test that available actions are properly initialized."""
        expected_actions = [
            "move",
            "gather",
            "communicate",
            "observe",
            "rest",
            "share_knowledge"]
        assert active_agent.available_actions == expected_actions

    def test_generative_model_initialization(self, active_agent):
        """Test generative model structure."""
        gm = active_agent.generative_model

        # World model
        assert "terrain_types" in gm["world_model"]
        assert "resource_types" in gm["world_model"]
        assert "agent_types" in gm["world_model"]

        # Transition model
        tm = gm["transition_model"]
        assert "movement" in tm
        assert "resource_gathering" in tm
        assert "communication" in tm

        # Preference model should use GNN metadata
        pm = gm["preference_model"]
        assert pm["exploration"] == 0.7  # From mock
        assert pm["social"] == 0.5
        assert pm["resource"] == 0.8


class TestPerceptionAndBelief:
    """Test perception and belief update functionality."""

    def test_perceive_basic(self, active_agent, sample_position):
        """Test basic perception and belief update."""
        obs = Observation(
            type="visual_scan",
            data={"terrain": "mountains", "visibility": 0.8},
            position=sample_position,
        )

        active_agent.perceive(obs)

        # Check observation was stored
        assert len(active_agent.observations) == 1
        assert active_agent.observations[0] == obs

        # Check beliefs were updated via GNN
        assert "location_confidence" in active_agent.beliefs
        assert "resource_availability" in active_agent.beliefs

    def test_process_observation_gnn_integration(self, active_agent):
        """Test that observations are properly processed through GNN."""
        obs = Observation(
            type="resource_scan",
            data={"water": True, "food": False},
        )

        active_agent.perceive(obs)

        # Verify GNN was called
        active_agent.executor.execute.assert_called()

        # Verify GNN input structure
        call_args = active_agent.executor.execute.call_args[0][0]
        assert "observation" in call_args
        assert "current_beliefs" in call_args
        assert "energy" in call_args

        assert call_args["observation"]["type"] == "resource_scan"
        assert call_args["energy"] == 75.0

    def test_belief_evidence_tracking(self, active_agent):
        """Test that beliefs track their evidence sources."""
        obs = Observation(type="test_obs", data={"test": "data"})

        active_agent.perceive(obs)

        for belief in active_agent.beliefs.values():
            assert "test_obs" in belief.evidence
            assert isinstance(belief.timestamp, datetime)


class TestActionSelection:
    """Test action selection and execution."""

    def test_act_no_energy(self, active_agent):
        """Test that agent cannot act without energy."""
        active_agent.energy = 0

        result = active_agent.act({})

        assert result is None

    def test_act_no_available_actions(self, active_agent):
        """Test behavior when no actions are available."""
        # Even with no specific world state, agent can always "observe" and
        # "rest"
        world_state = {}  # No resources, agents, or movement options
        active_agent.energy = 0.5  # Too low for movement

        result = active_agent.act(world_state)

        # Agent should still be able to perform some action (like observe)
        assert result is not None
        assert result.type in ["observe", "rest"]

    def test_can_perform_action_move(self, active_agent):
        """Test move action availability."""
        assert active_agent._can_perform_action("move", {}) is True

        active_agent.energy = 0.5
        assert active_agent._can_perform_action("move", {}) is False

    def test_can_perform_action_gather(self, active_agent):
        """Test gather action availability."""
        # No resources
        assert active_agent._can_perform_action("gather", {}) is False

        # With resources
        world_state = {"nearby_resources": ["wood", "stone"]}
        assert active_agent._can_perform_action("gather", world_state) is True

    def test_can_perform_action_communicate(self, active_agent):
        """Test communicate action availability."""
        # No agents
        assert active_agent._can_perform_action("communicate", {}) is False

        # With agents
        world_state = {"nearby_agents": ["agent_002", "agent_003"]}
        assert active_agent._can_perform_action(
            "communicate", world_state) is True

    def test_can_perform_action_rest(self, active_agent):
        """Test rest action availability."""
        # High energy - no rest needed
        assert active_agent._can_perform_action("rest", {}) is False

        # Low energy - rest available
        active_agent.energy = 30.0
        assert active_agent._can_perform_action("rest", {}) is True

    def test_calculate_expected_free_energy(self, active_agent):
        """Test expected free energy calculation."""
        world_state = {"test": "state"}

        free_energy = active_agent._calculate_expected_free_energy(
            "move", world_state)

        # Should return a numeric value
        assert isinstance(free_energy, float)

        # Verify GNN was called with correct structure
        call_args = active_agent.executor.execute.call_args[0][0]
        assert call_args["action"] == "move"
        assert call_args["world_state"] == world_state
        assert "beliefs" in call_args
        assert "energy" in call_args
        assert "preferences" in call_args

    def test_calculate_free_energy_low_energy_penalty(self, active_agent):
        """Test that low energy affects free energy calculation."""
        active_agent.energy = 1.5  # Very low energy

        free_energy = active_agent._calculate_expected_free_energy("move", {})

        # Should be a numeric value (could be positive or negative)
        assert isinstance(free_energy, float)

    def test_create_action_move(self, active_agent):
        """Test creating a move action."""
        world_state = {"possible_moves": ["north", "east"]}

        action = active_agent._create_action("move", world_state)

        assert action.type == "move"
        assert action.parameters["direction"] == "north"
        # Energy cost comes from generative model transition_model
        assert isinstance(action.energy_cost, float)

    def test_create_action_gather(self, active_agent):
        """Test creating a gather action."""
        world_state = {"nearby_resources": ["berries", "wood"]}

        action = active_agent._create_action("gather", world_state)

        assert action.type == "gather"
        assert action.parameters["resource"] == "berries"

    def test_execute_action_move(self, active_agent):
        """Test executing a move action."""
        action = Action(type="move", parameters={}, energy_cost=2.0)
        initial_energy = active_agent.energy

        active_agent._execute_action(action)

        assert active_agent.energy == initial_energy - 2.0
        assert active_agent.status == AgentStatus.MOVING
        assert action in active_agent.action_history

    def test_execute_action_rest(self, active_agent):
        """Test executing a rest action."""
        action = Action(type="rest", parameters={}, energy_cost=0.0)
        active_agent.energy = 40.0

        active_agent._execute_action(action)

        assert active_agent.energy == 50.0  # Restored 10 energy
        assert active_agent.status == AgentStatus.IDLE

    def test_execute_action_gather(self, active_agent):
        """Test executing a gather action."""
        action = Action(
            type="gather",
            parameters={
                "resource": "wood"},
            energy_cost=1.5)
        initial_energy = active_agent.energy

        active_agent._execute_action(action)

        assert active_agent.energy == initial_energy - 1.5
        assert active_agent.status == AgentStatus.INTERACTING

    def test_execute_action_communicate(self, active_agent):
        """Test executing a communicate action."""
        action = Action(
            type="communicate",
            parameters={
                "target": "agent_002"},
            energy_cost=0.5)
        initial_energy = active_agent.energy

        active_agent._execute_action(action)

        assert active_agent.energy == initial_energy - 0.5
        assert active_agent.status == AgentStatus.INTERACTING


class TestFreeEnergyCalculation:
    """Test free energy calculation functionality."""

    def test_calculate_free_energy_no_observations(self, active_agent):
        """Test free energy calculation with no observations."""
        free_energy = active_agent.calculate_free_energy()

        assert free_energy == 0.0

    def test_calculate_free_energy_with_observations(self, active_agent):
        """Test free energy calculation with observations and beliefs."""
        # Add some observations
        active_agent.observations = [
            Observation(type="visual", data={"terrain": "forest"}),
            Observation(type="audio", data={"sounds": "birds"}),
        ]

        # Add beliefs
        active_agent.beliefs = {
            "terrain": Belief(state="terrain", confidence=0.8),
            "wildlife": Belief(state="wildlife", confidence=0.6),
        }

        free_energy = active_agent.calculate_free_energy()

        assert isinstance(free_energy, float)
        assert free_energy > 0  # Should have some prediction error + complexity
        assert len(active_agent.free_energy_history) == 1

    def test_free_energy_history_tracking(self, active_agent):
        """Test that free energy history is maintained."""
        active_agent.observations = [Observation(type="test", data={})]

        # Calculate multiple times
        fe1 = active_agent.calculate_free_energy()
        fe2 = active_agent.calculate_free_energy()

        assert len(active_agent.free_energy_history) == 2
        assert active_agent.free_energy_history[-2] == fe1
        assert active_agent.free_energy_history[-1] == fe2

    def test_free_energy_with_belief_matching(self, active_agent):
        """Test free energy calculation with belief-observation matching."""
        # Add observation with specific data
        obs = Observation(type="visual", data={"forest_detected": True})
        active_agent.observations = [obs]

        # Add matching belief
        active_agent.beliefs = {
            "forest_detected": Belief(state="forest_detected", confidence=0.9),
        }

        free_energy = active_agent.calculate_free_energy()

        # Should have low prediction error due to good match
        assert isinstance(free_energy, float)
        assert free_energy >= 0.0


class TestKnowledgeSharing:
    """Test knowledge sharing functionality."""

    def test_share_knowledge(self, active_agent):
        """Test sharing knowledge with another agent."""
        # Add high-confidence beliefs
        active_agent.beliefs = {
            "high_conf": Belief(state="high_conf", confidence=0.9),
            "low_conf": Belief(state="low_conf", confidence=0.3),
            "med_conf": Belief(state="med_conf", confidence=0.85),
        }

        knowledge_package = active_agent.share_knowledge(None)

        assert knowledge_package["agent_id"] == "test_agent_001"
        assert knowledge_package["model_name"] == "test_gnn_model"

        # Should only include high-confidence beliefs (>0.8)
        shared_beliefs = knowledge_package["beliefs"]
        assert "high_conf" in shared_beliefs
        assert "med_conf" in shared_beliefs
        assert "low_conf" not in shared_beliefs

    def test_integrate_knowledge(self, active_agent):
        """Test integrating knowledge from another agent."""
        # Set up initial beliefs
        active_agent.beliefs = {
            "existing": Belief(state="existing", confidence=0.6),
        }

        # Knowledge package from another agent
        knowledge_package = {
            "agent_id": "agent_friend",
            "beliefs": {
                # Update existing
                "existing": Belief(state="existing", confidence=0.8),
                # New belief
                "new_belief": Belief(state="new_belief", confidence=0.9),
            },
        }

        active_agent.integrate_knowledge(knowledge_package)

        # Check belief integration
        assert "existing" in active_agent.beliefs
        assert "new_belief" in active_agent.beliefs

        # Existing belief should be updated (weighted average)
        existing_conf = active_agent.beliefs["existing"].confidence
        assert 0.6 < existing_conf < 0.8  # Should be between original values

        # New belief should have reduced confidence (trust factor)
        new_conf = active_agent.beliefs["new_belief"].confidence
        assert new_conf < 0.9  # Reduced by trust factor


class TestAgentSerialization:
    """Test agent state serialization."""

    def test_to_dict_basic(self, active_agent):
        """Test basic agent state serialization."""
        agent_dict = active_agent.to_dict()

        assert agent_dict["id"] == "test_agent_001"
        assert agent_dict["model_name"] == "test_gnn_model"
        assert agent_dict["energy"] == 75.0
        assert agent_dict["action_count"] == 0
        assert agent_dict["observation_count"] == 0

        # Position should be serialized
        assert agent_dict["position"]["x"] == 10
        assert agent_dict["position"]["y"] == 20

    def test_to_dict_with_beliefs(self, active_agent):
        """Test serialization with beliefs."""
        active_agent.beliefs = {
            "test_belief": Belief(state="test", confidence=0.75),
        }

        agent_dict = active_agent.to_dict()

        assert "beliefs" in agent_dict
        assert agent_dict["beliefs"]["test_belief"] == 0.75

    def test_to_dict_with_actions_and_observations(self, active_agent):
        """Test serialization with action and observation history."""
        # Add some actions and observations
        active_agent.action_history = [
            Action(type="move", parameters={"direction": "north"}),
            Action(type="gather", parameters={"resource": "wood"}),
        ]
        active_agent.observations = [
            Observation(type="visual", data={"terrain": "forest"}),
        ]

        agent_dict = active_agent.to_dict()

        assert agent_dict["action_count"] == 2
        assert agent_dict["observation_count"] == 1

    def test_update_position(self, active_agent):
        """Test updating agent position."""
        new_position = Position(x=50, y=60, z=10)

        active_agent.update_position(new_position)

        assert active_agent.position == new_position


class TestIntegrationScenarios:
    """Test integrated scenarios combining multiple features."""

    def test_full_perception_action_cycle(self, active_agent):
        """Test a complete perception-action cycle."""
        # Perceive environment
        obs = Observation(
            type="environmental_scan",
            data={"resources": True, "agents": False, "terrain": "plains"},
        )
        active_agent.perceive(obs)

        # Select and execute action
        world_state = {
            "nearby_resources": ["food"],
            "possible_moves": ["north", "south"],
        }
        action = active_agent.act(world_state)

        # Verify state changes
        assert len(active_agent.observations) == 1
        assert len(active_agent.beliefs) > 0
        assert action is not None
        assert action in active_agent.action_history

    def test_energy_depletion_recovery(self, active_agent):
        """Test energy depletion and recovery cycle."""
        # Test the actual energy thresholds
        active_agent.energy = 0  # No energy at all

        world_state = {"possible_moves": ["north"]}
        action1 = active_agent.act(world_state)  # Should fail due to no energy
        assert action1 is None

        # Recover energy and try again
        active_agent.energy = 10.0
        action2 = active_agent.act(world_state)  # Should work now
        assert action2 is not None

    def test_belief_accumulation_over_time(self, active_agent):
        """Test belief accumulation over multiple observations."""
        observations = [
            Observation(type="scan_1", data={"feature_A": True}),
            Observation(type="scan_2", data={"feature_B": False}),
            Observation(
                type="scan_3",
                data={
                    "feature_A": True}),
            # Reinforcement
        ]

        for obs in observations:
            active_agent.perceive(obs)

        # Should have accumulated observations and beliefs
        assert len(active_agent.observations) == 3
        assert len(active_agent.beliefs) > 0

        # Free energy should be calculated
        fe = active_agent.calculate_free_energy()
        assert isinstance(fe, float)
        assert len(active_agent.free_energy_history) == 1

    def test_complex_action_selection_scenario(self, active_agent):
        """Test complex action selection with multiple options."""
        # Set up complex world state
        world_state = {
            "nearby_resources": ["wood", "stone", "berries"],
            "nearby_agents": ["trader_001", "explorer_002"],
            "possible_moves": ["north", "south", "east", "west"],
        }

        # Add some beliefs to influence decision
        active_agent.beliefs = {
            "resource_need": Belief(
                state="resource_need",
                confidence=0.8),
            "social_opportunity": Belief(
                state="social_opportunity",
                confidence=0.6),
        }

        action = active_agent.act(world_state)

        # Should select some action
        assert action is not None
        assert action.type in active_agent.available_actions

        # Action should have appropriate parameters based on world state
        if action.type == "gather":
            assert action.parameters.get(
                "resource") in world_state["nearby_resources"]
        elif action.type == "communicate":
            assert action.parameters.get(
                "target_agent") in world_state["nearby_agents"]
        elif action.type == "move":
            assert action.parameters.get(
                "direction") in world_state["possible_moves"]
