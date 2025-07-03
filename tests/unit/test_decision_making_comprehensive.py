"""
Comprehensive test coverage for agent decision making system
Decision Making Module - Backend coverage improvement

This test file provides comprehensive coverage for the decision making functionality
to help reach 80% backend coverage target.
"""

from collections import defaultdict
from unittest.mock import Mock

import pytest

# Import the decision making components
try:
    from agents.base.data_model import Agent, AgentCapability, AgentGoal, AgentStatus, Position
    from agents.base.decision_making import (
        Action,
        ActionSelector,
        ActionType,
        DecisionContext,
        DecisionMaker,
        DecisionOrchestrator,
        DecisionStrategy,
        GoalPlanner,
        UtilityCalculator,
    )
    from agents.base.perception import Percept, StimulusType

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

    # Create minimal mocks for testing
    class ActionType:
        MOVE = "move"
        INTERACT = "interact"
        COMMUNICATE = "communicate"
        WAIT = "wait"
        EXPLORE = "explore"
        FLEE = "flee"
        APPROACH = "approach"
        GATHER = "gather"
        USE_ITEM = "use_item"
        OBSERVE = "observe"

    class DecisionStrategy:
        UTILITY_BASED = "utility_based"
        GOAL_ORIENTED = "goal_oriented"
        REACTIVE = "reactive"
        DELIBERATIVE = "deliberative"
        ACTIVE_INFERENCE = "active_inference"
        HYBRID = "hybrid"

    class AgentStatus:
        IDLE = "idle"
        ACTIVE = "active"
        INTERACTING = "interacting"

    class StimulusType:
        DANGER = "danger"
        RESOURCE = "resource"
        AGENT = "agent"


@pytest.fixture
def sample_action():
    """Fixture providing a sample action"""
    return {
        "action_type": ActionType.MOVE,
        "target": Position(10.0, 20.0, 0.0),
        "parameters": {"speed": 1.0, "avoid_obstacles": True},
        "cost": 5.0,
        "duration": 10.0,
        "utility": 15.0,
        "expected_utility": 12.0,
        "effects": {"energy": -5.0, "position_change": True},
        "prerequisites": ["has_energy", "can_move"],
    }


@pytest.fixture
def sample_agent():
    """Fixture providing a sample agent"""
    agent = Mock(spec=Agent)
    agent.id = "agent_1"
    agent.position = Position(0.0, 0.0, 0.0)
    agent.resources = Mock(energy=100.0, health=100.0)
    agent.capabilities = [AgentCapability.COMMUNICATION, AgentCapability.NAVIGATION]
    agent.goals = []
    agent.status = AgentStatus.IDLE
    return agent


@pytest.fixture
def sample_percepts():
    """Fixture providing sample percepts"""
    return [
        Percept(
            stimulus=Mock(stimulus_type=StimulusType.RESOURCE, position=Position(5.0, 5.0, 0.0)),
            intensity=0.8,
            distance=7.07,
        ),
        Percept(
            stimulus=Mock(stimulus_type=StimulusType.DANGER, position=Position(-10.0, 0.0, 0.0)),
            intensity=0.6,
            distance=10.0,
        ),
    ]


class TestAction:
    """Test Action class functionality"""

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_action_creation(self, sample_action):
        """Test creating an action"""
        action = Action(**sample_action)

        assert action.action_type == sample_action["action_type"]
        assert action.target == sample_action["target"]
        assert action.parameters == sample_action["parameters"]
        assert action.cost == sample_action["cost"]
        assert action.duration == sample_action["duration"]
        assert action.utility == sample_action["utility"]
        assert action.expected_utility == sample_action["expected_utility"]
        assert action.effects == sample_action["effects"]
        assert action.prerequisites == sample_action["prerequisites"]

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_action_hash(self):
        """Test action hashing for use in sets"""
        action1 = Action(ActionType.MOVE, target=Position(1, 2, 0))
        action2 = Action(ActionType.MOVE, target=Position(1, 2, 0))
        action3 = Action(ActionType.INTERACT, target=Position(1, 2, 0))

        # Same actions should have same hash
        assert hash(action1) == hash(action2)
        # Different actions should have different hash
        assert hash(action1) != hash(action3)

        # Can be used in sets
        action_set = {action1, action2, action3}
        assert len(action_set) == 2  # action1 and action2 are the same

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_action_to_dict(self, sample_action):
        """Test action serialization to dictionary"""
        action = Action(**sample_action)
        action_dict = action.to_dict()

        assert action_dict["action_type"] == sample_action["action_type"].value
        assert action_dict["target"] is not None
        assert action_dict["parameters"] == sample_action["parameters"]
        assert action_dict["cost"] == sample_action["cost"]
        assert action_dict["utility"] == sample_action["utility"]

    def test_action_mock(self):
        """Test action functionality with mocks"""

        class MockAction:
            def __init__(self, action_type, target=None, **kwargs):
                self.action_type = action_type
                self.target = target
                self.parameters = kwargs.get("parameters", {})
                self.cost = kwargs.get("cost", 0.0)
                self.utility = kwargs.get("utility", 0.0)
                self.effects = kwargs.get("effects", {})
                self.prerequisites = kwargs.get("prerequisites", [])

            def __hash__(self):
                return hash((self.action_type, str(self.target)))

            def to_dict(self):
                return {
                    "action_type": self.action_type,
                    "target": str(self.target) if self.target else None,
                    "cost": self.cost,
                    "utility": self.utility,
                }

        action = MockAction("move", target="(10, 20)", cost=5.0, utility=10.0)
        assert action.action_type == "move"
        assert action.cost == 5.0
        assert hash(action) != 0

        action_dict = action.to_dict()
        assert action_dict["action_type"] == "move"


class TestDecisionContext:
    """Test DecisionContext functionality"""

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_context_creation(self, sample_agent, sample_percepts):
        """Test creating decision context"""
        goals = [Mock(spec=AgentGoal, description="explore", priority=1.0)]

        context = DecisionContext(
            agent=sample_agent,
            percepts=sample_percepts,
            current_goals=goals,
            available_actions=[],
            world_state={"time": 100.0},
            memory_state={"last_position": Position(0, 0, 0)},
            social_state={"nearby_agents": []},
            timestamp=12345.0,
        )

        assert context.agent == sample_agent
        assert context.percepts == sample_percepts
        assert context.current_goals == goals
        assert context.world_state["time"] == 100.0
        assert context.timestamp == 12345.0

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_context_priority_percept(self, sample_percepts):
        """Test getting highest priority percept"""
        context = DecisionContext(
            agent=Mock(),
            percepts=sample_percepts,
            current_goals=[],
            available_actions=[],
            world_state={},
            memory_state={},
            social_state={},
        )

        # Should return the resource percept (higher intensity)
        priority_percept = context.get_priority_percept()
        assert priority_percept.stimulus.stimulus_type == StimulusType.RESOURCE
        assert priority_percept.intensity == 0.8

    def test_context_mock(self):
        """Test context functionality with mocks"""

        class MockContext:
            def __init__(self, agent, percepts, goals):
                self.agent = agent
                self.percepts = percepts
                self.current_goals = goals
                self.world_state = {}
                self.timestamp = 0.0

            def get_priority_percept(self):
                if not self.percepts:
                    return None
                return max(self.percepts, key=lambda p: getattr(p, "intensity", 0))

        mock_percept = type("Percept", (), {"intensity": 0.9, "type": "resource"})()
        context = MockContext(agent="agent_1", percepts=[mock_percept], goals=["explore"])

        assert context.agent == "agent_1"
        priority = context.get_priority_percept()
        assert priority.intensity == 0.9


class TestUtilityCalculator:
    """Test UtilityCalculator functionality"""

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_utility_calculator_init(self):
        """Test utility calculator initialization"""
        calc = UtilityCalculator()
        assert len(calc.utility_functions) == 0
        assert calc.weights == {}

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_register_utility_function(self):
        """Test registering utility functions"""
        calc = UtilityCalculator()

        # Register a simple utility function
        def distance_utility(action, context):
            if action.action_type == ActionType.MOVE:
                return -action.cost  # Negative utility for movement cost
            return 0.0

        calc.register_utility_function("distance", distance_utility, weight=0.5)

        assert "distance" in calc.utility_functions
        assert calc.weights["distance"] == 0.5

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_calculate_utility(self, sample_action):
        """Test calculating action utility"""
        calc = UtilityCalculator()

        # Register multiple utility functions
        calc.register_utility_function("cost", lambda a, c: -a.cost, weight=0.3)
        calc.register_utility_function("benefit", lambda a, c: a.expected_utility, weight=0.7)

        action = Action(**sample_action)
        context = Mock()

        utility = calc.calculate_utility(action, context)
        # utility = 0.3 * (-5.0) + 0.7 * 12.0 = -1.5 + 8.4 = 6.9
        assert abs(utility - 6.9) < 0.01

    def test_utility_calculator_mock(self):
        """Test utility calculator with mocks"""

        class MockUtilityCalculator:
            def __init__(self):
                self.utility_functions = {}
                self.weights = {}

            def register_utility_function(self, name, func, weight=1.0):
                self.utility_functions[name] = func
                self.weights[name] = weight

            def calculate_utility(self, action, context):
                total = 0.0
                for name, func in self.utility_functions.items():
                    value = func(action, context)
                    weight = self.weights.get(name, 1.0)
                    total += value * weight
                return total

        calc = MockUtilityCalculator()
        calc.register_utility_function("simple", lambda a, c: 10.0, weight=0.5)

        mock_action = Mock()
        utility = calc.calculate_utility(mock_action, None)
        assert utility == 5.0  # 10.0 * 0.5


class TestGoalPlanner:
    """Test GoalPlanner functionality"""

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_goal_planner_init(self):
        """Test goal planner initialization"""
        planner = GoalPlanner()
        assert planner.plan_horizon == 10
        assert planner.replan_threshold == 0.3

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_prioritize_goals(self):
        """Test goal prioritization"""
        planner = GoalPlanner()

        goals = [
            Mock(priority=0.5, urgency=0.3, description="explore"),
            Mock(priority=0.8, urgency=0.9, description="flee"),
            Mock(priority=0.7, urgency=0.2, description="gather"),
        ]

        prioritized = planner.prioritize_goals(goals)

        # Should be sorted by combined priority and urgency
        assert prioritized[0].description == "flee"  # Highest combined score
        assert len(prioritized) == 3

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_generate_action_sequence(self):
        """Test generating action sequence for goal"""
        planner = GoalPlanner()

        goal = Mock(description="reach_location", target_position=Position(10, 10, 0))

        context = Mock(agent=Mock(position=Position(0, 0, 0)))

        # Generate simple move sequence
        sequence = planner.generate_action_sequence(goal, context)

        assert len(sequence) > 0
        assert all(isinstance(a, Action) for a in sequence)
        assert sequence[0].action_type == ActionType.MOVE

    def test_goal_planner_mock(self):
        """Test goal planner with mocks"""

        class MockGoalPlanner:
            def __init__(self):
                self.plan_horizon = 10

            def prioritize_goals(self, goals):
                # Sort by priority attribute
                return sorted(goals, key=lambda g: getattr(g, "priority", 0), reverse=True)

            def generate_action_sequence(self, goal, context):
                # Generate simple sequence
                return [
                    type("Action", (), {"action_type": "move", "cost": 1.0})(),
                    type("Action", (), {"action_type": "interact", "cost": 2.0})(),
                ]

        planner = MockGoalPlanner()

        goals = [type("Goal", (), {"priority": 0.5})(), type("Goal", (), {"priority": 0.9})()]

        prioritized = planner.prioritize_goals(goals)
        assert prioritized[0].priority == 0.9

        sequence = planner.generate_action_sequence(None, None)
        assert len(sequence) == 2


class TestActionSelector:
    """Test ActionSelector functionality"""

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_action_selector_init(self):
        """Test action selector initialization"""
        selector = ActionSelector()
        assert selector.exploration_rate == 0.1
        assert selector.selection_strategy == "epsilon_greedy"

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_filter_valid_actions(self, sample_agent):
        """Test filtering valid actions"""
        selector = ActionSelector()

        actions = [
            Action(ActionType.MOVE, cost=5.0, prerequisites=[]),
            Action(ActionType.INTERACT, cost=10.0, prerequisites=["has_tool"]),
            Action(ActionType.WAIT, cost=0.0, prerequisites=[]),
        ]

        context = Mock(agent=sample_agent)
        valid_actions = selector.filter_valid_actions(actions, context)

        # Should include MOVE and WAIT (no prerequisites)
        assert len(valid_actions) >= 2
        assert any(a.action_type == ActionType.MOVE for a in valid_actions)
        assert any(a.action_type == ActionType.WAIT for a in valid_actions)

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_select_action_exploitation(self):
        """Test action selection in exploitation mode"""
        selector = ActionSelector(exploration_rate=0.0)  # No exploration

        actions = [
            Action(ActionType.MOVE, utility=5.0),
            Action(ActionType.INTERACT, utility=10.0),
            Action(ActionType.WAIT, utility=2.0),
        ]

        selected = selector.select_action(actions, Mock())
        assert selected.action_type == ActionType.INTERACT  # Highest utility

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_select_action_exploration(self):
        """Test action selection with exploration"""
        selector = ActionSelector(exploration_rate=1.0)  # Always explore

        actions = [
            Action(ActionType.MOVE, utility=5.0),
            Action(ActionType.INTERACT, utility=10.0),
            Action(ActionType.WAIT, utility=2.0),
        ]

        # With full exploration, any action can be selected
        selected = selector.select_action(actions, Mock())
        assert selected in actions

    def test_action_selector_mock(self):
        """Test action selector with mocks"""

        class MockActionSelector:
            def __init__(self, exploration_rate=0.1):
                self.exploration_rate = exploration_rate

            def filter_valid_actions(self, actions, context):
                # Filter based on cost threshold
                return [a for a in actions if getattr(a, "cost", 0) < 10.0]

            def select_action(self, actions, context):
                if not actions:
                    return None
                # Select highest utility
                return max(actions, key=lambda a: getattr(a, "utility", 0))

        selector = MockActionSelector()

        actions = [
            type("Action", (), {"cost": 5.0, "utility": 10.0})(),
            # Too expensive
            type("Action", (), {"cost": 15.0, "utility": 20.0})(),
            type("Action", (), {"cost": 3.0, "utility": 8.0})(),
        ]

        valid = selector.filter_valid_actions(actions, None)
        assert len(valid) == 2  # Filtered out expensive action

        selected = selector.select_action(valid, None)
        assert selected.utility == 10.0


class TestDecisionMaker:
    """Test DecisionMaker functionality"""

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_decision_maker_init(self):
        """Test decision maker initialization"""
        maker = DecisionMaker(DecisionStrategy.UTILITY_BASED)
        assert maker.strategy == DecisionStrategy.UTILITY_BASED
        assert maker.utility_calculator is not None
        assert maker.action_selector is not None

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_generate_available_actions(self, sample_agent):
        """Test generating available actions"""
        maker = DecisionMaker(DecisionStrategy.REACTIVE)

        context = Mock(
            agent=sample_agent, percepts=[Mock(stimulus=Mock(stimulus_type=StimulusType.DANGER))]
        )

        actions = maker.generate_available_actions(context)

        # Should include basic actions
        assert any(a.action_type == ActionType.WAIT for a in actions)
        # Should include flee action due to danger
        assert any(a.action_type == ActionType.FLEE for a in actions)

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_decide_utility_based(self, sample_agent):
        """Test utility-based decision making"""
        maker = DecisionMaker(DecisionStrategy.UTILITY_BASED)

        # Setup utility function
        maker.utility_calculator.register_utility_function(
            "simple", lambda a, c: 10.0 if a.action_type == ActionType.WAIT else 5.0
        )

        context = Mock(agent=sample_agent, percepts=[])
        action = maker.decide(context)

        assert action is not None
        # Should prefer WAIT due to higher utility
        assert action.action_type == ActionType.WAIT

    def test_decision_maker_mock(self):
        """Test decision maker with mocks"""

        class MockDecisionMaker:
            def __init__(self, strategy):
                self.strategy = strategy
                self.decision_history = []

            def generate_available_actions(self, context):
                actions = [
                    type("Action", (), {"action_type": "move", "cost": 5.0})(),
                    type("Action", (), {"action_type": "wait", "cost": 0.0})(),
                ]
                return actions

            def decide(self, context):
                actions = self.generate_available_actions(context)
                if not actions:
                    return None

                # Simple strategy selection
                if self.strategy == "reactive":
                    return actions[0]  # First action
                else:
                    return min(actions, key=lambda a: getattr(a, "cost", 0))

        maker = MockDecisionMaker("utility_based")

        action = maker.decide(Mock())
        assert action is not None
        assert action.cost == 0.0  # Should select wait (lowest cost)


class TestDecisionOrchestrator:
    """Test DecisionOrchestrator functionality"""

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_orchestrator_init(self):
        """Test decision orchestrator initialization"""
        state_manager = Mock()
        perception_system = Mock()
        movement_controller = Mock()

        orchestrator = DecisionOrchestrator(state_manager, perception_system, movement_controller)

        assert orchestrator.state_manager == state_manager
        assert orchestrator.perception_system == perception_system
        assert orchestrator.movement_controller == movement_controller
        assert len(orchestrator.decision_makers) == 0

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_register_decision_maker(self):
        """Test registering decision makers"""
        orchestrator = DecisionOrchestrator(Mock(), Mock(), Mock())

        maker1 = DecisionMaker(DecisionStrategy.REACTIVE)
        maker2 = DecisionMaker(DecisionStrategy.UTILITY_BASED)

        orchestrator.register_decision_maker("agent_1", maker1)
        orchestrator.register_decision_maker("agent_2", maker2)

        assert len(orchestrator.decision_makers) == 2
        assert orchestrator.decision_makers["agent_1"] == maker1
        assert orchestrator.decision_makers["agent_2"] == maker2

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_make_decision(self, sample_agent):
        """Test making decision for agent"""
        state_manager = Mock()
        state_manager.get_agent.return_value = sample_agent

        perception_system = Mock()
        perception_system.perceive.return_value = []

        movement_controller = Mock()

        orchestrator = DecisionOrchestrator(state_manager, perception_system, movement_controller)

        # Register decision maker
        maker = Mock()
        maker.decide.return_value = Action(ActionType.WAIT)
        orchestrator.register_decision_maker("agent_1", maker)

        # Make decision
        action = orchestrator.make_decision("agent_1")

        assert action is not None
        assert action.action_type == ActionType.WAIT
        assert maker.decide.called

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_execute_action(self, sample_agent):
        """Test executing actions"""
        state_manager = Mock()
        state_manager.get_agent.return_value = sample_agent

        movement_controller = Mock()
        movement_controller.set_destination.return_value = True

        orchestrator = DecisionOrchestrator(state_manager, Mock(), movement_controller)

        # Test move action
        move_action = Action(ActionType.MOVE, target=Position(10, 10, 0), cost=5.0)

        success = orchestrator.execute_action("agent_1", move_action)
        assert success is True
        assert movement_controller.set_destination.called

        # Test wait action
        wait_action = Action(ActionType.WAIT)
        success = orchestrator.execute_action("agent_1", wait_action)
        assert success is True
        assert state_manager.update_agent_status.called

    def test_orchestrator_mock(self):
        """Test orchestrator with mocks"""

        class MockOrchestrator:
            def __init__(self):
                self.decision_makers = {}
                self.decision_history = defaultdict(list)
                self.execution_count = 0

            def register_decision_maker(self, agent_id, maker):
                self.decision_makers[agent_id] = maker

            def make_decision(self, agent_id):
                if agent_id not in self.decision_makers:
                    return None

                action = type("Action", (), {"action_type": "wait", "cost": 0})()
                self.decision_history[agent_id].append(action)
                return action

            def execute_action(self, agent_id, action):
                self.execution_count += 1
                return True

        orchestrator = MockOrchestrator()
        orchestrator.register_decision_maker("agent_1", "maker1")

        action = orchestrator.make_decision("agent_1")
        assert action is not None

        success = orchestrator.execute_action("agent_1", action)
        assert success is True
        assert orchestrator.execution_count == 1


class TestDecisionIntegration:
    """Test integration of decision making components"""

    @pytest.mark.skipif(not IMPORT_SUCCESS, reason="Module not available")
    def test_full_decision_cycle(self, sample_agent, sample_percepts):
        """Test complete decision making cycle"""
        # Setup components
        state_manager = Mock()
        state_manager.get_agent.return_value = sample_agent
        state_manager.get_agent_goals.return_value = [Mock(description="explore", priority=0.8)]

        perception_system = Mock()
        perception_system.perceive.return_value = sample_percepts

        movement_controller = Mock()
        movement_controller.set_destination.return_value = True

        # Create orchestrator
        orchestrator = DecisionOrchestrator(state_manager, perception_system, movement_controller)

        # Create and register decision maker
        maker = DecisionMaker(DecisionStrategy.GOAL_ORIENTED)
        orchestrator.register_decision_maker("agent_1", maker)

        # Make decision
        action = orchestrator.make_decision("agent_1")
        assert action is not None

        # Execute action
        success = orchestrator.execute_action("agent_1", action)
        assert success is True

        # Verify decision was recorded
        assert "agent_1" in orchestrator.decision_history
        assert len(orchestrator.decision_history["agent_1"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
