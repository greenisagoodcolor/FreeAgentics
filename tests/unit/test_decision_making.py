"""
Module for FreeAgentics Active Inference implementation.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from agents.base.data_model import (
    Agent,
    AgentCapability,
    AgentGoal,
    AgentResources,
    AgentStatus,
    Position,
    SocialRelationship,
)
from agents.base.decision_making import (
    Action,
    ActionGenerator,
    ActionNode,
    ActionType,
    BehaviorTree,
    ConditionNode,
    DecisionContext,
    DecisionMaker,
    DecisionStrategy,
    DecisionSystem,
    GoalUtility,
    ResourceUtility,
    SafetyUtility,
    SelectorNode,
    SequenceNode,
    SocialUtility,
)
from agents.base.movement import MovementController
from agents.base.perception import Percept, PerceptionSystem, PerceptionType, Stimulus, StimulusType
from agents.base.state_manager import AgentStateManager

"""
Unit tests for agent decision-making framework
Tests all decision-making components including:
- Utility functions
- Decision strategies
- Action generation
- Behavior trees
- Decision execution
"""


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing"""
    agent = Agent(
        agent_id="test_agent_1",
        position=Position(5.0, 5.0, 0.0),
        resources=AgentResources(energy=75.0, health=100.0),
        capabilities={
            AgentCapability.MOVEMENT,
            AgentCapability.PERCEPTION,
            AgentCapability.SOCIAL_INTERACTION,
            AgentCapability.COMMUNICATION,
            AgentCapability.RESOURCE_MANAGEMENT,
        },
    )
    return agent


@pytest.fixture
def sample_percepts():
    """Create sample percepts for testing"""
    return [
        Percept(
            stimulus=Stimulus(
                stimulus_id="threat_1",
                stimulus_type=StimulusType.DANGER,
                position=Position(8.0, 8.0, 0.0),
                intensity=0.8,
            ),
            perception_type=PerceptionType.VISUAL,
            distance=4.24,
            timestamp=datetime.now(),
        ),
        Percept(
            stimulus=Stimulus(
                stimulus_id="agent_2",
                stimulus_type=StimulusType.AGENT,
                position=Position(6.0, 6.0, 0.0),
                intensity=0.6,
                metadata={"agent_id": "agent_2"},
            ),
            perception_type=PerceptionType.VISUAL,
            distance=1.41,
            timestamp=datetime.now(),
        ),
        Percept(
            stimulus=Stimulus(
                stimulus_id="resource_1",
                stimulus_type=StimulusType.OBJECT,
                position=Position(3.0, 3.0, 0.0),
                intensity=0.5,
                metadata={"is_resource": True},
            ),
            perception_type=PerceptionType.VISUAL,
            distance=2.83,
            timestamp=datetime.now(),
        ),
    ]


@pytest.fixture
def sample_actions():
    """Create sample actions for testing"""
    return [
        Action(ActionType.WAIT, cost=0.1),
        Action(ActionType.MOVE, target=Position(10.0, 10.0, 0.0), cost=5.0),
        Action(ActionType.FLEE, cost=10.0),
        Action(ActionType.GATHER, cost=2.0, effects={"energy": 20.0}),
        Action(ActionType.INTERACT, target=Mock(agent_id="agent_2"), cost=1.0),
        Action(ActionType.COMMUNICATE, cost=0.5),
    ]


@pytest.fixture
def decision_context(sample_agent, sample_percepts, sample_actions):
    """Create a decision context for testing"""
    goal = AgentGoal(
        goal_id="goal_1",
        description="Reach target position",
        priority=0.8,
        target_position=Position(20.0, 20.0, 0.0),
    )
    sample_agent.current_goal = goal
    return DecisionContext(
        agent=sample_agent,
        percepts=sample_percepts,
        current_goal=goal,
        available_actions=sample_actions,
        time_pressure=0.2,
    )


class TestAction:
    """Test Action dataclass"""

    def test_action_creation(self) -> None:
        """Test action creation and attributes"""
        action = Action(
            action_type=ActionType.MOVE,
            target=Position(
                10.0,
                10.0,
                0.0),
            cost=5.0,
            duration=2.0)
        assert action.action_type == ActionType.MOVE
        assert action.target.x == 10.0
        assert action.cost == 5.0
        assert action.duration == 2.0
        assert action.expected_utility == 0.0

    def test_action_hash(self) -> None:
        """Test action hashing for use in sets"""
        action1 = Action(ActionType.MOVE, target=Position(10.0, 10.0, 0.0))
        action2 = Action(ActionType.MOVE, target=Position(10.0, 10.0, 0.0))
        action3 = Action(ActionType.WAIT)
        assert hash(action1) == hash(action2)
        assert hash(action1) != hash(action3)
        action_set = {action1, action2, action3}
        assert len(action_set) == 2


class TestDecisionContext:
    """Test DecisionContext functionality"""

    def test_get_perceived_threats(self, decision_context) -> None:
        """Test threat filtering from percepts"""
        threats = decision_context.get_perceived_threats()
        assert len(threats) == 1
        assert threats[0].stimulus.stimulus_id == "threat_1"

    def test_get_perceived_agents(self, decision_context) -> None:
        """Test agent filtering from percepts"""
        agents = decision_context.get_perceived_agents()
        assert len(agents) == 1
        assert agents[0].stimulus.stimulus_id == "agent_2"

    def test_get_perceived_resources(self, decision_context) -> None:
        """Test resource filtering from percepts"""
        resources = decision_context.get_perceived_resources()
        assert len(resources) == 1
        assert resources[0].stimulus.stimulus_id == "resource_1"


class TestUtilityFunctions:
    """Test utility function implementations"""

    def test_safety_utility(self, decision_context) -> None:
        """Test safety utility calculation"""
        utility = SafetyUtility()
        flee_action = Action(ActionType.FLEE, cost=10.0)
        flee_utility = utility.calculate(flee_action, decision_context)
        assert flee_utility > 0
        move_away = Action(ActionType.MOVE, target=Position(1.0, 1.0, 0.0))
        away_utility = utility.calculate(move_away, decision_context)
        assert away_utility > 0
        move_toward = Action(ActionType.MOVE, target=Position(8.0, 8.0, 0.0))
        toward_utility = utility.calculate(move_toward, decision_context)
        assert toward_utility < 0
        wait_action = Action(ActionType.WAIT)
        wait_utility = utility.calculate(wait_action, decision_context)
        assert wait_utility < 0

    def test_goal_utility(self, decision_context) -> None:
        """Test goal utility calculation"""
        utility = GoalUtility()
        move_to_goal = Action(
            ActionType.MOVE, target=Position(
                15.0, 15.0, 0.0))
        goal_utility = utility.calculate(move_to_goal, decision_context)
        assert goal_utility > 0
        move_away = Action(ActionType.MOVE, target=Position(0.0, 0.0, 0.0))
        away_utility = utility.calculate(move_away, decision_context)
        assert away_utility < 0
        no_goal_context = DecisionContext(
            agent=decision_context.agent,
            percepts=[],
            current_goal=None,
            available_actions=[])
        assert utility.calculate(move_to_goal, no_goal_context) == 0

    def test_resource_utility(self, sample_agent, decision_context) -> None:
        """Test resource utility calculation"""
        utility = ResourceUtility()
        sample_agent.resources.energy = 20.0
        gather_action = Action(ActionType.GATHER, cost=2.0)
        gather_utility = utility.calculate(gather_action, decision_context)
        assert gather_utility > 0
        sample_agent.resources.energy = 90.0
        high_energy_utility = utility.calculate(
            gather_action, decision_context)
        assert high_energy_utility < gather_utility
        expensive_action = Action(ActionType.MOVE, cost=50.0)
        sample_agent.resources.energy = 10.0
        expensive_utility = utility.calculate(
            expensive_action, decision_context)
        assert expensive_utility < 0

    def test_social_utility(self, decision_context) -> None:
        """Test social utility calculation"""
        utility = SocialUtility()
        comm_action = Action(ActionType.COMMUNICATE)
        comm_utility = utility.calculate(comm_action, decision_context)
        assert comm_utility > 0
        agent_target = Mock(agent_id="agent_2")
        interact_action = Action(ActionType.INTERACT, target=agent_target)
        relationship = SocialRelationship(
            target_agent_id="agent_2",
            relationship_type="friend",
            trust_level=0.8)
        decision_context.agent.relationships["agent_2"] = relationship
        interact_utility = utility.calculate(interact_action, decision_context)
        assert interact_utility > 0


class TestDecisionMaker:
    """Test DecisionMaker strategies"""

    def test_utility_based_decision(self, decision_context) -> None:
        """Test utility-based decision making"""
        maker = DecisionMaker(DecisionStrategy.UTILITY_BASED)
        decision = maker.decide(decision_context)
        assert decision is not None
        assert isinstance(decision, Action)
        assert decision.expected_utility != 0.0

    def test_goal_oriented_decision(self, decision_context) -> None:
        """Test goal-oriented decision making"""
        maker = DecisionMaker(DecisionStrategy.GOAL_ORIENTED)
        move_to_goal = Action(
            ActionType.MOVE, target=Position(
                15.0, 15.0, 0.0))
        decision_context.available_actions.append(move_to_goal)
        decision = maker.decide(decision_context)
        assert decision is not None
        assert decision.action_type == ActionType.MOVE

    def test_reactive_decision(self, decision_context) -> None:
        """Test reactive decision making"""
        maker = DecisionMaker(DecisionStrategy.REACTIVE)
        decision = maker.decide(decision_context)
        assert decision is not None
        assert decision.action_type == ActionType.FLEE
        decision_context.agent.resources.energy = 15.0
        decision_context.percepts = []
        decision = maker.decide(decision_context)
        assert decision.action_type == ActionType.GATHER

    def test_hybrid_decision(self, decision_context) -> None:
        """Test hybrid decision making"""
        maker = DecisionMaker(DecisionStrategy.HYBRID)
        decision_context.time_pressure = 0.9
        decision = maker.decide(decision_context)
        assert decision is not None
        assert decision.action_type == ActionType.FLEE

    def test_active_inference_decision(self, decision_context) -> None:
        """Test active inference decision (fallback when not available)"""
        maker = DecisionMaker(DecisionStrategy.ACTIVE_INFERENCE)
        decision = maker.decide(decision_context)
        assert decision is not None
        assert isinstance(decision, Action)


class TestBehaviorTree:
    """Test behavior tree components"""

    def test_sequence_node(self, decision_context) -> None:
        """Test sequence node execution"""
        actions_executed = []

        def create_action(action_type):
            def generator(ctx):
                actions_executed.append(action_type)
                return Action(action_type)

            return ActionNode(generator)

        sequence = SequenceNode(
            [
                create_action(ActionType.WAIT),
                create_action(ActionType.MOVE),
                create_action(ActionType.INTERACT),
            ]
        )
        result = sequence.execute(decision_context)
        assert result is not None
        assert len(actions_executed) == 2
        assert result.action_type == ActionType.MOVE

    def test_selector_node(self, decision_context) -> None:
        """Test selector node execution"""

        def failing_generator(ctx):
            return None

        def success_generator(ctx):
            return Action(ActionType.WAIT)

        selector = SelectorNode(
            [
                ActionNode(failing_generator),
                ActionNode(failing_generator),
                ActionNode(success_generator),
            ]
        )
        result = selector.execute(decision_context)
        assert result is not None
        assert result.action_type == ActionType.WAIT

    def test_condition_node(self, decision_context) -> None:
        """Test condition node execution"""

        def low_energy(ctx):
            return ctx.agent.resources.energy < 50

        def gather_generator(ctx):
            return Action(ActionType.GATHER)

        def explore_generator(ctx):
            return Action(ActionType.EXPLORE)

        condition = ConditionNode(
            low_energy,
            ActionNode(gather_generator),
            ActionNode(explore_generator))
        decision_context.agent.resources.energy = 30
        result = condition.execute(decision_context)
        assert result.action_type == ActionType.GATHER
        decision_context.agent.resources.energy = 80
        result = condition.execute(decision_context)
        assert result.action_type == ActionType.EXPLORE

    def test_behavior_tree_execution(self, decision_context) -> None:
        """Test complete behavior tree"""
        tree = BehaviorTree()

        def always_true(ctx):
            return True

        def wait_generator(ctx):
            return Action(ActionType.WAIT)

        tree.root = ConditionNode(always_true, ActionNode(wait_generator))
        result = tree.execute(decision_context)
        assert result is not None
        assert result.action_type == ActionType.WAIT


class TestActionGenerator:
    """Test action generation"""

    def test_generate_movement_actions(
            self, sample_agent, sample_percepts) -> None:
        """Test movement action generation"""
        generator = ActionGenerator()
        actions = generator.generate_actions(sample_agent, sample_percepts)
        movement_actions = [
            a for a in actions if a.action_type == ActionType.MOVE]
        assert len(movement_actions) > 0
        explore_actions = [
            a for a in actions if a.action_type == ActionType.EXPLORE]
        assert len(explore_actions) == 1
        flee_actions = [a for a in actions if a.action_type == ActionType.FLEE]
        assert len(flee_actions) == 1

    def test_generate_interaction_actions(
            self, sample_agent, sample_percepts) -> None:
        """Test interaction action generation"""
        generator = ActionGenerator()
        actions = generator.generate_actions(sample_agent, sample_percepts)
        interact_actions = [
            a for a in actions if a.action_type == ActionType.INTERACT]
        assert len(interact_actions) > 0

    def test_generate_communication_actions(
            self, sample_agent, sample_percepts) -> None:
        """Test communication action generation"""
        generator = ActionGenerator()
        actions = generator.generate_actions(sample_agent, sample_percepts)
        comm_actions = [
            a for a in actions if a.action_type == ActionType.COMMUNICATE]
        assert len(comm_actions) > 0

    def test_generate_resource_actions(
            self, sample_agent, sample_percepts) -> None:
        """Test resource action generation"""
        generator = ActionGenerator()
        sample_percepts[2].distance = 1.5
        actions = generator.generate_actions(sample_agent, sample_percepts)
        gather_actions = [
            a for a in actions if a.action_type == ActionType.GATHER]
        assert len(gather_actions) > 0

    def test_capability_based_generation(
            self, sample_agent, sample_percepts) -> None:
        """Test that actions respect agent capabilities"""
        generator = ActionGenerator()
        sample_agent.remove_capability(AgentCapability.MOVEMENT)
        actions = generator.generate_actions(sample_agent, sample_percepts)
        movement_actions = [
            a
            for a in actions
            if a.action_type in [ActionType.MOVE, ActionType.EXPLORE, ActionType.FLEE]
        ]
        assert len(movement_actions) == 0
        wait_actions = [a for a in actions if a.action_type == ActionType.WAIT]
        assert len(wait_actions) == 1


class TestDecisionSystem:
    """Test integrated decision system"""

    @pytest.fixture
    def decision_system(self):
        """Create decision system with mocked dependencies"""
        state_manager = Mock(spec=AgentStateManager)
        perception_system = Mock(spec=PerceptionSystem)
        movement_controller = Mock(spec=MovementController)
        return DecisionSystem(
            state_manager,
            perception_system,
            movement_controller)

    def test_register_agent(self, decision_system, sample_agent) -> None:
        """Test agent registration"""
        decision_system.register_agent(sample_agent)
        assert sample_agent.agent_id in decision_system.decision_makers
        assert sample_agent.agent_id in decision_system.behavior_trees

    def test_make_decision(
            self,
            decision_system,
            sample_agent,
            sample_percepts) -> None:
        """Test making a decision for an agent"""
        decision_system.state_manager.get_agent.return_value = sample_agent
        decision_system.perception_system.perceive.return_value = sample_percepts
        decision_system.register_agent(sample_agent)
        decision = decision_system.make_decision(sample_agent.agent_id)
        assert decision is not None
        assert isinstance(decision, Action)
        history = decision_system.get_decision_history(sample_agent.agent_id)
        assert len(history) == 1

    def test_execute_action_move(self, decision_system, sample_agent) -> None:
        """Test executing a move action"""
        decision_system.state_manager.get_agent.return_value = sample_agent
        decision_system.movement_controller.set_destination.return_value = True
        move_action = Action(
            ActionType.MOVE, target=Position(
                10.0, 10.0, 0.0), cost=5.0)
        success = decision_system.execute_action(
            sample_agent.agent_id, move_action)
        assert success
        decision_system.movement_controller.set_destination.assert_called_once()
        decision_system.state_manager.update_agent_resources.assert_called_with(
            sample_agent.agent_id, energy_delta=-5.0)

    def test_execute_action_wait(self, decision_system, sample_agent) -> None:
        """Test executing a wait action"""
        decision_system.state_manager.get_agent.return_value = sample_agent
        wait_action = Action(ActionType.WAIT, cost=0.1)
        success = decision_system.execute_action(
            sample_agent.agent_id, wait_action)
        assert success
        decision_system.state_manager.update_agent_status.assert_called_with(
            sample_agent.agent_id, AgentStatus.IDLE
        )

    def test_execute_action_flee(
            self,
            decision_system,
            sample_agent,
            sample_percepts) -> None:
        """Test executing a flee action"""
        decision_system.state_manager.get_agent.return_value = sample_agent
        decision_system.perception_system.perceive.return_value = [
            sample_percepts[0]]
        decision_system.movement_controller.set_destination.return_value = True
        flee_action = Action(ActionType.FLEE, cost=10.0)
        success = decision_system.execute_action(
            sample_agent.agent_id, flee_action)
        assert success
        decision_system.movement_controller.set_destination.assert_called_once()
        call_args = decision_system.movement_controller.set_destination.call_args
        flee_position = call_args[0][1]
        assert flee_position.x < sample_agent.position.x
        assert flee_position.y < sample_agent.position.y

    def test_execute_action_with_effects(
            self, decision_system, sample_agent) -> None:
        """Test executing action with effects"""
        decision_system.state_manager.get_agent.return_value = sample_agent
        gather_action = Action(
            ActionType.WAIT, cost=2.0, effects={
                "energy": 20.0})
        success = decision_system.execute_action(
            sample_agent.agent_id, gather_action)
        assert success
        calls = decision_system.state_manager.update_agent_resources.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["energy_delta"] == -2.0
        assert calls[1][1]["energy_delta"] == 20.0

    def test_execute_action_insufficient_energy(
            self, decision_system, sample_agent) -> None:
        """Test executing action with insufficient energy"""
        sample_agent.resources.energy = 5.0
        decision_system.state_manager.get_agent.return_value = sample_agent
        expensive_action = Action(ActionType.MOVE, cost=10.0)
        success = decision_system.execute_action(
            sample_agent.agent_id, expensive_action)
        assert not success
        decision_system.state_manager.update_agent_resources.assert_not_called()

    def test_set_decision_strategy(
            self,
            decision_system,
            sample_agent) -> None:
        """Test changing decision strategy"""
        decision_system.register_agent(sample_agent)
        decision_system.set_decision_strategy(
            sample_agent.agent_id, DecisionStrategy.GOAL_ORIENTED)
        maker = decision_system.decision_makers[sample_agent.agent_id]
        assert maker.strategy == DecisionStrategy.GOAL_ORIENTED

    def test_time_pressure_calculation(
            self, decision_system, sample_agent) -> None:
        """Test time pressure calculation"""
        sample_agent.resources.energy = 20.0
        sample_agent.resources.health = 30.0
        pressure = decision_system._calculate_time_pressure(sample_agent)
        assert pressure > 0.3
        assert pressure <= 1.0
        sample_agent.resources.energy = 100.0
        sample_agent.resources.health = 100.0
        pressure = decision_system._calculate_time_pressure(sample_agent)
        assert pressure < 0.5
