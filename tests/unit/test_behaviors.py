"""
Comprehensive tests for Agent Behavior System.
"""

import pytest
import random
import math
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from agents.base.behaviors import (
    BehaviorPriority,
    BehaviorStatus,
    BaseBehavior,
    IdleBehavior,
    WanderBehavior,
    GoalSeekingBehavior,
    SocialInteractionBehavior,
    ExplorationBehavior,
    BehaviorTreeManager
)
from agents.base.data_model import (
    Agent as AgentData,
    AgentCapability,
    AgentStatus,
    Position,
    SocialRelationship,
    AgentGoal,
    AgentResources
)
from agents.base.decision_making import Action, ActionType


class TestBehaviorPriority:
    """Test BehaviorPriority enum."""
    
    def test_priority_values(self):
        """Test priority enum values."""
        assert BehaviorPriority.CRITICAL.value == 1.0
        assert BehaviorPriority.HIGH.value == 0.8
        assert BehaviorPriority.MEDIUM.value == 0.6
        assert BehaviorPriority.LOW.value == 0.4
        assert BehaviorPriority.BACKGROUND.value == 0.2
    
    def test_priority_ordering(self):
        """Test priority ordering."""
        priorities = [
            BehaviorPriority.BACKGROUND,
            BehaviorPriority.LOW,
            BehaviorPriority.MEDIUM,
            BehaviorPriority.HIGH,
            BehaviorPriority.CRITICAL
        ]
        
        # Should be sorted from lowest to highest value
        sorted_priorities = sorted(priorities, key=lambda p: p.value)
        assert sorted_priorities == priorities


class TestBehaviorStatus:
    """Test BehaviorStatus enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert BehaviorStatus.SUCCESS.value == "success"
        assert BehaviorStatus.FAILURE.value == "failure"
        assert BehaviorStatus.RUNNING.value == "running"
        assert BehaviorStatus.READY.value == "ready"


class TestBaseBehavior:
    """Test BaseBehavior class."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing."""
        agent = Mock(spec=AgentData)
        agent.agent_id = "test-agent"
        agent.position = Position(0.0, 0.0, 0.0)
        agent.status = AgentStatus.IDLE
        agent.resources = AgentResources()
        agent.metadata = {}
        
        # Mock has_capability method
        def has_capability(cap):
            return cap in {AgentCapability.MOVEMENT, AgentCapability.PERCEPTION}
        agent.has_capability = has_capability
        
        return agent
    
    @pytest.fixture
    def test_context(self):
        """Create test context."""
        return {"world_interface": Mock(), "timestamp": datetime.now()}
    
    @pytest.fixture
    def base_behavior(self):
        """Create base behavior for testing."""
        return BaseBehavior(
            name="TestBehavior",
            priority=BehaviorPriority.MEDIUM,
            required_capabilities={AgentCapability.MOVEMENT}
        )
    
    def test_behavior_initialization(self, base_behavior):
        """Test behavior initialization."""
        assert base_behavior.name == "TestBehavior"
        assert base_behavior.priority == BehaviorPriority.MEDIUM
        assert base_behavior.required_capabilities == {AgentCapability.MOVEMENT}
        assert base_behavior.status == BehaviorStatus.READY
        assert base_behavior.last_execution_time is None
        assert base_behavior.cooldown_time == timedelta(seconds=0)
    
    def test_behavior_with_cooldown(self):
        """Test behavior with cooldown period."""
        behavior = BaseBehavior(
            name="CooldownBehavior",
            cooldown_time=timedelta(seconds=5)
        )
        assert behavior.cooldown_time == timedelta(seconds=5)
    
    def test_can_execute_with_capabilities(self, base_behavior, mock_agent, test_context):
        """Test can_execute with sufficient capabilities."""
        result = base_behavior.can_execute(mock_agent, test_context)
        assert result == True
    
    def test_can_execute_without_capabilities(self, base_behavior, mock_agent, test_context):
        """Test can_execute without required capabilities."""
        # Mock agent without movement capability
        def has_capability(cap):
            return cap == AgentCapability.PERCEPTION  # Missing MOVEMENT
        mock_agent.has_capability = has_capability
        
        result = base_behavior.can_execute(mock_agent, test_context)
        assert result == False
    
    def test_can_execute_insufficient_energy(self, base_behavior, mock_agent, test_context):
        """Test can_execute with insufficient energy."""
        mock_agent.resources.energy = 0.5  # Low energy
        
        # Mock high energy cost
        with patch.object(base_behavior, 'get_energy_cost', return_value=10.0):
            result = base_behavior.can_execute(mock_agent, test_context)
            assert result == False
    
    def test_can_execute_cooldown_active(self, mock_agent, test_context):
        """Test can_execute during cooldown period."""
        behavior = BaseBehavior(
            name="CooldownTest",
            cooldown_time=timedelta(seconds=10)
        )
        
        # Set recent execution time
        behavior.last_execution_time = datetime.now() - timedelta(seconds=5)
        
        result = behavior.can_execute(mock_agent, test_context)
        assert result == False
    
    def test_can_execute_cooldown_expired(self, mock_agent, test_context):
        """Test can_execute after cooldown expires."""
        behavior = BaseBehavior(
            name="CooldownTest",
            cooldown_time=timedelta(seconds=5)
        )
        
        # Set old execution time
        behavior.last_execution_time = datetime.now() - timedelta(seconds=10)
        
        result = behavior.can_execute(mock_agent, test_context)
        assert result == True
    
    def test_execute_success(self, base_behavior, mock_agent, test_context):
        """Test successful execution."""
        # Mock successful custom execution
        with patch.object(base_behavior, '_execute_custom', return_value={"success": True}):
            result = base_behavior.execute(mock_agent, test_context)
        
        assert result["success"] == True
        assert base_behavior.status == BehaviorStatus.SUCCESS
        assert base_behavior.last_execution_time is not None
    
    def test_execute_failure(self, base_behavior, mock_agent, test_context):
        """Test failed execution."""
        # Mock failed custom execution
        with patch.object(base_behavior, '_execute_custom', return_value={"success": False}):
            result = base_behavior.execute(mock_agent, test_context)
        
        assert result["success"] == False
        assert base_behavior.status == BehaviorStatus.FAILURE
    
    def test_execute_exception_handling(self, base_behavior, mock_agent, test_context):
        """Test exception handling during execution."""
        # Mock custom execution that raises exception
        with patch.object(base_behavior, '_execute_custom', side_effect=Exception("Test error")):
            result = base_behavior.execute(mock_agent, test_context)
        
        assert result["success"] == False
        assert "error" in result
        assert result["error"] == "Test error"
        assert base_behavior.status == BehaviorStatus.FAILURE
    
    def test_energy_consumption(self, base_behavior, mock_agent, test_context):
        """Test energy consumption during execution."""
        initial_energy = mock_agent.resources.energy
        energy_cost = 5.0
        
        with patch.object(base_behavior, 'get_energy_cost', return_value=energy_cost):
            with patch.object(base_behavior, '_execute_custom', return_value={"success": True}):
                base_behavior.execute(mock_agent, test_context)
        
        # Energy should be consumed
        mock_agent.resources.consume_energy.assert_called_with(energy_cost)
    
    def test_get_priority_base(self, base_behavior, mock_agent, test_context):
        """Test base priority calculation."""
        priority = base_behavior.get_priority(mock_agent, test_context)
        
        # Should return base priority value
        assert priority == BehaviorPriority.MEDIUM.value
    
    def test_get_priority_with_modifiers(self, base_behavior, mock_agent, test_context):
        """Test priority with modifiers."""
        with patch.object(base_behavior, '_get_priority_modifier', return_value=1.5):
            with patch.object(base_behavior, '_get_personality_modifier', return_value=0.8):
                priority = base_behavior.get_priority(mock_agent, test_context)
        
        # Should be base * priority_modifier * personality_modifier
        expected = BehaviorPriority.MEDIUM.value * 1.5 * 0.8
        assert abs(priority - expected) < 0.001
    
    def test_get_priority_clamped(self, base_behavior, mock_agent, test_context):
        """Test priority is clamped to [0, 1] range."""
        with patch.object(base_behavior, '_get_priority_modifier', return_value=2.0):
            priority = base_behavior.get_priority(mock_agent, test_context)
        
        # Should be clamped to 1.0
        assert priority <= 1.0
        
        with patch.object(base_behavior, '_get_priority_modifier', return_value=0.1):
            priority = base_behavior.get_priority(mock_agent, test_context)
        
        # Should not go below 0
        assert priority >= 0.0
    
    def test_personality_modifier(self, base_behavior, mock_agent):
        """Test personality-based priority modifier."""
        # Mock personality profile
        mock_profile = Mock()
        mock_profile.get_behavior_modifier.return_value = 1.2
        mock_agent.metadata = {"personality_profile": mock_profile}
        
        modifier = base_behavior._get_personality_modifier(mock_agent)
        assert modifier == 1.2
        
        # Test without personality profile
        mock_agent.metadata = {}
        modifier = base_behavior._get_personality_modifier(mock_agent)
        assert modifier == 1.0
    
    def test_default_energy_cost(self, base_behavior, mock_agent, test_context):
        """Test default energy cost."""
        cost = base_behavior.get_energy_cost(mock_agent, test_context)
        assert cost == 1.0
    
    def test_custom_execution_methods(self, base_behavior, mock_agent, test_context):
        """Test custom execution method defaults."""
        # Default implementations should work
        assert base_behavior._can_execute_custom(mock_agent, test_context) == True
        
        result = base_behavior._execute_custom(mock_agent, test_context)
        assert result["success"] == True
        assert result["behavior"] == "TestBehavior"
        
        modifier = base_behavior._get_priority_modifier(mock_agent, test_context)
        assert modifier == 1.0


class TestIdleBehavior:
    """Test IdleBehavior class."""
    
    @pytest.fixture
    def idle_behavior(self):
        """Create idle behavior for testing."""
        return IdleBehavior()
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing."""
        agent = Mock(spec=AgentData)
        agent.resources = AgentResources()
        return agent
    
    @pytest.fixture
    def test_context(self):
        """Create test context."""
        return {}
    
    def test_idle_initialization(self, idle_behavior):
        """Test idle behavior initialization."""
        assert idle_behavior.name == "idle"
        assert idle_behavior.priority == BehaviorPriority.BACKGROUND
        assert idle_behavior.required_capabilities == set()
    
    def test_idle_can_always_execute(self, idle_behavior, mock_agent, test_context):
        """Test that idle behavior can always execute."""
        result = idle_behavior.can_execute(mock_agent, test_context)
        assert result == True
    
    def test_idle_execute(self, idle_behavior, mock_agent, test_context):
        """Test idle behavior execution."""
        result = idle_behavior._execute_custom(mock_agent, test_context)
        
        assert result["success"] == True
        assert result["action"] == "idle"
        assert result["energy_restored"] == 0.5
        
        # Should restore energy
        mock_agent.resources.restore_energy.assert_called_with(0.5)


class TestWanderBehavior:
    """Test WanderBehavior class."""
    
    @pytest.fixture
    def wander_behavior(self):
        """Create wander behavior for testing."""
        return WanderBehavior()
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing."""
        agent = Mock(spec=AgentData)
        agent.position = Position(0.0, 0.0, 0.0)
        agent.status = AgentStatus.IDLE
        agent.resources = AgentResources()
        
        def has_capability(cap):
            return cap == AgentCapability.MOVEMENT
        agent.has_capability = has_capability
        
        return agent
    
    @pytest.fixture
    def test_context(self):
        """Create test context with world interface."""
        world_interface = Mock()
        world_interface.can_move_to.return_value = True
        return {"world_interface": world_interface}
    
    def test_wander_initialization(self, wander_behavior):
        """Test wander behavior initialization."""
        assert wander_behavior.name == "wander"
        assert wander_behavior.priority == BehaviorPriority.LOW
        assert AgentCapability.MOVEMENT in wander_behavior.required_capabilities
        assert wander_behavior.cooldown_time == timedelta(seconds=2)
        assert wander_behavior.max_wander_distance == 5.0
    
    def test_wander_can_execute_valid_status(self, wander_behavior, mock_agent, test_context):
        """Test wander can execute with valid status."""
        mock_agent.status = AgentStatus.IDLE
        result = wander_behavior._can_execute_custom(mock_agent, test_context)
        assert result == True
        
        mock_agent.status = AgentStatus.MOVING
        result = wander_behavior._can_execute_custom(mock_agent, test_context)
        assert result == True
    
    def test_wander_cannot_execute_invalid_status(self, wander_behavior, mock_agent, test_context):
        """Test wander cannot execute with invalid status."""
        mock_agent.status = AgentStatus.INTERACTING
        result = wander_behavior._can_execute_custom(mock_agent, test_context)
        assert result == False
    
    def test_wander_execute_success(self, wander_behavior, mock_agent, test_context):
        """Test successful wander execution."""
        with patch('agents.base.behaviors.random.uniform') as mock_uniform:
            mock_uniform.side_effect = [math.pi/4, 3.0]  # angle, distance
            
            result = wander_behavior._execute_custom(mock_agent, test_context)
        
        assert result["success"] == True
        assert "action" in result
        assert "target_position" in result
        
        # Check the action is correct type
        action = result["action"]
        assert hasattr(action, 'action_type')
        
        # Check target position is within wander distance
        target_pos = result["target_position"]
        distance = mock_agent.position.distance_to(target_pos)
        assert distance <= wander_behavior.max_wander_distance
    
    def test_wander_execute_invalid_position(self, wander_behavior, mock_agent, test_context):
        """Test wander execution with invalid position."""
        # Mock world interface to reject movement
        test_context["world_interface"].can_move_to.return_value = False
        
        result = wander_behavior._execute_custom(mock_agent, test_context)
        
        assert result["success"] == False
        assert result["reason"] == "invalid_position"
    
    def test_wander_execute_without_world_interface(self, wander_behavior, mock_agent):
        """Test wander execution without world interface."""
        context = {}  # No world interface
        
        result = wander_behavior._execute_custom(mock_agent, context)
        
        # Should still succeed without world validation
        assert result["success"] == True
    
    def test_wander_energy_cost(self, wander_behavior, mock_agent, test_context):
        """Test wander energy cost."""
        cost = wander_behavior.get_energy_cost(mock_agent, test_context)
        assert cost == 2.0


class TestGoalSeekingBehavior:
    """Test GoalSeekingBehavior class."""
    
    @pytest.fixture
    def goal_behavior(self):
        """Create goal seeking behavior for testing."""
        return GoalSeekingBehavior()
    
    @pytest.fixture
    def mock_agent_with_goal(self):
        """Create mock agent with goal."""
        agent = Mock(spec=AgentData)
        agent.position = Position(0.0, 0.0, 0.0)
        agent.resources = AgentResources()
        
        # Create a goal
        goal = Mock(spec=AgentGoal)
        goal.target_position = Position(10.0, 10.0, 0.0)
        goal.completed = False
        
        agent.current_goal = goal
        agent.goals = [goal]
        
        def has_capability(cap):
            return cap in {AgentCapability.MOVEMENT, AgentCapability.PLANNING}
        agent.has_capability = has_capability
        
        return agent
    
    @pytest.fixture
    def mock_agent_no_goal(self):
        """Create mock agent without goal."""
        agent = Mock(spec=AgentData)
        agent.position = Position(0.0, 0.0, 0.0)
        agent.resources = AgentResources()
        agent.current_goal = None
        agent.goals = []
        
        def has_capability(cap):
            return cap in {AgentCapability.MOVEMENT, AgentCapability.PLANNING}
        agent.has_capability = has_capability
        
        return agent
    
    def test_goal_seeking_initialization(self, goal_behavior):
        """Test goal seeking behavior initialization."""
        assert goal_behavior.name == "goal_seeking"
        assert goal_behavior.priority == BehaviorPriority.HIGH
        assert AgentCapability.MOVEMENT in goal_behavior.required_capabilities
        assert AgentCapability.PLANNING in goal_behavior.required_capabilities
    
    def test_goal_seeking_can_execute_with_goal(self, goal_behavior, mock_agent_with_goal):
        """Test can execute with active goal."""
        context = {}
        result = goal_behavior._can_execute_custom(mock_agent_with_goal, context)
        assert result == True
    
    def test_goal_seeking_cannot_execute_without_goal(self, goal_behavior, mock_agent_no_goal):
        """Test cannot execute without goal."""
        context = {}
        result = goal_behavior._can_execute_custom(mock_agent_no_goal, context)
        assert result == False
    
    def test_goal_seeking_execute_with_goal(self, goal_behavior, mock_agent_with_goal):
        """Test execution with goal."""
        context = {}
        
        # Mock pathfinding or movement logic
        with patch('agents.base.behaviors.Action') as mock_action_class:
            mock_action = Mock()
            mock_action_class.return_value = mock_action
            
            result = goal_behavior._execute_custom(mock_agent_with_goal, context)
        
        assert result["success"] == True
        assert "action" in result
    
    def test_goal_seeking_execute_without_goal(self, goal_behavior, mock_agent_no_goal):
        """Test execution without goal."""
        context = {}
        result = goal_behavior._execute_custom(mock_agent_no_goal, context)
        
        assert result["success"] == False
        assert "no_active_goal" in result["reason"]


class TestSocialInteractionBehavior:
    """Test SocialInteractionBehavior class."""
    
    @pytest.fixture
    def social_behavior(self):
        """Create social interaction behavior."""
        return SocialInteractionBehavior()
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent with social capabilities."""
        agent = Mock(spec=AgentData)
        agent.agent_id = "agent-1"
        agent.position = Position(0.0, 0.0, 0.0)
        agent.resources = AgentResources()
        agent.relationships = {}
        
        def has_capability(cap):
            return cap in {AgentCapability.COMMUNICATION, AgentCapability.SOCIAL_INTERACTION}
        agent.has_capability = has_capability
        
        return agent
    
    def test_social_initialization(self, social_behavior):
        """Test social interaction behavior initialization."""
        assert social_behavior.name == "social_interaction"
        assert social_behavior.priority == BehaviorPriority.MEDIUM
        assert AgentCapability.COMMUNICATION in social_behavior.required_capabilities
        assert AgentCapability.SOCIAL_INTERACTION in social_behavior.required_capabilities
    
    def test_social_can_execute_with_nearby_agents(self, social_behavior, mock_agent):
        """Test can execute with nearby agents."""
        context = {
            "nearby_agents": [
                {"agent_id": "agent-2", "distance": 2.0},
                {"agent_id": "agent-3", "distance": 3.0}
            ]
        }
        
        result = social_behavior._can_execute_custom(mock_agent, context)
        assert result == True
    
    def test_social_cannot_execute_without_nearby_agents(self, social_behavior, mock_agent):
        """Test cannot execute without nearby agents."""
        context = {"nearby_agents": []}
        
        result = social_behavior._can_execute_custom(mock_agent, context)
        assert result == False
    
    def test_social_execute_with_agents(self, social_behavior, mock_agent):
        """Test social interaction execution."""
        context = {
            "nearby_agents": [
                {"agent_id": "agent-2", "distance": 2.0}
            ]
        }
        
        result = social_behavior._execute_custom(mock_agent, context)
        
        assert result["success"] == True
        assert "interaction_target" in result


class TestExplorationBehavior:
    """Test ExplorationBehavior class."""
    
    @pytest.fixture
    def exploration_behavior(self):
        """Create exploration behavior."""
        return ExplorationBehavior()
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent with exploration capabilities."""
        agent = Mock(spec=AgentData)
        agent.agent_id = "explorer-1"
        agent.position = Position(0.0, 0.0, 0.0)
        agent.resources = AgentResources()
        agent.metadata = {"explored_areas": set()}
        
        def has_capability(cap):
            return cap in {
                AgentCapability.MOVEMENT,
                AgentCapability.PERCEPTION,
                AgentCapability.PLANNING
            }
        agent.has_capability = has_capability
        
        return agent
    
    def test_exploration_initialization(self, exploration_behavior):
        """Test exploration behavior initialization."""
        assert exploration_behavior.name == "exploration"
        assert exploration_behavior.priority == BehaviorPriority.MEDIUM
        assert AgentCapability.MOVEMENT in exploration_behavior.required_capabilities
        assert AgentCapability.PERCEPTION in exploration_behavior.required_capabilities
        assert exploration_behavior.exploration_radius == 10.0
    
    def test_exploration_can_execute(self, exploration_behavior, mock_agent):
        """Test exploration can execute."""
        context = {"unexplored_areas": True}
        result = exploration_behavior._can_execute_custom(mock_agent, context)
        assert result == True
    
    def test_exploration_execute(self, exploration_behavior, mock_agent):
        """Test exploration execution."""
        context = {
            "world_interface": Mock(),
            "unexplored_areas": True
        }
        
        result = exploration_behavior._execute_custom(mock_agent, context)
        
        assert result["success"] == True
        assert "exploration_target" in result


class TestBehaviorTreeManager:
    """Test BehaviorTreeManager class."""
    
    @pytest.fixture
    def behavior_manager(self):
        """Create behavior tree manager."""
        return BehaviorTreeManager()
    
    @pytest.fixture
    def sample_behaviors(self):
        """Create sample behaviors."""
        return [
            IdleBehavior(),
            WanderBehavior(),
            GoalSeekingBehavior(),
            SocialInteractionBehavior(),
            ExplorationBehavior()
        ]
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent with all capabilities."""
        agent = Mock(spec=AgentData)
        agent.agent_id = "managed-agent"
        agent.position = Position(0.0, 0.0, 0.0)
        agent.status = AgentStatus.IDLE
        agent.resources = AgentResources()
        agent.current_goal = None
        agent.goals = []
        agent.metadata = {}
        
        def has_capability(cap):
            return True  # All capabilities
        agent.has_capability = has_capability
        
        return agent
    
    def test_manager_initialization(self, behavior_manager):
        """Test behavior manager initialization."""
        assert hasattr(behavior_manager, 'behaviors')
        assert hasattr(behavior_manager, 'current_behavior')
        assert behavior_manager.current_behavior is None
    
    def test_add_behavior(self, behavior_manager, sample_behaviors):
        """Test adding behaviors to manager."""
        for behavior in sample_behaviors:
            behavior_manager.add_behavior(behavior)
        
        # Should have all behaviors
        assert len(behavior_manager.behaviors) >= len(sample_behaviors)
    
    def test_execute_behavior_tree(self, behavior_manager, sample_behaviors, mock_agent):
        """Test executing the behavior tree."""
        # Add behaviors
        for behavior in sample_behaviors:
            behavior_manager.add_behavior(behavior)
        
        context = {}
        
        # Execute behavior tree
        result = behavior_manager.execute(mock_agent, context)
        
        # Should return a valid result
        assert result is not None


class TestBehaviorIntegration:
    """Integration tests for behavior system."""
    
    def test_complete_behavior_workflow(self):
        """Test complete behavior execution workflow."""
        # Create agent
        agent = Mock(spec=AgentData)
        agent.agent_id = "integration-test"
        agent.position = Position(0.0, 0.0, 0.0)
        agent.status = AgentStatus.IDLE
        agent.resources = AgentResources()
        agent.resources.energy = 100.0
        agent.metadata = {}
        agent.current_goal = None
        agent.goals = []
        
        def has_capability(cap):
            return cap in {
                AgentCapability.MOVEMENT,
                AgentCapability.PERCEPTION,
                AgentCapability.COMMUNICATION
            }
        agent.has_capability = has_capability
        
        # Create behavior manager
        manager = BehaviorTreeManager()
        manager.add_behavior(IdleBehavior())
        manager.add_behavior(WanderBehavior())
        
        # Execute multiple steps
        context = {}
        for _ in range(3):
            result = manager.execute(agent, context)
            # Should successfully execute some behavior
            assert result is not None
    
    def test_behavior_energy_management(self):
        """Test behavior energy consumption and management."""
        agent = Mock(spec=AgentData)
        agent.resources = AgentResources()
        agent.resources.energy = 10.0  # Low energy
        
        def has_capability(cap):
            return True
        agent.has_capability = has_capability
        
        # High energy cost behavior
        behavior = WanderBehavior()  # Costs 2.0 energy
        
        context = {}
        
        # Should be able to execute with sufficient energy
        can_execute = behavior.can_execute(agent, context)
        assert can_execute == True
        
        # Set very low energy
        agent.resources.energy = 0.5
        can_execute = behavior.can_execute(agent, context)
        assert can_execute == False  # Insufficient energy
    
    def test_behavior_cooldown_system(self):
        """Test behavior cooldown timing."""
        agent = Mock(spec=AgentData)
        agent.resources = AgentResources()
        
        def has_capability(cap):
            return True
        agent.has_capability = has_capability
        
        # Behavior with cooldown
        behavior = WanderBehavior()  # Has 2 second cooldown
        context = {}
        
        # Should be able to execute initially
        assert behavior.can_execute(agent, context) == True
        
        # Execute behavior
        with patch.object(behavior, '_execute_custom', return_value={"success": True}):
            behavior.execute(agent, context)
        
        # Should not be able to execute immediately due to cooldown
        assert behavior.can_execute(agent, context) == False
        
        # Mock time passage
        behavior.last_execution_time = datetime.now() - timedelta(seconds=10)
        assert behavior.can_execute(agent, context) == True